#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : train_text.py
# Author            : Pranava Madhyastha <pranava@imperial.ac.uk>
# Date              : 01.11.2020
# Last Modified Date: 09.11.2021
# Last Modified By  : Pranava Madhyastha <pranava@imperial.ac.uk>
#
# Copyright (c) 2020, Imperial College, London
# All rights reserved.
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#   1. Redistributions of source code must retain the above copyright notice, this
#      list of conditions and the following disclaimer.
#   2. Redistributions in binary form must reproduce the above copyright notice,
#      this list of conditions and the following disclaimer in the documentation
#      and/or other materials provided with the distribution.
#   3. Neither the name of Imperial College nor the names of its contributors may
#      be used to endorse or promote products derived from this software without
#      specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR 
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# text only code base for benchmarking

import argparse
import math
import os
import warnings
from datetime import datetime
import logging
import getpass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import transformers
from transformers import AutoConfig
from tqdm import tqdm
import numpy as np

from models.textonly import TextOnlyClassification
from data.datacollator import PadCollate
from data.datareader import read_jsonl_alt
from data.dataset import ToxDataset
from utils.metrics import acc_pre_rec
from utils.weighted_loss import classweight

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--huggingface_model",
        type=str,
        default="facebook/bart-large",
        help="The name of the model to use",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Path to the jsonl containing the training data",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        required=True,
        help="Path to the jsonl containing the val data",
    )
    
    parser.add_argument('--encrypted',
        action = 'store_true',
        default = False,
        help = 'Indicates that the input files are encrypted using AES-256'
    )
    
    parser.add_argument(
        "--pos_example_loss_weight",
        type=float,
        default=1.0,
        help="Multiplicative weight of the loss for instances of the pos class",
    )
    parser.add_argument(
        "--neg_example_loss_weight",
        type=float,
        default=1.0,
        help="Multiplicative weight of the loss for instances of the neg class",
    )
    parser.add_argument(
        "--hidden_dropout_prob",
        type=float,
        default=0.1,
        help="Dropout prob in fully connceted layers in transformer",
    )
    parser.add_argument(
        "--attention_probs_dropout_prob",
        type=float,
        default=0.1,
        help="Dropout prob for tokens in attention layers",
    )

    now = datetime.now()
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=now.strftime("%m_%d__%H_%M%S"),
        help="Dir to save weights",
    )
    parser.add_argument(
        "--multi_gpu",
        type=int,
        default=0,
        help="Whether to use multiple GPUs on a machine or just one",
    )
    parser.add_argument(
        "--distributed_parallel",
        type=int,
        default=0,
        help="When using multi_gpu, whether to use DistributedDataParallel or DataParallel",
    )
    parser.add_argument(
        "--gpu_ids",
        nargs="+",
        type=int,
        default=[0],
        help="IDs of the GPUs to use. Only used if args.multi_gpu is true",
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="Rank of the node running this script"
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Number of process participating in job",
    )

    args = parser.parse_args()
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    traindir = args.save_dir + "/train"
    valdir = args.save_dir + "/val"
    
    if args.encrypted:
        password = getpass.getpass()
    
    else:
        password = None

    if not os.path.isdir(traindir):
        os.mkdir(traindir)
    if not os.path.isdir(valdir):
        os.mkdir(valdir)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = TextOnlyClassification(
        args.huggingface_model,
        hidden_dropout_prob=args.hidden_dropout_prob,
        attention_probs_dropout_prob=args.attention_probs_dropout_prob,
    ).to(device)
    if args.multi_gpu:
        if args.distributed_parallel:
            torch.distributed.init_process_group(
                "nccl", rank=args.local_rank, world_size=args.world_size
            )
            model = nn.parallel.DistributedDataParallel(model, device_ids=args.gpu_ids)
        else:
            model = nn.DataParallel(model, device_ids=args.gpu_ids)
    # optimizer taken from huggingface defaults
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-5, betas=(0.9, 0.999), eps=1e-8
    )

    dataset_train = ToxDataset(
        read_jsonl_alt(args.train_data, password),
        text_only=1,
        tokenizer=args.huggingface_model,
        read_img=False,
    )
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, collate_fn=PadCollate())
    dataset_val = ToxDataset(
        read_jsonl_alt(args.val_data, password),
        text_only=1,
        tokenizer=args.huggingface_model,
        read_img=False,
    )
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, collate_fn=PadCollate())

    if args.multi_gpu:
        model.module.model.save_pretrained(args.save_dir)
    else:
        model.model.save_pretrained(args.save_dir)
    dataset_train.tokenizer.save_pretrained(args.save_dir)

    loss = nn.BCELoss(reduction="none")

    best_val_loss = math.inf
    best_train_loss = math.inf  # temporary, this has to fixed
    """
    Currently the dev set and the test set are disjoined
    This has to be fixed.
    """
    for epoch in range(args.num_epochs):
        model.train()
        print(f"Epoch: {epoch}")

        train_loss = 0.0
        train_acc = 0.0
        train_prec = 0.0
        train_rec = 0.0

        for idx, batch in tqdm(enumerate(dataloader_train)):
            if not idx % 100:
                print(f"Step: {idx}/{len(dataloader_train)}")
            optimizer.zero_grad()

            preds = model.forward(
                batch["embeddings"].to(device), batch["attn"].squeeze(1).to(device)
            )

            pred_loss = loss(preds, batch["label"].to(device))

            loss_weight = classweight(batch["label"].flatten(), num_classes=2)
            pred_loss[batch["label"].to(device) < 0.5] *= loss_weight[0]
            pred_loss[batch["label"].to(device) >= 0.5] *= loss_weight[1]
            pred_loss = pred_loss.mean()
            pred_loss.backward()

            train_loss += pred_loss.item()

            metrics = acc_pre_rec(
                preds.reshape(-1, 1), batch["label"].to(device).reshape(-1, 1)
            )

            train_acc += metrics["accuracy"]
            train_prec += metrics["precision"]
            train_rec += metrics["recall"]

            optimizer.step()

        train_loss /= idx + 1.0
        train_acc /= idx + 1.0
        train_prec /= idx + 1.0
        train_rec /= idx + 1.0

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            if args.multi_gpu:
                model.module.model.save_pretrained(args.save_dir)
            else:
                model.model.save_pretrained(args.save_dir)

        # validate
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            val_acc = 0.0
            val_prec = 0.0
            val_rec = 0.0

            for idx, batch in tqdm(enumerate(dataloader_val)):
                preds = model.forward(
                    batch["embeddings"].to(device), batch["attn"].squeeze(1).to(device)
                )
                val_pred_loss = loss(preds, batch["label"].to(device))

                val_pred_loss = val_pred_loss.mean()

                val_loss += val_pred_loss.item()
                metrics = acc_pre_rec(
                    preds.reshape(-1, 1), batch["label"].to(device).reshape(-1, 1)
                )

                val_acc += metrics["accuracy"]
                val_prec += metrics["precision"]
                val_rec += metrics["recall"]

            val_loss /= idx + 1.0
            val_acc /= idx + 1.0
            val_prec /= idx + 1.0
            val_rec /= idx + 1.0

            print(f"Val loss: {val_loss}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if args.multi_gpu:
                    model.module.model.save_pretrained(args.save_dir)
                else:
                    model.model.save_pretrained(args.save_dir)
