#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : train_multimodal.py
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
# script for training multimodal benchmarks

import argparse
import math
import os
import warnings
from datetime import datetime
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import transformers
from transformers import AutoConfig
from tqdm import tqdm
import numpy as np
from sklearn.utils import class_weight
import wandb

from models.mmbt import MMBT
from data.datacollator import PadCollate
from data.datareader import read_jsonl
from data.dataset import ToxDataset
from utils.metrics import acc_pre_rec
from utils.weighted_loss import classweight

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
# disable massive warning from huggingface tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
        default="data/toxic_data/train.jsonl",
        help="Path to the jsonl containing the training data",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default="data/toxic_data/dev.jsonl",
        help="Path to the jsonl containing the val data",
    )
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")

    now = datetime.now()
    parser.add_argument(
        "--save_dir",
        type=str,
        default=now.strftime("%m_%d__%H_%M%S"),
        help="Dir to save weights",
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
        "--output_dropout_prob",
        type=float,
        default=0.1,
        help="Dropout prob when classifying transformer output",
    )
    parser.add_argument(
        "--hidden_dropout_prob",
        type=float,
        default=0.1,
        help="Dropout prob on linear layers of transformer",
    )
    parser.add_argument(
        "--attention_probs_dropout_prob",
        type=float,
        default=0.1,
        help="Dropout prob on tokens in attention layers",
    )

    args = parser.parse_args()
    wandb.init(project="textonly", name=args.huggingface_model)
    wandb.config.update(args)
    os.mkdir(args.save_dir)

    traindir = args.save_dir + "/train"
    valdir = args.save_dir + "/val"

    os.mkdir(traindir)
    os.mkdir(valdir)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = MMBT(
        args.huggingface_model,
        output_dropout_prob=args.output_dropout_prob,
        hidden_dropout_prob=args.hidden_dropout_prob,
        attention_probs_dropout_prob=args.attention_probs_dropout_prob,
    ).to(device)

    wandb.watch(model)
    # optimizer taken from huggingface defaults
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-5, betas=(0.9, 0.999), eps=1e-8
    )

    transform = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()]
    )
    dataset_train = ToxDataset(
        read_jsonl(args.train_data, text_only=0),
        text_only=0,
        tokenizer=args.huggingface_model,
        transforms=transform,
    )
    dataloader_train = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=PadCollate()
    )
    dataset_val = ToxDataset(
        read_jsonl(args.val_data, text_only=0),
        text_only=0,
        tokenizer=args.huggingface_model,
        transforms=transform,
    )
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, collate_fn=PadCollate())

    model.save_model(args.save_dir)
    dataset_train.tokenizer.save_pretrained(args.save_dir)

    loss = nn.BCELoss(reduction="none")

    best_val_loss = math.inf
    best_train_loss = math.inf  # temporary, this has to fixed

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
                batch["image"].to(device).detach(),
                batch["embeddings"].to(device).detach(),
                batch["attn"].squeeze(1).to(device).detach(),
            )

            pred_loss = loss(preds, batch["label"].to(device).detach())
            neg_example_loss_weight, pos_example_loss_weight = classweight(
                batch["label"].flatten(), num_classes=2
            )

            pred_loss[batch["label"].to(device) < 0.5] *= neg_example_loss_weight
            pred_loss[batch["label"].to(device) >= 0.5] *= pos_example_loss_weight

            pred_loss = pred_loss.mean()
            pred_loss.backward()
            train_loss += pred_loss.item()

            metrics = acc_pre_rec(preds.reshape(-1, 1), batch["label"].to(device).reshape(-1, 1))
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
            model.transformer.save_pretrained(traindir)

        # validate
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            val_acc = 0.0
            val_prec = 0.0
            val_rec = 0.0
            for idx, batch in tqdm(enumerate(dataloader_val)):
                preds = model.forward(
                    batch["image"].to(device).detach(),
                    batch["embeddings"].to(device).detach(),
                    batch["attn"].squeeze(1).to(device).detach(),
                )

                val_pred_loss = loss(preds, batch["label"].to(device))
                #                val_pred_loss[
                #                    batch["label"].to(device) == 0
                #                ] *= args.neg_example_loss_weight
                #                val_pred_loss[
                #                    batch["label"].to(device) == 1
                #                ] *= args.pos_example_loss_weight
                #                val_pred_loss = val_pred_loss.mean()

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
            wandb.log(
                {
                    "Val loss": val_loss,
                    "Val accuracy": val_acc,
                    "Val precision": val_prec,
                    "Val recall": val_rec,
                    "Train loss": train_loss,
                    "Train accuracy": train_acc,
                    "Train precision": train_prec,
                    "Train recall": train_rec,
                }
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_model(valdir)
