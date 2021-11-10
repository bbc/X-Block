#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : train_multitask.py
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
# primary scripts for training multistream, multitask models

import argparse
import math
import os
import time
import warnings
import logging


from models.multitask import MultitaskModel
from data.datareader import read_jsonl_alt
from data.dataset import ToxDataset
from utils.metrics import acc_pre_rec
from utils.weighted_loss import classweight

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import transformers
from transformers import AutoConfig
from tqdm import tqdm

transformers.logger.setLevel(transformers.logging.ERROR)
warnings.filterwarnings("ignore")
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--huggingface_model",
        type=str,
        default="bart-large",
        help="The name of the huggingface model to use",
    )
    parser.add_argument(
        "--feature_dim",
        type=int,
        default=200,
        help="Dimension input features are reduced to before combining",
    )
    parser.add_argument(
        "--combine_dim",
        type=int,
        default=800,
        help="Hidden size in combination of features",
    )
    parser.add_argument(
        "--large_cnn",
        type=int,
        default=0,
        help="Whether to use the large CNN (ResNet152) in model, or small (ResNet18)",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=-1,
        help="Max sequence length to use in transformer, use -1 to default to transformer maximum",
    )
    parser.add_argument(
        "--hidden_dropout_prob",
        type=float,
        default=0.1,
        help="Dropout prob in linear layers of transformer",
    )
    parser.add_argument(
        "--attention_probs_dropout_prob",
        type=float,
        default=0.1,
        help="Dropout prob on tokens in attention in transformer",
    )
    parser.add_argument(
        "--projection_dropout",
        type=float,
        default=0.1,
        help="Dropout prob in projection/feature combination",
    )
    parser.add_argument(
        "--train_cnn", type=int, default=1, help="Whether to train the CNN"
    )
    parser.add_argument(
        "--train_transformer",
        type=int,
        default=1,
        help="Whether to train the transformer",
    )
    parser.add_argument(
        "--train_data",
        nargs="+",
        type=str,
        required=True,
        help="A list of paths to the training jsonl files. Pass like: "
        '--train_data "file1.jsonl" "file2.jsonl" "file3.jsonl"',
    )
    parser.add_argument(
        "--val_data",
        nargs="+",
        type=str,
        required=True,
        help="A list of paths to the validation jsonl files. Pass like: "
        '--train_data "file1.jsonl" "file2.jsonl" "file3.jsonl"',
    )
    parser.add_argument(
        "--task_probs",
        nargs="+",
        type=float,
        default=[],
        help="Probabilities of selecting an item from the corresponding dataframe. "
        "Defaults to equal weighting. Pass like: --task_probs 0.5 0.2 0.3",
    )
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.00005, help="Learning Rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--save_dir", type=str, default=str(time.time()), help="Dir to save weights"
    )
    parser.add_argument(
        "--multi_gpu",
        type=int,
        default=0,
        help="Whether to use all GPUs on a machine or just one",
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
    parser.add_argument(
            "fusion approach: attention (0) or concatenation (1)",
            type=int,
            default=1,
            help="type of multimodal fusion - attention or concatenation
            (concatenation preferred)"
    )


    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    transform = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()]
    )
    dataset_train = ToxDataset(
        [read_jsonl_alt(df) for df in args.train_data],
        tokenizer=args.huggingface_model,
        transforms=transform,
        selection_probs=args.task_probs,
        max_sequence_length=args.max_sequence_length
        if args.max_sequence_length != -1
        else None,
    )
    dataloader_train = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True
    )
    dataset_val = ToxDataset(
        [read_jsonl_alt(df) for df in args.val_data],
        tokenizer=args.huggingface_model,
        transforms=transform,
        max_sequence_length=args.max_sequence_length
        if args.max_sequence_length != -1
        else None,
        test=True,
    )
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size)

    task_output_dims = [
        len(df.label.unique()) if len(df.label.unique()) != 2 else 1
        for df in dataset_train.dataframes
    ]

    model = MultitaskModel(
        args.save_dir if os.path.isdir(args.save_dir) else args.huggingface_model,
        feature_dim=args.feature_dim,
        combine_dim=args.combine_dim,
        output_dims=task_output_dims,
        hidden_dropout_prob=args.projection_dropout,
        attention_probs_dropout_prob=args.attention_probs_dropout_prob,
        projection_dropout=args.projection_dropout,
        large_cnn=args.large_cnn,
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
    if not args.train_cnn:
        for parameters in (
            model.module.image_cnn.parameters()
            if args.multi_gpu
            else model.image_cnn.parameters()
        ):
            parameters.requires_grad = False
    if not args.train_transformer:
        for parameters in (
            model.module.transformer.parameters()
            if args.multi_gpu
            else model.transformer.parameters()
        ):
            parameters.requires_grad = False
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    traindir = args.save_dir + "/train"
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
        os.mkdir(traindir)
    else:
        # resume from where we left off
        if args.multi_gpu:
            model.module.load_model(traindir)
        else:
            model.load_model(traindir)

    if args.multi_gpu:
        model.module.save_model(args.save_dir)
    else:
        model.save_model(args.save_dir)
    dataset_train.tokenizer.save_pretrained(args.save_dir)

    bce_loss = nn.BCELoss(reduction="none")
    ce_loss = nn.CrossEntropyLoss(reduction="none")

    best_val_loss = math.inf

    for epoch in range(args.num_epochs):
        model.train()
        print(f"Epoch: {epoch}")
        # seperate losses and metrics on per task basis
        train_loss = 0.0
        train_acc = [0.0 for _ in range(len(args.train_data))]
        train_prec = [0.0 for _ in range(len(args.train_data))]
        train_rec = [0.0 for _ in range(len(args.train_data))]
        dloader = tqdm(enumerate(dataloader_train))
        for idx, batch in dloader:
            optimizer.zero_grad()

            batch["task_indices"] = batch["task_indices"].to(device).detach()

            preds = model.forward(
                batch["task_indices"],
                comment=batch["embeddings"].squeeze(1).to(device).detach(),
                comment_attn=batch["attn"].squeeze(1).to(device).detach(),
                title=batch["title"].squeeze(1).to(device).detach(),
                title_attn=batch["title_attn"].squeeze(1).to(device).detach(),
                image_text=batch["img_text"].squeeze(1).to(device).detach(),
                image_text_attn=batch["img_text_attn"].squeeze(1).to(device).detach(),
                image=batch["image"].to(device).detach(),
                modalities=batch["modalities"].to(device).detach(),
            )

            batch["label"] = batch["label"].to(preds.device).detach()

            pred_loss = torch.zeros((len(preds), 1), device=preds.device)
            for task, dim in enumerate(task_output_dims):
                if dim == 1:
                    loss_weights = classweight(
                        batch["label"][batch["task_indices"] == task], num_classes=2
                    )
                    pred_loss[batch["task_indices"] == task] = bce_loss(
                        preds[:, 0][(batch["task_indices"] == task).flatten()],
                        batch["label"][batch["task_indices"] == task].to(preds.device),
                    )
                else:
                    loss_weights = classweight(
                        batch["label"][batch["task_indices"] == task] % dim,
                        num_classes=dim,
                    )
                    pred_loss[batch["task_indices"] == task] = ce_loss(
                        preds[:, :dim][(batch["task_indices"] == task).flatten()],
                        (batch["label"][batch["task_indices"] == task] % dim)
                        .type(torch.long)
                        .to(preds.device),
                    )
                pred_loss[batch["task_indices"] == task] *= loss_weights[
                    batch["label"][batch["task_indices"] == task].type(torch.long) % dim
                ]

            pred_loss_mean = torch.mean(pred_loss)
            pred_loss_mean.backward()
            train_loss += pred_loss_mean.item()
            dloader.set_description("Batch Loss %f" % pred_loss_mean.item())
            optimizer.step()

            if args.multi_gpu:
                model.module.save_model(traindir)
            else:
                model.save_model(traindir)
            for task, dim in enumerate(task_output_dims):
                if (batch["task_indices"] == task).sum().item() == 0:
                    continue
                metrics = acc_pre_rec(
                    preds[batch["task_indices"].flatten() == task].reshape(
                        (batch["task_indices"] == task).sum(), -1
                    )[:, :dim],
                    batch["label"][batch["task_indices"] == task],
                )
                train_acc[task] += metrics["accuracy"]
                train_prec[task] += metrics["precision"]
                train_rec[task] += metrics["recall"]

        train_loss /= len(dataloader_train)
        train_acc = [i / len(dataloader_train) for i in train_acc]
        train_prec = [i / len(dataloader_train) for i in train_prec]
        train_rec = [i / len(dataloader_train) for i in train_rec]

        # validate
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_acc = [0.0 for _ in range(len(args.val_data))]
            val_prec = [0.0 for _ in range(len(args.val_data))]
            val_rec = [0.0 for _ in range(len(args.val_data))]
            for idx, batch in tqdm(enumerate(dataloader_val)):
                batch["task_indices"] = batch["task_indices"].to(device).detach()

                preds = model.forward(
                    batch["task_indices"],
                    comment=batch["embeddings"].squeeze(1).to(device).detach(),
                    comment_attn=batch["attn"].squeeze(1).to(device).detach(),
                    title=batch["title"].squeeze(1).to(device).detach(),
                    title_attn=batch["title_attn"].squeeze(1).to(device).detach(),
                    image_text=batch["img_text"].squeeze(1).to(device).detach(),
                    image_text_attn=batch["img_text_attn"]
                    .squeeze(1)
                    .to(device)
                    .detach(),
                    image=batch["image"].to(device).detach(),
                    modalities=batch["modalities"].to(device).detach(),
                )

                batch["label"] = batch["label"].to(preds.device).detach()

                val_pred_loss = torch.zeros((len(preds), 1), device=device)
                for task, dim in enumerate(task_output_dims):
                    if dim == 1:
                        val_pred_loss[batch["task_indices"] == task] = bce_loss(
                            preds[:, 0][(batch["task_indices"] == task).flatten()],
                            batch["label"][batch["task_indices"] == task].to(
                                preds.device
                            ),
                        )
                    else:
                        val_pred_loss[batch["task_indices"] == task] = ce_loss(
                            preds[:, :dim][batch["task_indices"] == task],
                            (batch["label"][batch["task_indices"] == task] % dim)
                            .type(torch.long)
                            .to(preds.device),
                        )
                val_pred_loss_mean = torch.mean(val_pred_loss)
                val_loss += val_pred_loss_mean.item()

                for task, dim in enumerate(task_output_dims):
                    if (batch["task_indices"] == task).sum().item() == 0:
                        continue
                    metrics = acc_pre_rec(
                        preds[batch["task_indices"].flatten() == task].reshape(
                            (batch["task_indices"] == task).sum(), -1
                        )[:, :dim],
                        batch["label"][batch["task_indices"] == task],
                    )
                    val_acc[task] += metrics["accuracy"]
                    val_prec[task] += metrics["precision"]
                    val_rec[task] += metrics["recall"]

            val_loss /= len(dataloader_val)
            val_acc = [i / len(dataloader_val) for i in val_acc]
            val_prec = [i / len(dataloader_val) for i in val_prec]
            val_rec = [i / len(dataloader_val) for i in val_rec]

            print(f"Val loss: {val_loss}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if args.multi_gpu:
                    model.module.save_model(args.save_dir)
                else:
                    model.save_model(args.save_dir)
