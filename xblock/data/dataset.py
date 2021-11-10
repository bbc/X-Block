#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : dataset.py
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
#
# this library provides support for the primary dataset class

import random

import numpy as np
import pandas as pd
import torch
import cv2
import pytesseract
from PIL import Image
from transformers import AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class ToxDataset(Dataset):
    def __init__(
        self,
        dataframes,
        transforms=None,
        tokenizer="bert-base-uncased",
        text_only=0,
        selection_probs=[],
        max_sequence_length=None,
        test=False,
        language_model=False,
        read_text=True,
        read_title=True,
        read_img=True,
        read_img_text=True,
    ):
        """
        Dataset, supports mono and multitask learning

        Args:
            dataframes: a list of dataframes to use
            transforms: transforms to apply to the images
            tokenizer: name of the huggingface tokenizer to use
            text_only: deprecated, ignored
            selection_probs: list of probabilities of selecting each task
            max_sequence_length: maximum number of tokens in sequence, will default to the max
                number allowed in the specified transformer
            test: whether the dataset is in test mode or train mode
            language_model: whether the model to be trained is a LM, if true and max_seq_len None,
                will pad title and text to half max len
            read_text: whether to read the text from the dataframe
            read_title: whether to read the title from the dataframe
            read_img: whether to read the image (and any text in the image) from the dataframe
        """
        if isinstance(dataframes, pd.DataFrame):
            # then there is a single data source
            self.dataframes = [dataframes]
            self.selection_probs = [1.0]
        else:
            self.dataframes = dataframes
            # if the dataframes and selection probs aren't the same size, then default to equal weighting
            self.selection_probs = (
                selection_probs
                if len(selection_probs) == len(dataframes)
                else [1.0 / len(dataframes) for _ in range(len(dataframes))]
            )
        self.max_sequence_length = (
            AutoConfig.from_pretrained(tokenizer).max_position_embeddings
            if max_sequence_length is None and not language_model
            else AutoConfig.from_pretrained(tokenizer).max_position_embeddings // 2 - 1
            if max_sequence_length is None and language_model
            else max_sequence_length
        )
        if self.max_sequence_length == 514:
            # this fixes distil roberta
            self.max_sequence_length = 512
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        if self.tokenizer.pad_token is None:
            # if using gpt or something
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.pad_token
        )
        self.transforms = transforms
        self.test = test
        if test:
            # if testing we want to check all of the examples, and not massage etc
            examples_per_class = [len(df) for df in self.dataframes]
            self.dataframes = pd.concat(self.dataframes)
            self.dataframes["task_index"] = sum(
                [[task] * num for task, num in enumerate(examples_per_class)], []
            )
        self.read_text = read_text
        self.read_title = read_title
        self.read_img = read_img
        self.read_img_text = read_img_text

    def __len__(self):
        return len(self.dataframes) if self.test else max(map(len, self.dataframes))

    def __getitem__(self, idx):
        if self.test:
            data = self.dataframes.iloc[idx]
            task_indices = data.task_index
        else:
            task_indices = random.choices(
                list(range(len(self.selection_probs))), weights=self.selection_probs
            )[0]
            data = self.dataframes[task_indices].iloc[
                idx % len(self.dataframes[task_indices])
            ]

        # some datasets might not have some modes for some points or the whole dataset,
        # so we keep track of what it does have
        # elements represent text, title, image, label
        modalities = [1, 1, 1, 1]

        if self.read_text and "text" in data and not pd.isna(data.text) and data.text: 
            encoding = torch.cat(
                (
                    self.tokenizer.encode(
                        data.text,
                        max_length=self.max_sequence_length,
                        padding=False,
                        truncation=True,
                        return_tensors="pt",
                    ).flatten(),
                    torch.tensor([self.tokenizer.pad_token_id]),
                )
            )
            if encoding.size() == torch.Size([1, 0]):
                encoding = torch.full((1,), self.pad_token_id, dtype=torch.long)
            mask = torch.zeros(encoding.size(), dtype=torch.float)
            mask[encoding != self.pad_token_id] = 1
        else:
            encoding = torch.zeros(1, dtype=torch.long)
            mask = encoding.clone().type(torch.float)
            modalities[0] = 0

        if (
            self.read_title
            and "title" in data
            and not pd.isna(data.title)
            and data.title
        ):
            title = torch.cat(
                (
                    self.tokenizer.encode(
                        data.title,
                        max_length=self.max_sequence_length,
                        padding=False,
                        truncation=True,
                        return_tensors="pt",
                    ).flatten(),
                    torch.tensor([self.tokenizer.pad_token_id]),
                )
            )
            if title.size() == torch.Size([1, 0]):
                title = torch.full((1,), self.pad_token_id, dtype=torch.long)
            title_mask = torch.zeros(title.size(), dtype=torch.float)
            title_mask[title != self.pad_token_id] = 1
        else:
            title = torch.zeros(1, dtype=torch.long)
            title_mask = title.clone().type(torch.float)
            modalities[1] = 0

        if self.read_img and "img" in data and not pd.isna(data.img) and data.img:
            img = cv2.imread(data.img)
            if self.read_img_text:
                img_text_string = pytesseract.image_to_string(data.img)
                try:
                    img_text = torch.cat(
                        (
                            self.tokenizer.encode(
                                img_text_string,
                                max_length=self.max_sequence_length,
                                padding=False,
                                truncation=True,
                                return_tensors="pt",
                            ).flatten(),
                            torch.tensor([self.tokenizer.pad_token_id]),
                        )
                    )
                    if img_text.size() == torch.Size([1, 0]):
                        img_text = torch.full((1,), self.pad_token_id, dtype=torch.long)
                except pytesseract.pytesseract.TesseractError:
                    # if the image doesn't have dimensions in it's metadata, will throw
                    # an error, this catches
                    img_text = torch.full((1,), self.pad_token_id, dtype=torch.long)
                img_text_mask = torch.zeros(img_text.size(), dtype=torch.float)
                img_text_mask[img_text != self.pad_token_id] = 1
            else:
                img_text = torch.full((1,), self.pad_token_id, dtype=torch.long)
                img_text_mask = torch.zeros(img_text.size(), dtype=torch.float)
                img_text_mask[img_text != self.pad_token_id] = 1
            if self.transforms:
                img = self.transforms(img)
        else:
            img = torch.zeros(3, 1, 1)
            img_text = torch.full((1,), self.pad_token_id, dtype=torch.long)
            img_text_mask = torch.zeros(img_text.size(), dtype=torch.float)
            img_text_mask[img_text != self.pad_token_id] = 1
            if self.transforms:
                img = self.transforms(img)
            modalities[2] = 0

        if "label" in data and not pd.isna(data.label):
            label = [data.label]
        else:
            label = [-1]
            modalities[3] = 0

        return {
            "image": img,
            "img_text": img_text,
            "img_text_attn": img_text_mask,
            "embeddings": encoding,
            "attn": mask,
            "title": title,
            "title_attn": title_mask,
            "label": torch.FloatTensor(label),
            "task_indices": torch.LongTensor([task_indices]),
            "modalities": torch.LongTensor(modalities),
        }
