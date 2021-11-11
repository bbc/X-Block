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
# library for processing dataset; contains the main toxdataset class

from .pad import pad_sequences

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
        """
        if isinstance(dataframes, pd.DataFrame):
            # then there is a single data source
            self.dataframes = [dataframes]
            self.selection_probs = [1.0]
        else:
            assert len(dataframes) == len(
                selection_probs
            ), "Must have same number of selection probs as dataframes"
            self.dataframes = dataframes
            self.selection_probs = selection_probs
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_sequence_length = (
            AutoConfig.from_pretrained(tokenizer).max_position_embeddings
            if max_sequence_length is None
            else max_sequence_length
        )
        self.transforms = transforms

    def __len__(self):
        return max(map(len, self.dataframes))

    def __getitem__(self, idx):
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

        if "text" in data and not pd.isna(data.text):
            encoding = pad_sequences(
                [self.tokenizer.encode(data.text)],
                maxlen=self.max_sequence_length,
                padding="post",
            )
            mask = encoding.copy()
            mask[mask > 0] = 1
        else:
            encoding = torch.zeros(self.max_sequence_length, dtype=torch.long)
            mask = encoding.clone().type(torch.float)
            modalities[0] = 0

        if "title" in data and not pd.isna(data.title):
            title = pad_sequences(
                [self.tokenizer.encode(data.title)],
                maxlen=self.max_sequence_length,
                padding="post",
            )
            title_mask = title.copy()
            title_mask[title_mask > 0] = 1
        else:
            title = torch.zeros(self.max_sequence_length, dtype=torch.long)
            title_mask = title.clone().type(torch.float)
            modalities[1] = 0

        if "img" in data and not pd.isna(data.img):
            img = cv2.imread(data.img)
            img_text = pad_sequences(
                [self.tokenizer.encode(pytesseract.image_to_string(data.img))],
                maxlen=self.max_sequence_length,
                padding="post",
            )
            img_text_mask = img_text.copy()
            img_text_mask[img_text_mask > 0] = 1
            if self.transforms:
                img = self.transforms(img)
        else:
            img = torch.zeros(3, 1, 1)
            img_text = pad_sequences(
                [""], maxlen=self.max_sequence_length, padding="post"
            )
            img_text_mask = img_text.copy()
            img_text_mask[img_text_mask > 0] = 1
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
            "img_text": torch.LongTensor(img_text),
            "img_text_attn": torch.FloatTensor(img_text_mask),
            "embeddings": torch.LongTensor(encoding),
            "attn": torch.FloatTensor(mask),
            "title": torch.LongTensor(title),
            "title_attn": torch.FloatTensor(title_mask),
            "label": torch.FloatTensor(label),
            "task_indices": torch.LongTensor([task_indices]),
            "modalities": torch.LongTensor(modalities),
        }

