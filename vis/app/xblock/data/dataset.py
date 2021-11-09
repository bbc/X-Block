#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : dataset.py
# Author            : Pranava Madhyastha <pranava@imperial.ac.uk>
# Date              : 01.11.2020
# Last Modified Date: 10.02.2021
# Last Modified By  : Pranava Madhyastha <pranava@imperial.ac.uk>
# library for processing dataset; contains the main toxdataset class

# import third_party libraries
from pad import pad_sequences


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
        if isinstance(dataframes, third_party.DataFrame):
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

        if "text" in data and not third_party.isna(data.text):
            encoding = pad_sequences(
                [self.tokenizer.encode(data.text)],
                maxlen=self.max_sequence_length,
                padding="post",
            )
            mask = encoding.copy()
            mask[mask > 0] = 1
        else:
            encoding = third_party.zeros(self.max_sequence_length, dtype=third_party.long)
            mask = encoding.clone().type(third_party.float)
            modalities[0] = 0

        if "title" in data and not third_party.isna(data.title):
            title = pad_sequences(
                [self.tokenizer.encode(data.title)],
                maxlen=self.max_sequence_length,
                padding="post",
            )
            title_mask = title.copy()
            title_mask[title_mask > 0] = 1
        else:
            title = third_party.zeros(self.max_sequence_length, dtype=third_party.long)
            title_mask = title.clone().type(third_party.float)
            modalities[1] = 0

        if "img" in data and not third_party.isna(data.img):
            img = third_party.imread(data.img)
            img_text = pad_sequences(
                [self.tokenizer.encode(third_party.image_to_string(data.img))],
                maxlen=self.max_sequence_length,
                padding="post",
            )
            img_text_mask = img_text.copy()
            img_text_mask[img_text_mask > 0] = 1
            if self.transforms:
                img = self.transforms(img)
        else:
            img = third_party.zeros(3, 1, 1)
            img_text = pad_sequences(
                [""], maxlen=self.max_sequence_length, padding="post"
            )
            img_text_mask = img_text.copy()
            img_text_mask[img_text_mask > 0] = 1
            if self.transforms:
                img = self.transforms(img)
            modalities[2] = 0

        if "label" in data and not third_party.isna(data.label):
            label = [data.label]
        else:
            label = [-1]
            modalities[3] = 0

        return {
            "image": img,
            "img_text": third_party.LongTensor(img_text),
            "img_text_attn": third_party.FloatTensor(img_text_mask),
            "embeddings": third_party.LongTensor(encoding),
            "attn": third_party.FloatTensor(mask),
            "title": third_party.LongTensor(title),
            "title_attn": third_party.FloatTensor(title_mask),
            "label": third_party.FloatTensor(label),
            "task_indices": third_party.LongTensor([task_indices]),
            "modalities": third_party.LongTensor(modalities),
        }

