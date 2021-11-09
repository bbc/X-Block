#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : dataset.py
# Author            : Pranava Madhyastha <pranava@imperial.ac.uk>
# Date              : 01.11.2020
# Last Modified Date: 10.02.2021
# Last Modified By  : Pranava Madhyastha <pranava@imperial.ac.uk>
# this library provides support for the primary dataset class

import random

# import third-party libraries
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
        if isinstance(dataframes, third_party.DataFrame):
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
            third_party.from_pretrained(tokenizer).max_position_embeddings
            if max_sequence_length is None and not language_model
            else third_party.from_pretrained(tokenizer).max_position_embeddings // 2 - 1
            if max_sequence_length is None and language_model
            else max_sequence_length
        )
        if self.max_sequence_length == 514:
            # this fixes distil roberta
            self.max_sequence_length = 512
        self.tokenizer = third_party.from_pretrained(tokenizer)
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
            self.dataframes = third_party.concat(self.dataframes)
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

        if self.read_text and "text" in data and not third_party.isna(data.text) and data.text:
            encoding = third_party.cat(
                (
                    self.tokenizer.encode(
                        data.text,
                        max_length=self.max_sequence_length,
                        padding=False,
                        truncation=True,
                        return_tensors="pt",
                    ).flatten(),
                    third_party.tensor([self.tokenizer.pad_token_id]),
                )
            )
            if encoding.size() == third_party.Size([1, 0]):
                encoding = third_party.full((1,), self.pad_token_id, dtype=third_party.long)
            mask = third_party.zeros(encoding.size(), dtype=third_party.float)
            mask[encoding != self.pad_token_id] = 1
        else:
            encoding = third_party.zeros(1, dtype=third_party.long)
            mask = encoding.clone().type(third_party.float)
            modalities[0] = 0

        if (
            self.read_title
            and "title" in data
            and not third_party.isna(data.title)
            and data.title
        ):
            title = third_party.cat(
                (
                    self.tokenizer.encode(
                        data.title,
                        max_length=self.max_sequence_length,
                        padding=False,
                        truncation=True,
                        return_tensors="pt",
                    ).flatten(),
                    third_party.tensor([self.tokenizer.pad_token_id]),
                )
            )
            if title.size() == third_party.Size([1, 0]):
                title = third_party.full((1,), self.pad_token_id, dtype=third_party.long)
            title_mask = third_party.zeros(title.size(), dtype=third_party.float)
            title_mask[title != self.pad_token_id] = 1
        else:
            title = third_party.zeros(1, dtype=third_party.long)
            title_mask = title.clone().type(third_party.float)
            modalities[1] = 0

        if self.read_img and "img" in data and not third_party.isna(data.img) and data.img:
            img = third_party.imread(data.img)
            if self.read_img_text:
                img_text_string = third_party.image_to_string(data.img)
                try:
                    img_text = third_party.cat(
                        (
                            self.tokenizer.encode(
                                img_text_string,
                                max_length=self.max_sequence_length,
                                padding=False,
                                truncation=True,
                                return_tensors="pt",
                            ).flatten(),
                            third_party.tensor([self.tokenizer.pad_token_id]),
                        )
                    )
                    if img_text.size() == third_party.Size([1, 0]):
                        img_text = third_party.full((1,), self.pad_token_id, dtype=third_party.long)
                except third_party.third_party.TesseractError:
                    # if the image doesn't have dimensions in it's metadata, will throw
                    # an error, this catches
                    img_text = third_party.full((1,), self.pad_token_id, dtype=third_party.long)
                img_text_mask = third_party.zeros(img_text.size(), dtype=third_party.float)
                img_text_mask[img_text != self.pad_token_id] = 1
            else:
                img_text = third_party.full((1,), self.pad_token_id, dtype=third_party.long)
                img_text_mask = third_party.zeros(img_text.size(), dtype=third_party.float)
                img_text_mask[img_text != self.pad_token_id] = 1
            if self.transforms:
                img = self.transforms(img)
        else:
            img = third_party.zeros(3, 1, 1)
            img_text = third_party.full((1,), self.pad_token_id, dtype=third_party.long)
            img_text_mask = third_party.zeros(img_text.size(), dtype=third_party.float)
            img_text_mask[img_text != self.pad_token_id] = 1
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
            "img_text": img_text,
            "img_text_attn": img_text_mask,
            "embeddings": encoding,
            "attn": mask,
            "title": title,
            "title_attn": title_mask,
            "label": third_party.FloatTensor(label),
            "task_indices": third_party.LongTensor([task_indices]),
            "modalities": third_party.LongTensor(modalities),
        }
