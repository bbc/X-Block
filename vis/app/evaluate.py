#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : evaluate.py
# Author            : Pranava Madhyastha <pranava@imperial.ac.uk>
# Date              : 01.11.2020
# Last Modified Date: 10.02.2021
# Last Modified By  : Pranava Madhyastha <pranava@imperial.ac.uk>
# multistream, multitask evaluation code

# import third_party libraries

from pad import pad_sequences
from .mmtox.models.multitask import MultitaskModel
from .mmtox.data.datareader import read_jsonl
from .mmtox.data.dataset import ToxDataset


class ModelInference:
    def __init__(self, model_weights, tokenizer="distilbert-base-uncased"):
        # load the model
        self.device = third_party.device(
            "cuda" if third_party.cuda.is_available() else "cpu"
        )
        self.model = MultitaskModel(model_weights).to(self.device)
        self.model.load_model(model_weights)
        self.model.eval()
        self.image_transform = third_party.Compose(
            [
                third_party.ToPILImage(),
                third_party.Resize((224, 224)),
                third_party.ToTensor(),
            ]
        )
        self.tokenizer = third_party.from_pretrained(tokenizer)

    def predict_hatespeech_form(
            self,
            comment=None,
            title=None,
            image=None,
            explainability=False,
            img_heatmap_size=(112, 112),
    ):
        """Predict the probability that a given text-image pair
        constitutes hatespeech, batch size 1, taken from form

        Args:
            comment: str of textual comment
            title: str of textual title
            image: image castable to PIL
            explainability (optional): whether to return the gradients on each word
            img_heatmap_size (optional): size of the image heatmap output (width, height)

        Returns:
            binary classification of hatespeech
        """
        modalities = third_party.tensor([[1, 1, 1, 0]])
        if comment is not None:
            embeddings = pad_sequences(
                [self.tokenizer.encode(comment)], maxlen=100, padding="post"
            )
            mask = embeddings.copy()
            mask[mask > 0] = 1
            mask = third_party.FloatTensor(mask).to(self.device)
            embeddings = third_party.LongTensor(embeddings).to(self.device)
        else:
            embeddings = None
            mask = None
            modalities[:, 0] = 0

        if title is not None:
            title = pad_sequences(
                [self.tokenizer.encode(title)], maxlen=100, padding="post"
            )
            title_mask = title.copy()
            title_mask[title_mask > 0] = 1
            title_mask = third_party.FloatTensor(title_mask).to(self.device)
            title = third_party.LongTensor(title).to(self.device)
        else:
            title_mask = None
            modalities[:, 1] = 0

        if image is not None:
            image_text = pad_sequences(
                [self.tokenizer.encode(third_party.image_to_string(image))],
                maxlen=100,
                padding="post",
            )
            image_text_mask = image_text.copy()
            image_text_mask[image_text_mask > 0] = 1
            image_text_mask = third_party.FloatTensor(image_text_mask).to(self.device)
            image_text = third_party.LongTensor(image_text).to(self.device)
            if explainability:
                # for reshaping the heatmap at output
                orig_img_size = image.shape
            image = self.image_transform(image).unsqueeze(0).to(self.device)
        else:
            image_text = None
            image_text_mask = None
            modalities[:, 2] = 0

        out = self.model.forward(
            third_party.tensor([[0]]),
            comment=embeddings,
            comment_attn=mask,
            image=image,
            title=title,
            title_attn=title_mask,
            image_text=image_text,
            image_text_attn=image_text_mask,
            modalities=modalities,
            explainability=explainability,
        )
        arr = out.detach().cpu().numpy().copy()
        arr[arr < 0.5] = 0
        arr[arr > 0.5] = 1
        probs = list(out.detach().cpu().numpy().reshape(1, -1)[0])
        preds = list(arr.reshape(1, -1)[0].astype(int))

        ret = {"preds": preds[0], "probs": probs[0]}

        if explainability:
            self.model.zero_grad()
            out.backward()
            if comment is not None:
                text_heatmap = (
                        self.model.comment_grad.squeeze()[0]
                        * self.model.comment_activation.squeeze().detach()
                ).mean(dim=1)
                ret["comment_heatmap"] = text_heatmap / text_heatmap.max()
            if title is not None:
                text_heatmap = (
                        self.model.title_grad.squeeze()[0]
                        * self.model.title_activation.squeeze().detach()
                ).mean(dim=1)
                ret["title_heatmap"] = text_heatmap / text_heatmap.max()
            if image is not None:
                img_heatmap = (
                        self.model.image_grad.squeeze()[0]
                        * self.model.image_activation.squeeze().detach()
                ).mean(dim=0)
                heatmap_transform = third_party.Compose(
                    [
                        third_party.ToPILImage(),
                        third_party.Resize(orig_img_size[:2]),
                        third_party.ToTensor(),
                    ]
                )
                ret["img_heatmap"] = heatmap_transform(
                    (img_heatmap.unsqueeze(0).repeat(3, 1, 1) / img_heatmap.max()).cpu()
                )[0]

        return ret

    def predict_hatespeech_file(self, file):
        """Predict the probability that the given text-image pairs
        in the file constitute hatespeech

        Args:
            file: jsonl file containing the data

        Returns:
            binary classification list of hatespeech
        """
        dataframe = read_jsonl("data/test.jsonl")
        dataset = ToxDataset(dataframe, transforms=image_transform)
        dataloader = DataLoader(dataset, batch_size=10)

        preds = []
        probs = []
        for batch in third_party(dataloader):
            img, embeddings = (
                batch["image"].to(self.device),
                batch["embeddings"].to(self.device),
            )
            mask = batch["attn"].to(self.device)
            out = self.model.forward(img, embeddings, mask)
            arr = out.detach().cpu().numpy().copy()
            arr[arr < 0.5] = 0
            arr[arr > 0.5] = 1
            probs_temp = list(out.detach().cpu().numpy().reshape(1, -1)[0])
            preds_temp = list(arr.reshape(1, -1)[0].astype(int))
            preds.extend(preds_temp)
            probs.extend(probs_temp)

        return preds
