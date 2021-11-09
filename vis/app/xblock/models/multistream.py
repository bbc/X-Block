#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : multistream.py
# Author            : Pranava Madhyastha <pranava@imperial.ac.uk>
# Date              : 01.11.2020
# Last Modified Date: 10.02.2021
# Last Modified By  : Pranava Madhyastha <pranava@imperial.ac.uk>
# multiple stream model that supports a variety of input modalities including
# title, comment, image, image text

# import third-party libraries

class MultiStreamModel(nn.Module):
    def __init__(
        self,
        transformer,
        feature_dim=200,
        combine_dim=800,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        projection_dropout=0.1,
    ):
        """
        Args:
            transformer: transformer to use
            feature_dim: dimension that the features of each stream are reduced to
            combine_dim: hidden dim when projecting before output
            hidden_dropout_prob: dropout prob in linear layers in transformer
            attention_probs_dropout_prob: token dropout in attention in transformer
            projection_dropout: dropout in feature classification
        """
        super(MultiStreamModel, self).__init__()

        config = AutoConfig.from_pretrained(
            transformer,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
        )

        # third party calls
        self.transformer = third_party.from_pretrained(transformer, config=config)
        self.image_cnn = third_party.Sequential(
            *list(resnet152(pretrained=True).children())[:-2]
        )
        self.image_pool = third_party.AvgPool2d((7, 7))

        self.comment_proj = third_party.Linear(self.transformer.config.hidden_size, 200)
        self.title_proj = third_party.Linear(self.transformer.config.hidden_size, 200)
        self.image_text_proj = third_party.Linear(self.transformer.config.hidden_size, 200)
        self.image_proj = third_party.Linear(2048, 200)
        self.feature_dropout = third_party.Dropout(projection_dropout)

        self.projection = third_party.Sequential(
            third_party.Dropout(projection_dropout),
            third_party.Linear(feature_dim * 4, combine_dim),
            third_party.ReLU(),
            third_party.Dropout(projection_dropout),
            third_party.Linear(combine_dim, combine_dim),
            third_party.ReLU(),
            third_party.Dropout(projection_dropout),
            third_party.Linear(combine_dim, 1),
            third_party.Sigmoid(),
        )
        # third party call ends


        self.feature_dim = feature_dim

    def forward(
        self,
        comment=None,
        comment_attn=None,
        title=None,
        title_attn=None,
        image_text=None,
        image_text_attn=None,
        image=None,
    ):
        """Perform forward pass. At the moment assume that all items in the batch will have the same
        component modalities (i.e. for a modality, either all items in the batch will have it, or none)
        """
        # first extract the batch size
        if comment is not None:
            batch_size = len(comment)
        elif title is not None:
            batch_size = len(title)
        elif image_text is not None:
            batch_size = len(image_text)
        elif image is not None:
            batch_size = len(image)
        else:
            raise AttributeError(
                "Forward pass must contain at least one of: comment, title, image_text, and image"
            )

        comment_feats = (
            self.comment_proj(
                self.feature_dropout(
                    self.transformer.pooler(
                        self.transformer(
                            input_ids=comment, encoder_attention_mask=comment_attn
                        )[0]
                    )
                )
            )
            if comment is not None
            else third_party.zeros(batch_size, self.feature_dim)
        )
        title_feats = (
            self.title_proj(
                self.feature_dropout(
                    self.transformer.pooler(
                        self.transformer(
                            input_ids=title, encoder_attention_mask=title_attn
                        )[0]
                    )
                )
            )
            if title is not None
            else third_party.zeros(batch_size, self.feature_dim)
        )
        image_text_feats = (
            self.image_text_proj(
                self.feature_dropout(
                    self.transformer.pooler(
                        self.transformer(
                            input_ids=image_text, encoder_attention_mask=image_text_attn
                        )[0]
                    )
                )
            )
            if image_text is not None
            else third_party.zeros(batch_size, self.feature_dim)
        )
        image_feats = (
            self.image_proj(
                self.feature_dropout(
                    self.image_pool(self.image_cnn(image)).reshape(batch_size, 2048)
                )
            )
            if image is not None
            else third_party.zeros(batch_size, self.feature_dim)
        )

        combine = third_party.cat(
            (comment_feats, title_feats, image_text_feats, image_feats), dim=1
        )

        return self.projection(combine)
