#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : multistream.py
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
# primary codebase for multistream model - takes in input from a variety of streams

import torch
import torch.nn as nn
from torchvision.models import resnet152
from transformers import AutoModel, AutoConfig

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
        self.transformer = AutoModel.from_pretrained(transformer, config=config)
        self.image_cnn = nn.Sequential(
            *list(resnet152(pretrained=True).children())[:-2]
        )
        self.image_pool = nn.AvgPool2d((7, 7))

        self.comment_proj = nn.Linear(self.transformer.config.hidden_size, 200)
        self.title_proj = nn.Linear(self.transformer.config.hidden_size, 200)
        self.image_text_proj = nn.Linear(self.transformer.config.hidden_size, 200)
        self.image_proj = nn.Linear(2048, 200)
        self.feature_dropout = nn.Dropout(projection_dropout)

        self.projection = nn.Sequential(
            nn.Dropout(projection_dropout),
            nn.Linear(feature_dim * 4, combine_dim),
            nn.ReLU(),
            nn.Dropout(projection_dropout),
            nn.Linear(combine_dim, combine_dim),
            nn.ReLU(),
            nn.Dropout(projection_dropout),
            nn.Linear(combine_dim, 1),
            nn.Sigmoid(),
        )
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
            else torch.zeros(batch_size, self.feature_dim)
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
            else torch.zeros(batch_size, self.feature_dim)
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
            else torch.zeros(batch_size, self.feature_dim)
        )
        image_feats = (
            self.image_proj(
                self.feature_dropout(
                    self.image_pool(self.image_cnn(image)).reshape(batch_size, 2048)
                )
            )
            if image is not None
            else torch.zeros(batch_size, self.feature_dim)
        )

        combine = torch.cat(
            (comment_feats, title_feats, image_text_feats, image_feats), dim=1
        )

        return self.projection(combine)