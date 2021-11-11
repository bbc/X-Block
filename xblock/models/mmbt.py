#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : mmbt.py
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
# base code for benchmarking

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from torchvision.models import resnet152

from .visualfeats import TransformerImageFeats

class MMBT(nn.Module):
    def __init__(
        self,
        model,
        output_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    ):
        """
        Args:
            model: the name of the model to use
            output_dropout_prob (optional): dropout when classifying the output of the transformer
            hidden_dropout_prob (optional): dropout on linear layers in transformer
            attention_probs_dropout_prob (optional): dropout on tokens in attention in transformer
        """
        super(MMBT, self).__init__()
        config = AutoConfig.from_pretrained(
            model,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
        )
        self.transformer = AutoModel.from_pretrained(model, config=config)
        self.head_mask = [None] * self.transformer.config.num_hidden_layers

        self.image_feats = TransformerImageFeats()
        self.img_emb_proj = nn.Linear(2048, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)

        self.dropout = nn.Dropout(output_dropout_prob)
        self.transformer2out = nn.Linear(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image, text, attn):
        # create input embeddings
        input_embeddings = torch.cat(
            (
                self.transformer.embeddings.word_embeddings(text).squeeze(1)[
                    :,
                    : min(
                        self.transformer.config.max_position_embeddings - 9,
                        text.size(1),
                    ),
                ],
                self.img_emb_proj(self.image_feats(image)),
            ),
            dim=1,
        )
        input_embeddings = (
            input_embeddings
            + self.transformer.embeddings.position_embeddings(
                torch.cat(
                    (
                        torch.arange(
                            input_embeddings.size(1) - 9, device=self.transformer.device
                        ),
                        torch.arange(9, device=self.transformer.device),
                    )
                )
            )
            + self.token_type_embeddings(
                torch.cat(
                    (
                        torch.zeros(
                            input_embeddings.size(1) - 9,
                            dtype=torch.long,
                            device=self.transformer.device,
                        ),
                        torch.ones(9, dtype=torch.long, device=self.transformer.device),
                    )
                )
            )
        )
        input_embeddings = self.transformer.embeddings.dropout(
            self.transformer.embeddings.LayerNorm(input_embeddings)
        )
        # create attention mask
        attn = torch.cat(
            (
                attn[
                    :,
                    : min(
                        self.transformer.config.max_position_embeddings - 9,
                        text.size(1),
                    ),
                ],
                torch.ones(len(attn), 9, device=self.transformer.device),
            ),
            dim=1,
        )
        if attn.dim() == 2:
            attn = attn[:, None, None, :]
        elif attn.dim() == 2:
            attn = attn[:, None, :, :]
        else:
            raise ValueError

        # pass through transformer
        out = self.transformer.pooler(
            self.transformer.encoder(
                input_embeddings, attention_mask=attn, head_mask=self.head_mask
            )[0]
        )

        # classify and return
        return self.sigmoid(self.transformer2out(self.dropout(out)))

    def save_model(self, directory):
        self.transformer.save_pretrained(directory)
        torch.save(self.img_emb_proj.state_dict(), f"{directory}/img_emb_proj.pth")
        torch.save(
            self.transformer2out.state_dict(), f"{directory}/transformer2out.pth"
        )
        torch.save(self.image_feats.state_dict(), f"{directory}/image_feats.pth")
        torch.save(self.token_type_embeddings.state_dict(), f"{directory}/token_type_embeddings.pth")

    def load_model(self, directory):
        self.transformer.from_pretrained(directory)
        self.img_emb_proj.load_state_dict(torch.load(f"{directory}/img_emb_proj.pth"))
        self.transformer2out.load_state_dict(
            torch.load(f"{directory}/transformer2out.pth")
        )
        self.image_feats.load_state_dict(torch.load(f"{directory}/image_feats.pth"))
        self.token_type_embeddings.load_state_dict(torch.load(f"{directory}/token_type_embeddings.pth"))
