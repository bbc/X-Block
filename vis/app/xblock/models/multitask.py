#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : multitask.py
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
# multitask, multistream codebase


import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from torchvision.models import resnet152, resnet18

class MultitaskModel(nn.Module):
    def __init__(
        self,
        transformer,
        feature_dim=200,
        combine_dim=800,
        output_dims=[1, 1, 1],
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        projection_dropout=0.1,
        large_cnn=False,
    ):
        super(MultitaskModel, self).__init__()

        config = AutoConfig.from_pretrained(
            transformer,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
        )
        self.transformer = AutoModel.from_pretrained(transformer)
        self.image_cnn = (
            nn.Sequential(*list(resnet152(pretrained=True).children())[:-1])
            if large_cnn
            else nn.Sequential(*list(resnet18(pretrained=True).children())[:-1])
        )

        self.transformer_hidden_size = self.transformer.config.hidden_size
        self.comment_proj = nn.Linear(self.transformer_hidden_size, 200)
        self.title_proj = nn.Linear(self.transformer_hidden_size, 200)
        self.image_text_proj = nn.Linear(self.transformer_hidden_size, 200)
        self.image_proj = nn.Linear(2048, 200) if large_cnn else nn.Linear(512, 200)
        self.feature_dropout = nn.Dropout(projection_dropout)

        self.comment_layernorm = nn.LayerNorm(feature_dim)
        self.title_layernorm = nn.LayerNorm(feature_dim)
        self.image_layernorm = nn.LayerNorm(feature_dim)
        self.image_text_layernorm = nn.LayerNorm(feature_dim)

        self.projection = nn.Sequential(
            nn.Dropout(projection_dropout),
            nn.ReLU(),
            nn.Linear(feature_dim * 4, combine_dim),
            nn.Dropout(projection_dropout),
            nn.ReLU(),
            nn.Linear(combine_dim, combine_dim),
            nn.Dropout(projection_dropout),
            nn.ReLU(),
        )
        self.output_dims = output_dims
        self.task_outputs = nn.ModuleList(
            [nn.Linear(combine_dim, i) for i in output_dims]
        )
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.feature_dim = feature_dim

    def forward(
        self,
        task_indices,
        comment=None,
        comment_attn=None,
        title=None,
        title_attn=None,
        image_text=None,
        image_text_attn=None,
        image=None,
        modalities=None,
        explainability=False,
    ):
        """        
        Args:
            task_indices: the task that each example corresponds to
            modalities: give the indices that are avaliable at each data point
        """

        comment_feats = torch.zeros(len(task_indices), self.feature_dim).to(
            self.transformer.device
        )
        if comment_feats is not None and (modalities[:, 0] == 1).sum().item():
            transformer_output = self.transformer(
                input_ids=comment[modalities[:, 0] == 1],
                attention_mask=comment_attn[modalities[:, 0] == 1],
            )
            hiddens = transformer_output[0]
            transformer_output = transformer_output[-1]
            
            if explainability:
                h_comment = hiddens.register_hook(self.assign_comment_grad)
                self.comment_activation = hiddens
            comment_feats[modalities[:, 0] == 1] = self.comment_layernorm(
                self.comment_proj(
                    self.feature_dropout(
                        transformer_output
                        if transformer_output.dim() == 2
                        else transformer_output[:, 0]
                    )
                )
            )
        title_feats = torch.zeros_like(comment_feats)
        if title is not None and (modalities[:, 1] == 1).sum().item():
            transformer_output = self.transformer(
                input_ids=title[modalities[:, 1] == 1],
                attention_mask=title_attn[modalities[:, 1] == 1],
            )
            hiddens = transformer_output[0]
            transformer_output = transformer_output[-1]
            
            if explainability:
                h_title = hiddens.register_hook(self.assign_title_grad)
                self.title_activation = hiddens
            title_feats[modalities[:, 1] == 1] = self.title_layernorm(
                self.title_proj(
                    self.feature_dropout(
                        transformer_output
                        if transformer_output.dim() == 2
                        else transformer_output[:, 0]
                    )
                )
            )

        image_text_feats = torch.zeros_like(comment_feats)
        if (
            image_text is not None
            and (modalities[:, 2] == 1).sum().item()
            and image_text_attn.sum() > 2
        ):
            transformer_output = self.transformer(
                input_ids=image_text[modalities[:, 2] == 1],
                attention_mask=image_text_attn[modalities[:, 2] == 1],
            )[-1]
            image_text_feats[modalities[:, 2] == 1] = self.image_text_layernorm(
                self.image_text_proj(
                    self.feature_dropout(
                        transformer_output
                        if transformer_output.dim() == 2
                        else transformer_output[:, 0]
                    )
                )
            )
        image_feats = torch.zeros_like(comment_feats)
        if image is not None and (modalities[:, 2] == 1).sum().item():
            if explainability:
                cnn1_output = self.image_cnn[0](image[modalities[:, 2] == 1])
                h_image = cnn1_output.register_hook(self.assign_image_grad)
                self.image_activation = cnn1_output
                image_feats[modalities[:, 2] == 1] = self.image_layernorm(
                    self.image_proj(
                        self.feature_dropout(
                            self.image_cnn[1:](cnn1_output).reshape(
                                (modalities[:, 2] == 1).sum(), -1
                            )
                        )
                    )
                )
                del cnn1_output
            else:
                image_feats[modalities[:, 2] == 1] = self.image_layernorm(
                    self.image_proj(
                        self.feature_dropout(
                            self.image_cnn(image[modalities[:, 2] == 1]).reshape(
                                (modalities[:, 2] == 1).sum(), -1
                            )
                        )
                    )
                )

        if "transformer_output" in locals():
            del transformer_output

        combine = torch.cat(
            (comment_feats, title_feats, image_text_feats, image_feats), dim=1
        )

        projection = self.projection(combine)
        output = torch.empty(
            len(projection), max(self.output_dims), device=projection.device
        )
        for task, dim in enumerate(self.output_dims):
            if dim == 1:
                output[task_indices[:, 0] == task, :1] = self.sigmoid(
                    self.task_outputs[task](projection[task_indices[:, 0] == task])
                )
            else:
                output[task_indices[:, 0] == task, :dim] = self.softmax(
                    self.task_outputs[task](projection[task_indices[:, 0] == task])
                )
        return output

    def save_model(self, dir):
        self.transformer.save_pretrained(dir)
        torch.save(self.image_cnn.state_dict(), f"{dir}/image_cnn.pth")
        torch.save(self.comment_proj.state_dict(), f"{dir}/comment_proj.pth")
        torch.save(self.title_proj.state_dict(), f"{dir}/title_proj.pth")
        torch.save(self.image_text_proj.state_dict(), f"{dir}/image_text_proj.pth")
        torch.save(self.image_proj.state_dict(), f"{dir}/image_proj.pth")
        torch.save(self.projection.state_dict(), f"{dir}/projection.pth")
        torch.save(self.task_outputs.state_dict(), f"{dir}/task_output.pth")
        torch.save(self.comment_layernorm.state_dict(), f"{dir}/comment_layernorm.pth")
        torch.save(self.title_layernorm.state_dict(), f"{dir}/title_layernorm.pth")
        torch.save(
            self.image_text_layernorm.state_dict(), f"{dir}/image_text_layernorm.pth"
        )
        torch.save(self.image_layernorm.state_dict(), f"{dir}/image_layernorm.pth")

    def load_model(self, dir):
        self.transformer.from_pretrained(dir)
        self.image_cnn.load_state_dict(torch.load(f"{dir}/image_cnn.pth"))
        self.comment_proj.load_state_dict(torch.load(f"{dir}/comment_proj.pth"))
        self.title_proj.load_state_dict(torch.load(f"{dir}/title_proj.pth"))
        self.image_text_proj.load_state_dict(torch.load(f"{dir}/image_text_proj.pth"))
        self.image_proj.load_state_dict(torch.load(f"{dir}/image_proj.pth"))
        self.projection.load_state_dict(torch.load(f"{dir}/projection.pth"))
        self.task_outputs.load_state_dict(torch.load(f"{dir}/task_output.pth"))
        import os

        # check to see if the model was previously saved with layernorm for compatability reasons
        if os.path.exists(f"{dir}/comment_layernorm.pth"):
            self.comment_layernorm.load_state_dict(
                torch.load(f"{dir}/comment_layernorm.pth")
            )
            self.title_layernorm.load_state_dict(
                torch.load(f"{dir}/title_layernorm.pth")
            )
            self.image_text_layernorm.load_state_dict(
                torch.load(f"{dir}/image_text_layernorm.pth")
            )
            self.image_layernorm.load_state_dict(
                torch.load(f"{dir}/image_layernorm.pth")
            )

    def assign_comment_grad(self, grad):
        self.comment_grad = grad

    def assign_title_grad(self, grad):
        self.title_grad = grad

    def assign_image_grad(self, grad):
        self.image_grad = grad