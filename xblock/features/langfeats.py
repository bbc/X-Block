#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : langfeats.py
# Author            : Pranava Madhyastha <pranava@imperial.ac.uk>
# Date              : 08.02.2021
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

from transformers import BertModel
import torch.nn as nn

class BERTFeats(nn.Module):
    def __init__(self):

        super(BERTFeats, self).__init__(kernal_size=3, dropout=0.6,)
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.pooling = nn.AvgPool1d(kernel_size=3)
        self.flat_layer = nn.Flatten()
        self.dropout = nn.Dropout(0.6)
        self.fc3 = nn.Linear(25600, 768)

    def forward(self, x, attn):

        x = self.model(input_ids=x.squeeze(1), encoder_attention_mask=attn)[0]
        x = self.pooling(x)
        x = self.flat_layer(x)

        out = F.relu(self.fc3(x), inplace=False)

        return out
