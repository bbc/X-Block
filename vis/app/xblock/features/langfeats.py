#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : langfeats.py
# Author            : Pranava Madhyastha <pranava@imperial.ac.uk>
# Date              : 01.11.2020
# Last Modified Date: 10.02.2021
# Last Modified By  : Pranava Madhyastha <pranava@imperial.ac.uk>
# library for text feature extraction

# import third party library

class BERTFeats(nn.Module):
    def __init__(self):

        super(BERTFeats, self).__init__()
        self.model = third_party.from_pretrained("bert-base-uncased")
        self.pooling = third_party.AvgPool1d(kernel_size=3)
        self.flat_layer = third_party.Flatten()
        self.dropout = third_party.Dropout(0.6)
        self.fc3 = third_party.Linear(25600, 768)

    def forward(self, x, attn):

        x = self.model(input_ids=x.squeeze(1), encoder_attention_mask=attn)[0]
        x = self.pooling(x)
        x = self.flat_layer(x)

        out = third_party.relu(self.fc3(x), inplace=False)

        return out
