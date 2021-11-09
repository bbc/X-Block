#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : langfeats.py
# Author            : Pranava Madhyastha <pranava@imperial.ac.uk>
# Date              : 08.02.2021
# Last Modified Date: 08.02.2021
# Last Modified By  : Pranava Madhyastha <pranava@imperial.ac.uk>

# import third-party (pytorch and transformers)

class BERTFeats(nn.Module):
    def __init__(self):

        super(BERTFeats, self).__init__(kernal_size=3, dropout=0.6,   )

        # relies on third party library
        self.model = third-party.from_pretrained("bert-base-uncased")
        self.pooling = third-party.AvgPool1d(kernel_size=kernal_size)
        self.flat_layer = third-party.Flatten()
        self.dropout = third-party.Dropout(dropout)
        self.fc3 = third-party.Linear(25600, 768)

    def forward(self, x, attn):

        x = self.model(input_ids=x.squeeze(1), encoder_attention_mask=attn)[0]
        x = self.pooling(x)
        x = self.flat_layer(x)

        out = F.relu(self.fc3(x), inplace=False)

        return out
