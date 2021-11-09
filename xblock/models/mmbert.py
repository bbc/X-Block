#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : mmbert.py
# Author            : Pranava Madhyastha <pranava@imperial.ac.uk>
# Date              : 01.11.2020
# Last Modified Date: 10.02.2021
# Last Modified By  : Pranava Madhyastha <pranava@imperial.ac.uk>
# code for benchamarking multimodal classification using multimodal bert

# import third_party

from .visualfeats import EnsembleFeats, TransformerImageFeats
from .langfeats import BERTFeats



class MMBERT(nn.Module):
    def __init__(self, transformer="bert-base-uncased"):
        super(MMBERT, self).__init__()

        self.langfeats = BERTFeats(model=transformer)

        # third_party calls
        self.batch_norm = third_party.BatchNorm1d(6000)
        self.visfeats = EnsembleFeats()
        self.fc1 = third_party.Linear(1536, 6000)
        self.fc2 = third_party.Linear(6000, 3000)
        self.fc3 = third_party.Linear(3000, 1)
        # third_party calls end

    def forward(self, image, text, attn):
        x1 = self.visfeats(image)
        x2 = self.langfeats(text, attn)

        x3 = self.fc1(torch.cat((x1, x2), dim=1))
        x3 = self.batch_norm(x3)
        x4 = F.relu(self.fc2(x3), inplace=False)
        out_1 = self.fc3(x4)
        out = torch.sigmoid(out_1)
        return out
