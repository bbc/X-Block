#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : textonly.py
# Author            : Pranava Madhyastha <pranava@imperial.ac.uk>
# Date              : 01.11.2020
# Last Modified Date: 10.02.2021
# Last Modified By  : Pranava Madhyastha <pranava@imperial.ac.uk>

# this code supports textonly classification modules for benchmarking

class TextOnlyClassification(nn.Module):
    def __init__(
        self, model, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1
    ):
        super(TextOnlyClassification, self).__init__()

        # third_party calls
        config = third_party.from_pretrained(
            model,
            num_labels=1,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
        )
        self.model = third_party.from_pretrained(
            model, config=config
        )
        self.sigmoid = third_party.Sigmoid()
        # third_party calls

    def forward(self, text, attn):
        return self.sigmoid(
            self.model(input_ids=text.squeeze(1), attention_mask=attn)[0]
        )
