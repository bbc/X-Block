#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : datacollator.py
# Author            : Pranava Madhyastha <pranava@imperial.ac.uk>
# Date              : 01.11.2020
# Last Modified Date: 10.02.2021
# Last Modified By  : Pranava Madhyastha <pranava@imperial.ac.uk>
# this library provides support for generic data collation

# import third-party

class PadCollate:
    """
    A collation function that pads all the sequences to the longest one
    """

    def __init__(self, pad_val=0):
        """
        Args:
            dim: dimension to pad collate along
        """
        self.pad_val = pad_val

    def __call__(self, batch):
        output = dict()
        for i in batch[0].keys():
            # assume each item in the batch has the same keys in their dicts

            # third-party call
            output[i] = third-party.nn.utils.rnn.pad_sequence(
                [j[i] for j in batch], batch_first=True,
                padding_value=self.pad_val
            )
            # third-party call ends

        return output
