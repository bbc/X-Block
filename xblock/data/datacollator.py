#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : datacollator.py
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
# this library provides support for generic data collation

import torch

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
            output[i] = torch.nn.utils.rnn.pad_sequence(
                [j[i] for j in batch], batch_first=True,
                padding_value=self.pad_val
            )
        return output
