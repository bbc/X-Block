#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : metrics.py
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
# metric code base


import torch


def acc_pre_rec(out, label):
    """Compute accuracy, precision, and recall for some given input.
    In a binary classification setup, expects a single prediction, in multiclass expects class probabilities.
    Inputs should be tensors
    """
    aggregate = dict()
    out = out.reshape(len(out), -1)
    label = label.reshape(len(label), -1)
    if out.size(1) == 1:
        # then binary classification
        preds = torch.empty(len(label), 1).to(out.device)
        preds[out > 0.5] = 1
        preds[out <= 0.5] = 0
    else:
        preds = out.argmax(dim=1)
        preds = preds.reshape(-1, 1)

    aggregate["accuracy"] = (preds == label).sum().item() / len(out)
    aggregate["precision"] = sum(
        [
            0
            if (preds[label == i] == i).sum().item() == 0  # avoid 0 division errors
            else (preds[label == i] == i).sum().item()
            / (
                (preds[label == i] == i).sum().item()
                + (preds[label == i] != i).sum().item()
            )
            for i in range(out.size(1))
        ]
    ) / out.size(1)
    aggregate["recall"] = sum(
        [
            0
            if (preds[label == i] == i).sum().item() == 0  # avoid 0 division errors
            else (preds[label == i] == i).sum().item()
            / (
                (preds[label == i] == i).sum().item()
                + (preds[label != i] == i).sum().item()
            )
            for i in range(out.size(1))
        ]
    ) / out.size(1)

    return aggregate
