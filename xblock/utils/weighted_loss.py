#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : weighted_loss.py
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
# utility class for weighted loss computation

import numpy as np
from sklearn.utils import class_weight
import torch

def classweight(labels, num_classes=None):
    """Compute the class weight of each item in the batch

    Args:
        labels: tensor of ground labels for each item in the batch
        num_claasses (optional): the number of classes in the classificaiton.
            Defaults to max in labels + 1
            
    Returns:
        weights for each class
    """
    if num_classes is None:
        num_classes = max(labels) + 1

    labels_device = labels.device
    labels = labels.cpu().type(torch.long).numpy()
    class_weights = class_weight.compute_class_weight(
        "balanced", classes=np.unique(labels), y=labels
    )

    output = np.ones(num_classes)
    for idx, label in enumerate(np.unique(labels)):
        output[label] = class_weights[idx]

    return torch.FloatTensor(output).to(labels_device)
