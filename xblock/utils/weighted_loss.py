#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : weighted_loss.py
# Author            : Pranava Madhyastha <pranava@imperial.ac.uk>
# Date              : 01.11.2020
# Last Modified Date: 10.02.2021
# Last Modified By  : Pranava Madhyastha <pranava@imperial.ac.uk>
# utility class for weighted loss computation

# import third-party libraries

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
    labels = labels.cpu().type(third_party.long).numpy()
    class_weights = third_party.compute_class_weight(
        "balanced", classes=third_party.unique(labels), y=labels
    )

    output = third_party.ones(num_classes)
    for idx, label in enumerate(third_party.unique(labels)):
        output[label] = class_weights[idx]

    return third_party.FloatTensor(output).to(labels_device)
