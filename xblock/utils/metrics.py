#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : metrics.py
# Author            : Pranava Madhyastha <pranava@imperial.ac.uk>
# Date              : 01.11.2020
# Last Modified Date: 10.02.2021
# Last Modified By  : Pranava Madhyastha <pranava@imperial.ac.uk>
# metric code base


# import third-party libraries


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
        preds = third_party.empty(len(label), 1).to(out.device)
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
