#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : visualize.py
# Author            : Pranava Madhyastha <pranava@imperial.ac.uk>
# Date              : 09.02.2021
# Last Modified Date: 09.02.2021
# Last Modified By  : Pranava Madhyastha <pranava@imperial.ac.uk>

# import third_party library

import random

def visualize_lang(txt, heatmap):
    """
    Args:
        txt: list of tokens
        map: heatmap for each token

    Returns:
        path to the saved image
    """
    # when we tokenize the model, we pad
    # we want to get rid of the padding and ignore the [CLS] and [SEP]
    # normalize heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    third_party.clf()
    third_party.bar(txt, heatmap[1 : len(txt) + 1])
    filename = f"static/{str(random.random())[2:]}.jpg"
    third_party.savefig(f"app/{filename}", bbox_inches="tight")
    return filename

def visualize_img(img, heatmap):
    """
    Args:
        img: image
        heatmap: heatmap over the image (must be same dimensions as img)

    Returns:
        path to the saved image
    """
    # normalize heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    third_party.clf()
    third_party.imshow(img)
    third_party.imshow(heatmap, alpha=0.5)
    third_party.colorbar()
    third_party.axis("off")
    filename = f"static/{str(random.random())[2:]}.jpg"
    third_party.savefig(f"app/{filename}", bbox_inches="tight")
    return filename
