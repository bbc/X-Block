#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : evaluate_text.py
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
# this code evaluates text only inputs using the text only model

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

from xblock.models.text_only import TextOnlyClassification
from xblock.data.datareader import read_jsonl
from xblock.data.dataset import ToxDataset

transform = transforms.Compose(
    [transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()]
)

device = "cuda"

def evaluate(jsonfile, modeldir):

    dataframe = read_jsonl(jsonfile)
    dataset = ToxDataset(dataframe, transforms=transform, tokenizer=modeldir)
    dataloader = DataLoader(dataset, batch_size=10)

    model = TextOnlyClassification(modeldir).to(device)
    model.eval()

    preds = []
    probs = []
    for batch in tqdm(dataloader):
        out = model(batch["embeddings"].to(device), batch["attn"].squeeze(1).to(device))
        arr = out.detach().cpu().numpy().copy()
        arr[arr < 0.5] = 0
        arr[arr > 0.5] = 1
        probs_temp = list(out.detach().cpu().numpy().reshape(1, -1)[0])
        preds_temp = list(arr.reshape(1, -1)[0].astype(int))
        preds.extend(preds_temp)
        probs.extend(probs_temp)

    print(preds)

    return preds


if __name__ == "__main__":
    import plac

    plac.call(evaluate)
