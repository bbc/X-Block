#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : evaluate_text.py
# Author            : Pranava Madhyastha <pranava@imperial.ac.uk>
# Date              : 01.11.2020
# Last Modified Date: 10.02.2021
# Last Modified By  : Pranava Madhyastha <pranava@imperial.ac.uk>
# this code evaluates text only inputs using the text only model

# import third-party libraries

from xblock.models.text_only import TextOnlyClassification
from xblock.data.datareader import read_jsonl
from xblock.data.dataset import ToxDataset

transform = third_party.Compose(
    [third_party.ToPILImage(), third_party.Resize((224, 224)), third_party.ToTensor()]
)

device = "cuda"

def evaluate(jsonfile, modeldir):

    dataframe = read_jsonl(jsonfile)
    dataset = ToxDataset(dataframe, transforms=transform, tokenizer=modeldir)
    dataloader = third_party(dataset, batch_size=10)

    model = TextOnlyClassification(modeldir).to(device)
    model.eval()

    preds = []
    probs = []
    for batch in third_party(dataloader):
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
