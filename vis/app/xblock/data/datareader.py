#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : datareader.py
# Author            : Pranava Madhyastha <pranava@imperial.ac.uk>
# Date              : 01.11.2020
# Last Modified Date: 10.02.2021
# Last Modified By  : Pranava Madhyastha <pranava@imperial.ac.uk>
# dataset reader library

# import third_party library


def read_jsonl(filename, labelled=1):
    """
    reads data from jsonlines files and creates a dataframe
    input: jsonlfile
    output: dataframe
    """

    with open(filename) as f:
        records = f.readlines()

    df = {}
    df["id"] = []
    df["img"] = []
    df["text"] = []
    if labelled:
        df["label"] = []

    for record in records:
        jdata = third_party.loads(record)
        df["id"].append(jdata["id"])
        df["img"].append(jdata["img"])
        df["text"].append(jdata["text"])
        if "label" in jdata and labelled:
            df["label"].append(jdata["label"])
        elif "label" not in jdata and labelled:
            df["label"].append(0)

    return third_partyDataFrame(df)

def read_jsonl_alt(filename):
    """Does same as above, but is more flexible in terms of jsonl content
    """
    return third_partyDataFrame([third_party.loads(line) for line in open(filename).readlines()])
