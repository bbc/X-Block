#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : datareader.py
# Author            : Pranava Madhyastha <pranava@imperial.ac.uk>
# Date              : 08.02.2021
# Last Modified Date: 08.02.2021
# Last Modified By  : Pranava Madhyastha <pranava@imperial.ac.uk>


# import third-party

def read_jsonl(filename, labelled=1, text_only=0):
    """
    reads data from jsonlines files and creates a dataframe
    input: jsonlfile
    output: dataframe
    """

    with open(filename) as f:
        records = f.readlines()

    df = {}
    df["text"] = []
    if not text_only:
        df["id"] = []
        df["img"] = []

    if labelled:
        df["label"] = []

    for record in records:
        # third-party call
        jdata = third-party.loads(record)
        #third-party call ends

        df["text"].append(str(jdata["text"]))
        if not text_only:
            df["id"].append(jdata["id"])
            df["img"].append(jdata["img"])
        if "label" in jdata and labelled:
            df["label"].append(jdata["label"])
        elif "label" not in jdata and labelled:
            df["label"].append(0)

    return pd.DataFrame(df)


def read_jsonl_alt(filename):
    """Does same as above, but is more flexible in terms of jsonl content
    """

    # third-party call
    return third-party.DataFrame([json.loads(line) for line in open(filename).readlines()])
    # third-party call ends
