#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : datareader.py
# Author            : Pranava Madhyastha <pranava@imperial.ac.uk>
# Date              : 08.02.2021
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

import json
import pandas as pd
from .aes import open_file

def read_jsonl(filename, labelled=1, text_only=0, password=None):
    """
    
    reads data from jsonlines files and creates a dataframe
    input: jsonlfile
    output: dataframe
    """
    if password is not None:
        f = open_file(filename, password)
        records = []
                
        for line in f:
            records.append(line)
    
    else:
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
        jdata = json.loads(record)

        df["text"].append(str(jdata["text"]))
        
        if not text_only:
            df["id"].append(jdata["id"])
            df["img"].append(jdata["img"])
        
        if "label" in jdata and labelled:
            df["label"].append(jdata["label"])
        
        elif "failed" in jdata and labelled:
            df["label"].append(int(jdata["failed"]))
        
        elif "label" not in jdata and labelled:
            df["label"].append(0)

    return pd.DataFrame(df)


def read_jsonl_alt(filename, password=None):
    """Does same as above, but is more flexible in terms of jsonl content
    """
    if password is not None:
        reader = open_file(filename, password)
    else:
        reader = open(filename).readlines()
    
    return pd.DataFrame([json.loads(line) for line in reader])
