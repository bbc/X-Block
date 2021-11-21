# X-Block

This repository contains code for multimodal toxicity detection.

## Installation
Requirements:
- python=3.9
- torch=1.8
```
pip install -r requirements.txt
```

## Data
The input is in format of jsonl file. The `text` and `label` fields are required for all models. The `img` field is used when training multimodal (including multitask) models, and the `title` field can be used to train multitask models.
```
{"text": XXX, "label": 0, "img": XXX.jpg, "title": XXX, ...}
{"text": XXX, "label": 0, "img": XXX.jpg, "title": XXX, ...}
...
```

## Training and Inference
The codebase consists of two parts, training and inference. The codes for training are under the `xblock` directory. There are three training scripts:
* text-only training: `x-block/train_text.py`
* multimodal training: `x-block/train_multimodal.py`
* multimodal and mult-istream training: `x-block/train_multitask.py`

The `vis` directory is the primary directory for inference which will also start a web application with `vis/main.py`. The trained model weights needs to be copied to `vis/app/weights` so that the web application can load the trained model.


## LICENSE
Copyright (c) 2020, Imperial College, London
All rights reserved.
Contributors: Pranava Madhyastha and Lucia Specia

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of Imperial College nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

