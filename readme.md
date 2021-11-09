```
 File              : readme.md
 Author            : Pranava Madhyastha <pranava@imperial.ac.uk>
 Date              : 09.02.2021
 Last Modified Date: 10.02.2021
 Last Modified By  : Pranava Madhyastha <pranava@imperial.ac.uk>
```
# X-Block

This repository includes a snapshot of the codebase for the X-Block: it
contains a stripped down version (third-party libraries are stripped) along for
training, evaluation and visualisation of X-Block.


## Brief introduction


X-Block is written in python3.9 with pytorch 1.7.1 as the backend. The codebase
consists of two parts, inference and training. We will first focus
on the inference. The starting point is the ‘vis/’ directory in the distributed
code. The `vis` directory is the primary directory for inference which will
also start a web application.

There are three training scripts:
* text-only training: `x-block/train_text.py`
* multimodal training: `x-block/train_multimodal.py`
* multimodal and mult-istream training: `x-block/train_multitask.py`

that borrow from `xblock/models`
