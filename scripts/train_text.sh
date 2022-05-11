#!/bin/bash

source ~/python-virtual-environments/x-block/bin/activate

export PYTHONPATH="${PYTHONPATH}:/Users/newell/workspace/starfruit-tagger/offline_training"

python /Users/newell/workspace/X-Block/xblock/train_text.py \
--huggingface_model facebook/bart-large \
--encrypted \
--train_data /Users/newell/datasets/moderation/20201218/abusive_and_offensive/train.jsonl.enc \
--val_data /Users/newell/datasets/moderation/20201218/abusive_and_offensive/test.jsonl.enc \
