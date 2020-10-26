#!/bin/bash

SRC=$1
TGT=$2
seed=$3
subset=$4

DATA=PATH/TO/DATA/$SRC-$TGT/
MPATH=checkpoints/bpe_dropout_post/$SRC-$TGT/seed$seed
CKPT=$MPATH/checkpoint_best.pt
TASK=translation
BATCH=35
OUTPUT=$MPATH/$subset.$SRC-$TGT.$TGT
python3 ./find_seg.py $DATA --task $TASK --segment --raw-text -s $SRC -t $TGT --path $CKPT --batch-size $BATCH --max-len-b 100 --gen-subset $subset > $OUTPUT
