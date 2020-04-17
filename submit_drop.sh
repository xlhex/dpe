#!/bin/bash

set -o nounset

SRC=$1 # source language
TGT=$2 # target language
seed=$3 # seed
DATA=PATH/TO/DATA/bin_${4}

SAVE_DIR=checkpoints/$SRC-$TGT/seed$seed
TASK=translation
MODEL=transformer_wmt_en_de

if [ ! -d $SAVE_DIR ];then
    mkdir -p $SAVE_DIR
fi

TOKENS=8000
FREQ=2 # this is for 8 GPUs, if you use one GPU, change to 16

log=$SAVE_DIR/log.txt
subset="train_$4"
echo $subset
python3 ./train.py $DATA --task $TASK -s $SRC -t $TGT --segment --raw-text \
     -a $MODEL --optimizer adam --lr 0.0005 \
     --dropout 0.3 --max-tokens $TOKENS --clip-norm 0.0 \
     --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0 \
     --criterion dynamic_programming_cross_entropy --max-epoch $4 --update-freq $FREQ \
     --warmup-updates 4000 --warmup-init-lr '1e-07' \
     --adam-betas '(0.9, 0.98)' --save-dir $SAVE_DIR \
     --encoder-layers 4 --decoder-layers 4 --ddp-backend=no_c10d \
     --share-all-embeddings --no-epoch-checkpoints --seed $seed >> $log
