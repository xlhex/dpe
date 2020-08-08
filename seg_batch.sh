#!/bin/bash
set -o nounset

SRC=$1
TGT=$2
seed=$3
echo "valid"
sh find_seg.sh $SRC $TGT $seed valid
echo "test"
sh find_seg.sh $SRC $TGT $seed test
echo "train"
sh find_seg.sh $SRC $TGT $seed train
