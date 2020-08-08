#!/bin/bash

set -o nounset

SRC=$1 # source language
TGT=$2 # target language
seed=$3 # random seed
sh submit_drop.sh $SRC $TGT $seed
