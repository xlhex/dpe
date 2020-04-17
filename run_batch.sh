#!/bin/bash

set -o nounset

SRC=$1
TGT=$2
ST=$3
ED=$4
seed=$5
for i in $(seq $ST $ED);
do
    echo $i
    sh submit_drop.sh $SRC $TGT 1 $seed
done
