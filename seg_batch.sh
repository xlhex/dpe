#!/bin/bash
set -o nounset

SRC=$1
TGT=$2
ST=$3
ED=$4
seed=$5
for i in $(seq $ST $ED);
do
    echo train_$i
    if [ $i -eq 1 ];
    then
        echo "valid"
        sh find_seg.sh $SRC $TGT $seed valid
        echo "test"
        sh find_seg.sh $SRC $TGT $seed test
    fi
    sh find_seg.sh $SRC $TGT $seed train_$i
done
