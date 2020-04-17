# Dynamic Programming Encoding for Subword Segmentation in Neural Machine Translation

## Descriptions
This repo contains source code and pre-processed corpora for Dynamic Programming Encoding (DPE) for Subword Segmentation in Neural Machine Translation (accepted to ACL2020)


## Dependencies
* python3
* [fairseq](https://github.com/pytorch/fairseq) (commit: 58b912f)
* pytorch1.1
* cuda 10.0

## Training
To train a DPE segmenter
```shell
# SRC: source language
# TGT: target language
# ST: index of the start file
# ED: the total number of files
# SEED: a seed for reproducibility
sh run_batch SRC TGT ST ED SEED
```

## MAP Inference
To segment a corpus
```shell
# SRC: source language
# TGT: target language
# ST: index of the start file
# ED: the total number of files
# SEED: a seed for reproducibility
sh seg_batch.sh SRC TGT ST ED SEED
```

## Machine Translation
Once your corpus is segmented, you can use your favourite MT toolkit to train a MT system. We use fairseq for our experiments.
