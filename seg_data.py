#!/usr/bin/python
#-*-coding:utf-8 -*-
#Author   : Xuanli He
#Version  : 1.0
#Filename : seg_data.py
from __future__ import print_function

import os
import sys

import sentencepiece as spm

def main(base_dir, lang_pair):
    src, tgt = lang_pair.split("-")

    files = list(map(lambda x: os.path.join(base_dir, x), os.listdir(base_dir)))
    files = list(filter(lambda x: "tok" in x, files))

    src_vocab_size = 32003
    src_model_prefix = os.path.join(base_dir, "m_bpe")

    src_model_path = src_model_prefix + ".model"

    src_data_path = os.path.join(base_dir, "train.tok.{}".format(src))
    tgt_data_path = os.path.join(base_dir, "train.tok.{}".format(tgt))

    if not os.path.exists(src_model_prefix+".model"):
        input_files = "{},{}".format(src_data_path, tgt_data_path)
        spm.SentencePieceTrainer.train('--input={} --model_prefix={} --vocab_size={} --model_type=bpe'.format(input_files, src_model_prefix, src_vocab_size))

    sp_bpe = spm.SentencePieceProcessor()

    src_files = filter(lambda x: x.endswith(src), files)
    sp_bpe.load(src_model_path)
    for org_f in src_files:
        pro_f = org_f.replace("tok", "bpe") 
        print("Processing {} -> {}".format(org_f, pro_f))
        with open(org_f) as f1, open(pro_f, "w") as f2:
            for i, line in enumerate(f1):
                line = sp_bpe.encode_as_pieces(line.strip())
                f2.write(" ".join(line))
                f2.write("\n")

    tgt_files = filter(lambda x: x.endswith(tgt), files)
    for org_f in tgt_files:
        pro_f = org_f.replace("tok", "bpe") 
        print("Processing {} -> {}".format(org_f, pro_f))
        with open(org_f) as f1, open(pro_f, "w") as f2:
            for i, line in enumerate(f1):
                line = sp_bpe.encode_as_pieces(line.strip())
                f2.write(" ".join(line))
                f2.write("\n")

if __name__ == "__main__":
    main(*sys.argv[1:])
