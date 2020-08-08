#!/usr/bin/python
#-*-coding:utf-8 -*-
#Author   : Xuanli He
#Version  : 1.0
#Filename : seg_data.py
from __future__ import print_function

import os
import sys

from collections import Counter

def main(base_dir, lang_pair):
    src, tgt = lang_pair.split("-")
    src_chars = Counter()
    with open(os.path.join(base_dir, "train.bpe.{}".format(src))) as f:
        for i, line in enumerate(f):
            chars = "".join(line.split())
            src_chars.update(chars)
    with open(os.path.join(base_dir, "train.bpe.{}".format(tgt))) as f:
        for i, line in enumerate(f):
            chars = "".join(line.split())
            src_chars.update(chars)
    for k, v in src_chars.items():
        print(k,v)

if __name__ == "__main__":
    main(*sys.argv[1:])
