# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import Counter
from functools import reduce
import os, re

import torch
from multiprocessing import Pool

SPACE_NORMALIZER = re.compile(r"\s+")


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos) # search where this character begins

class Tokenizer:

    @staticmethod
    def add_file_to_dictionary_single_worker(filename, tokenize, eos_word, worker_id=0, num_workers=1):
        counter = Counter()
        with open(filename, 'r') as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_workers
            offset = worker_id * chunk_size
            end = offset + chunk_size
            f.seek(offset)
            if offset > 0:
                safe_readline(f) # drop first incomplete line
            line = f.readline()
            while line:
                for word in tokenize(line):
                    counter.update([word])
                counter.update([eos_word])
                if f.tell() > end:
                    break
                line = f.readline()
        return counter

    @staticmethod
    def add_file_to_dictionary(filename, dict, tokenize, num_workers):
        def merge_result(counter):
            for w, c in counter.items():
                dict.add_symbol(w, c)
        if num_workers > 1:
            pool = Pool(processes=num_workers)
            results = []
            for worker_id in range(num_workers):
                results.append(pool.apply_async(
                    Tokenizer.add_file_to_dictionary_single_worker,
                    (filename, tokenize, dict.eos_word, worker_id, num_workers)
                ))
            pool.close()
            pool.join()
            for r in results:
                merge_result(r.get())
        else:
            merge_result(Tokenizer.add_file_to_dictionary_single_worker(filename, tokenize, dict.eos_word))

    @staticmethod
    def binarize(filename, dict, consumer, tokenize=tokenize_line,
                            append_eos=True, reverse_order=False,
                            offset=0, end=-1):
        nseq, ntok = 0, 0
        replaced = Counter()
        def replaced_consumer(word, idx):
            if idx == dict.unk_index and word != dict.unk_word:
                replaced.update([word])
        with open(filename, 'r') as f:
            f.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)
            while line:
                if end > 0 and f.tell() > end:
                    break
                ids = Tokenizer.tokenize(
                    line=line,
                    dict=dict,
                    tokenize=tokenize,
                    add_if_not_exist=False,
                    consumer=replaced_consumer,
                    append_eos=append_eos,
                    reverse_order=reverse_order,
                )
                nseq += 1
                ntok += len(ids)
                consumer(ids)
                line = f.readline()
        return {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': replaced}

    @staticmethod
    def find_offsets(filename, num_chunks):
        with open(filename, 'r') as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_chunks
            offsets = [0 for _ in range(num_chunks + 1)]
            for i in range(1, num_chunks):
                f.seek(chunk_size * i)
                safe_readline(f)
                offsets[i] = f.tell()
            return offsets

    @staticmethod
    def tokenize(line, dict, tokenize=tokenize_line, add_if_not_exist=True,
                 consumer=None, append_eos=True, reverse_order=False):
        words = tokenize(line)
        if reverse_order:
            words = list(reversed(words))
        nwords = len(words)
        ids = torch.IntTensor(nwords + 1 if append_eos else nwords)

        for i, word in enumerate(words):
            if add_if_not_exist:
                idx = dict.add_symbol(word)
            else:
                idx = dict.index(word)
            if consumer is not None:
                consumer(word, idx)
            ids[i] = idx
        if append_eos:
            ids[nwords] = dict.eos_index
        return ids

    @staticmethod
    def segment(line, dict, ex_dict, is_tgt, maxspan,
                tokenize=tokenize_line, append_eos=True):
        """
        Run the segmentation for DPE decoder

        Args:
            dict : dictionary for the output of DPE decoder (a list of subwords)
            ex_dict: dictionary for the input of DPE decoder (a list of chars)
            max_span: the length of longest subword in dict (see algo 1)
        Returns:
            ids: ids
            pos_list: the ending position of a subword in sequence
            segs_list: all subwords starting at time step t
        """
        #words = tokenize(line)
        words = tokenize(line)
        nwords = len(words)

        if not is_tgt:
            ids = torch.IntTensor(nwords + 1 if append_eos else nwords)
            for i, word in enumerate(words):
                idx = dict.index(word)
                ids[i] = idx
            if append_eos:
                ids[nwords] = dict.eos_index
            return ids.long(), None, None, None

        ids = []
        segments = []
        lens = []
        current_pos = 0

        # construct a lookup list for all subwords starting at the position t
        # which will be used for algo 1 
        for i, word in enumerate(words):
            flags = [False for _ in range(len(word))]
            cur_word_segs = []
            for j in range(0, len(word)):
                segment = []
                ids.append(ex_dict.index(word[j]))
                # span of a subword cannot exceed the word boundary
                for k in range(j, len(word)):
                    if len(word[j:k+1]) > maxspan: break
                    abs_pos = current_pos + k
                    tok = dict.index(word[j:k+1])
                    if tok != dict.unk():
                        flags[k] = True
                        segment.append((abs_pos, tok))
                    else:
                        segment.append((abs_pos, dict.pad()))
                lens.append(len(segment))
                cur_word_segs.append(list(zip(*segment)))

            # sanity check for all positions within a word
            # if no subword starting from this position, we use <unk> for this position
            if not all(flags):
                for j, flag in enumerate(flags):
                    if not flag:
                        left_dis = j + 1
                        right_dis = len(word) - j
                        if left_dis < right_dis:
                            st = j - min(maxspan, j+1) + 1
                            ed = j - st
                            cur_word_segs[st][1] = list(cur_word_segs[st][1])
                            cur_word_segs[st][1][ed] = dict.unk()
                        else:
                            st = j
                            ed = min(maxspan, len(word)-j) - 1
                            cur_word_segs[st][1] = list(cur_word_segs[st][1])
                            cur_word_segs[st][1][ed] = dict.unk()

                        cur_word_segs[st][1] = tuple(cur_word_segs[st][1])
                
            segments.extend(cur_word_segs)
            current_pos += len(word)

        if append_eos:
            ids.append(ex_dict.eos())
            segments.append([(current_pos,), (dict.eos(),)])
            lens.append(1)

        ids = torch.tensor(ids, dtype=torch.long)
        # subwords and their ending positions
        pos_list, segs_list = list(zip(*segments))
        # conatenate segments as a list to save memory footprint
        pos_list = torch.tensor(reduce(lambda x,y:x+y, pos_list), dtype=torch.long)
        segs_list = torch.tensor(reduce(lambda x,y:x+y, segs_list), dtype=torch.long)

        return ids, pos_list, segs_list, torch.tensor(lens).long()
