# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch

from fairseq import utils

from . import data_utils, LanguagePairDataset


def collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, item=None, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] if item is None else s[key][item] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    def split(tensors, key, maxspan=False):
        """
        The original segments and positions are concatenated as as list for the sake of memory-saving.
        Now split them into chunks.
        """
        tensors = torch.split(tensors, key.tolist())

        if maxspan:
            maxspan = max(len(t) for t in tensors)
            return tensors, maxspan

        return tensors
    

    def merge_tgt():
        # positions of each segment
        segs_pos = [split(s["target"][1], s["target"][3], True) for s in samples]
        # segments
        segments = [split(s["target"][2], s["target"][3]) for s in samples]

        maxspan = max([v[1] for v in segs_pos])
        max_step = max(map(lambda x:len(x), segments))
        bsz = len(samples)

        segs_pos_tensor = torch.full((bsz, max_step+1, maxspan), max_step, dtype=torch.long)
        segments_tensor = torch.full((bsz, max_step+1, maxspan), pad_idx, dtype=torch.long)

        for i in range(bsz):
            for j, segment in enumerate(segments[i]):
                segs_pos_tensor[i][j][:len(segment)].copy_(segs_pos[i][0][j])
                segments_tensor[i][j][:len(segment)].copy_(segment)
                #segs_pos_tensor[i][j][:len(segment)] = torch.tensor(segs_pos[i][0][j]).long()
                #segments_tensor[i][j][:len(segment)] = torch.tensor(segment).long()

        return segs_pos_tensor, segments_tensor

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'][0].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        # maxspan = max([s["target"][-1] for s in samples])
        segs_pos, segments = merge_tgt()
        segs_pos = segs_pos.index_select(0, sort_order)
        segments = segments.index_select(0, sort_order)
        ntokens = sum(len(s['target'][0]) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                item=0,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)

    else:
        ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': {
            'segs_pos': segs_pos,
            'segments': segments,
        },
        'nsentences': samples[0]['source'].size(0),
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


class LanguagePairSegDataset(LanguagePairDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target output vocabulary
        tgt_in_dict (~fairseq.data.Dictionary, optional): target input vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side.
            Default: ``True``
        left_pad_target (bool, optional): pad target tensors on the left side.
            Default: ``False``
        max_source_positions (int, optional): max number of tokens in the source
            sentence. Default: ``1024``
        max_target_positions (int, optional): max number of tokens in the target
            sentence. Default: ``1024``
        shuffle (bool, optional): shuffle dataset elements before batching.
            Default: ``True``
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for input feeding/teacher forcing.
            Default: ``True``
        remove_eos_from_source (bool, optional): if set, removes eos from end of
            source if it's present. Default: ``False``
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent. Default: ``False``
    """
    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None, tgt_in_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True, remove_eos_from_source=False, append_eos_to_target=False,
    ):
        super().__init__(src, src_sizes, src_dict,
                         tgt, tgt_sizes, tgt_dict,
                         left_pad_source, left_pad_target,
                         max_source_positions, max_target_positions,
                         shuffle, input_feeding, remove_eos_from_source, append_eos_to_target)
        if tgt_in_dict is not None:
            assert tgt_in_dict.pad() == tgt_dict.pad()
            assert tgt_in_dict.eos() == tgt_dict.eos()
            assert tgt_in_dict.unk() == tgt_dict.unk()
        self.tgt_in_dict = tgt_in_dict

    def __getitem__(self, index):
        tgt_item, tgt_segs_pos, tgt_segs, tgt_segs_len = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][0][-1] != eos:
                tgt_item = torch.cat([self.tgt[index][0], torch.LongTensor([eos])])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        return {
            'id': index,
            'source': src_item,
            'target': (tgt_item, tgt_segs_pos, tgt_segs, tgt_segs_len),
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=128):
        """Return a dummy batch with a given number of tokens."""
        src_len, tgt_len = utils.resolve_max_positions(
            (src_len, tgt_len),
            max_positions,
            (self.max_source_positions, self.max_target_positions),
        )
        bsz = max(num_tokens // max(src_len, tgt_len), 1)
        return self.collater([
            {
                'id': i,
                'source': self.src_dict.dummy_sentence(src_len),
                'target': (self.tgt_in_dict.dummy_sentence(tgt_len), *self.tgt_dict.dummy_sentence(tgt_len, True))
            }
            for i in range(bsz)
        ])
