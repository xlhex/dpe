#!/usr/bin/env python3 -u
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Translate pre-processed data with a trained model.
"""
import sys
unused = [p for p in sys.path if ".local" in p]
for p in unused:
    sys.path.remove(p)

import torch

from fairseq import options, progress_bar, tasks, tokenizer, utils
from fairseq.meters import StopwatchMeter, TimeMeter


def compute_loss(model, net_output, sample, padding_idx):
    lprobs = model.get_normalized_probs(net_output, log_probs=True)
    segs_pos = sample["target"]["segs_pos"]
    segments = sample["target"]["segments"]
    bsz, step, dim = lprobs.size()
    padding_lprob = lprobs.new(bsz, 1, dim)
    mask = segments.eq(padding_idx)
    segments_lprobs = torch.gather(torch.cat([lprobs, padding_lprob], 1), -1, segments)
    segments_lprobs = segments_lprobs.masked_fill(mask, -1e30)
    rearrange = torch.full_like(segments_lprobs, -1e30).scatter_(1, segs_pos, segments_lprobs)
    rearrange = rearrange[:, :-1, :] # remove the padding step

    # dynamic programming for MAP
    padding_mask = sample['net_input']['prev_output_tokens'].ne(padding_idx).float()
    maxspan = rearrange.size(-1)
    acc_lprobs = lprobs.new_zeros(bsz, maxspan)
    book_keeper = []
    for i in range(step):
        intermidiate_lprobs, preceding_char = torch.max(rearrange[:, i, :] + acc_lprobs, dim=-1)
        book_keeper.append(preceding_char)
        inf_mask = torch.isinf(intermidiate_lprobs)
        intermidiate_lprobs = intermidiate_lprobs.masked_fill(inf_mask, 0.)
        prev_lprobs = (1-padding_mask[:, i]) * acc_lprobs[:, 0]
        intermidiate_lprobs = intermidiate_lprobs * padding_mask[:,i] + prev_lprobs

        # update accumulative lprobs
        acc_lprobs = torch.cat([torch.unsqueeze(intermidiate_lprobs, dim=-1), acc_lprobs[:, :-1]],
                               dim=-1)
    return torch.stack(book_keeper, dim=1).cpu()

def segmentation(ref, book_keeper, dict_in):
    book_keeper = (book_keeper+1).tolist()
    #print(book_keeper)
    pos = len(book_keeper) - 1
    seg_pos = []
    while pos >= 0:
        pos -= book_keeper[pos]
        seg_pos.append(pos)
    seg_pos = seg_pos[::-1]
    tgt = "".join(dict_in.string(ref).split())
    tgt = tgt.replace("<unk>", u"\u2753") # replace <unk> with ? to avoid collapse during backtracing
    segments = []
    for i in range(len(seg_pos)-1):
        st = seg_pos[i]+1
        ed = seg_pos[i+1]+1
        segments.append(tgt[st:ed])
    return tgt, " ".join(segments).replace(u"\u2753", "<unk>")

def search(model, sample, dict, dict_in, cuda=False, beam=4, maxspan=1, minlen=1,
           maxlen_a=0., maxlen_b=100, len_penalty=1., normalize_scores=True):

    if 'net_input' not in sample:
        return None

    sample = utils.move_to_cuda(sample) if cuda else sample
    net_input = sample['net_input']

    # compute scores for each model in the ensemble
    with torch.no_grad():
        decoder_out = model.forward(**net_input)
        book_keeper = compute_loss(model, decoder_out, sample, dict.pad()) 
        bsz = book_keeper.size(0)
        for i in range(bsz):
            #ref = utils.strip_pad(sample['net_input']['prev_output_tokens'].data[0, :], dict.pad()) if sample['target'] is not None else None
            ref = utils.strip_pad(sample['net_input']['prev_output_tokens'].data[i, :], dict.pad()) if sample['target'] is not None else None
            id = sample["id"][i].cpu().item()
            orig, segs = segmentation(ref, book_keeper[i, :len(ref)], dict_in) 
            print("{}:{}".format(id, segs))

def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    #assert args.max_sentences == 1
    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)
    print('| {} {} {} examples'.format(args.data, args.gen_subset, len(task.dataset(args.gen_subset))))

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary
    tgt_in_dict = task.ex_target_dictionary

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = utils.load_ensemble_for_inference(
        args.path.split(':'), task, model_arg_overrides=eval(args.model_overrides),
    )

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        model.eval()
        if use_cuda:
            model.cuda()
        if args.fp16:
            model.half()

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=8,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    ).next_epoch_itr(shuffle=False)

    with progress_bar.build_progress_bar(args, itr) as t:
        for sample in t:
            #if num_sentences > 10: break
            #num_sentences += 1
            search(models[0], sample, tgt_dict, tgt_in_dict,
                   cuda=use_cuda, beam=args.beam, 
                   minlen=args.min_len, maxlen_a=args.max_len_a,
                   maxlen_b=args.max_len_b, len_penalty=args.lenpen,
                   normalize_scores=(not args.unnormalized))

if __name__ == '__main__':
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)
