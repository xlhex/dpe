# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('dynamic_programming_cross_entropy')
class DPCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        #print("loss")
        #print(loss)
        sample_size = sample['net_input']["prev_output_tokens"].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['net_input']["prev_output_tokens"].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        segs_pos = sample["target"]["segs_pos"]
        assert torch.isnan(lprobs.detach()).sum().item() == 0
        assert torch.isinf(lprobs.detach()).sum().item() == 0
        segments = sample["target"]["segments"]
        bsz, step, dim = lprobs.size()
        padding_lprob = lprobs.new(bsz, 1, dim)
        mask = segments.eq(self.padding_idx)
        assert (segments >= dim).sum().item() == 0
        segments_lprobs = torch.gather(torch.cat([lprobs, padding_lprob], 1), -1, segments)
        segments_lprobs = segments_lprobs.masked_fill(mask, -1e30)

        # rearrange the order of segments from start-oriented to ending-oriented, so that one can use
        # DP for marginalization at each position, ie backtrace all segments ending at time step t
        rearrange = torch.full_like(segments_lprobs, -1e30).scatter_(1, segs_pos, segments_lprobs)
        rearrange = rearrange[:, :-1, :] # remove the padding step

        # dynamic programming
        padding_mask = sample['net_input']['prev_output_tokens'].ne(self.padding_idx).float()
        maxspan = rearrange.size(-1)
        acc_lprobs = lprobs.new_zeros(bsz, maxspan)
        for i in range(step):
            intermidiate_lprobs = torch.logsumexp(rearrange[:, i, :] + acc_lprobs, dim=-1)
            inf_mask = torch.isinf(intermidiate_lprobs)
            intermidiate_lprobs = intermidiate_lprobs.masked_fill(inf_mask, 0.)
            # copy losses if there is a padding
            prev_lprobs = (1-padding_mask[:, i]) * acc_lprobs[:, 0]
            intermidiate_lprobs = intermidiate_lprobs * padding_mask[:,i] + prev_lprobs

            # update accumulative lprobs
            acc_lprobs = torch.cat([torch.unsqueeze(intermidiate_lprobs, dim=-1), acc_lprobs[:, :-1]],
                                   dim=-1)
        loss = -acc_lprobs[:, 0]
        if reduce:
            loss = loss.sum()
        return loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
