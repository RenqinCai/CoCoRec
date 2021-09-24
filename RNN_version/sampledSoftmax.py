import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np

import sys
sys.path.insert(0, '../PyTorch_GBW_LM')

### compute logits for negative samples
class SampledSoftmax(nn.Module):
    def __init__(self, ntokens, nsampled, nhid, tied_weight):
        super(SampledSoftmax, self).__init__()

        # Parameters
        self.ntokens = ntokens
        self.nsampled = nsampled

        self.params = nn.Linear(nhid, ntokens)

        if tied_weight is not None:
            self.params.weight = tied_weight
        else:
            initialize(self.params.weight)

    def forward(self, inputs, labels, sample_ids, true_freq, sample_freq, acc_hits, device, remove_accidental_match):
        if self.training:
            return self.sampled(inputs, labels, sample_ids, true_freq, sample_freq, acc_hits, device, remove_accidental_match=remove_accidental_match)
        else:
            return self.full(inputs, labels)

    def sampled(self, inputs, labels, sample_ids, true_freq, sample_freq, acc_hits, device, remove_accidental_match=False):
        # assert(inputs.data.get_device() == labels.data.get_device())
        # device_id = labels.data.get_device()

        batch_size, d = inputs.size()
    
        sample_ids = torch.LongTensor(sample_ids).to(device)
        true_freq = torch.FloatTensor(true_freq).to(device)
        sample_freq = torch.FloatTensor(sample_freq).to(device)

        # gather true labels - weights and frequencies
        true_weights = torch.index_select(self.params.weight, 0, labels)
        true_bias = torch.index_select(self.params.bias, 0, labels)

        # gather sample ids - weights and frequencies
        sample_weights = torch.index_select(self.params.weight, 0, sample_ids)
        sample_bias = torch.index_select(self.params.bias, 0, sample_ids)

        # calculate logits
        true_logits = torch.sum(torch.mul(inputs, true_weights), dim=1) + true_bias
        sample_logits = torch.matmul(inputs, torch.t(sample_weights)) + sample_bias
        # remove true labels from sample set
        if remove_accidental_match:
            # acc_hits = self.sampler.accidental_match(labels.data.cpu().numpy(), sample_ids.data.cpu().numpy())
            # acc_hits = list(zip(*acc_hits))
            sample_logits[acc_hits] = -1e37

        # perform correction
        true_logits = true_logits.sub(torch.log(true_freq))
        sample_logits = sample_logits.sub(torch.log(sample_freq))

        # return logits and new_labels
        logits = torch.cat((torch.unsqueeze(true_logits, dim=1), sample_logits), dim=1)

        new_targets = torch.zeros(batch_size).long().to(device)
      
        return logits, new_targets

    def full(self, inputs, labels):
        return self.params(inputs), labels


def initialize(matrix):
    in_, out_ = matrix.size()
    stdv = math.sqrt(3. / (in_ + out_))
    matrix.data.uniform_(-stdv, stdv)
