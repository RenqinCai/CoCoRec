import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class LossFunction(nn.Module):
    def __init__(self, loss_type='XE', use_cuda=False):
        """ An abstract loss function that can supports custom loss functions compatible with PyTorch."""
        super(LossFunction, self).__init__()
        self.loss_type = loss_type
        self.use_cuda = use_cuda
        
        if loss_type == "XE":
            print(loss_type)
            self._loss_fn = SampledCrossEntropyLoss(use_cuda)

    def forward(self, input_, target_, type="logit"):
        return self._loss_fn(input_, target_, type)

class SampledCrossEntropyLoss(nn.Module):
    """ CrossEntropyLoss with n_classes = batch_size = the number of samples in the session-parallel mini-batch """
    def __init__(self, use_cuda):
        """
        Args:
             use_cuda (bool): whether to use cuda or not
        """
        super(SampledCrossEntropyLoss, self).__init__()
        self.xe_loss = nn.CrossEntropyLoss()
        self.use_cuda = use_cuda

    def forward(self, input_, target_, type="logit"):
        
        ### the type of input is logit
        if type=="logit":
            logit = input_
            target = target_

            loss = self.xe_loss(logit.view(target.size(0), -1), target)

            return loss

        ### the type of input is prob
        if type=="prob":
            log_likelihood = input_
            loss = -torch.mean(log_likelihood[:, 0])
    
            return loss
