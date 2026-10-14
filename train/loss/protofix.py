import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import accuracy

class LossFunction(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.test_normalize = False

        self.criterion  = torch.nn.CrossEntropyLoss()

        print('Initialised Prototypical Loss (squared-Euclidean, torch.cdist (N,N))')

    def forward(self, x, label=None):

        assert x.size()[1] >= 2

        out_anchor      = torch.mean(x[:,1:,:],1)
        out_positive    = x[:,0,:]
        stepsize        = out_anchor.size()[0]

        output  = -1 * (torch.cdist(out_positive, out_anchor, p=2) ** 2)
        label   = torch.arange(stepsize, device=x.device)
        nloss   = self.criterion(output, label)
        prec1   = accuracy(output.detach(), label.detach(), topk=(1,))[0]

        return nloss, prec1
