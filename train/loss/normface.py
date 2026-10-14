import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import accuracy

class LossFunction(nn.Module):
    def __init__(self, nOut, nClasses, scale=30, **kwargs):
        super().__init__()

        self.test_normalize = True

        self.s = scale
        self.in_feats = nOut
        self.W = torch.nn.Parameter(torch.randn(nOut, nClasses), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

        print(f'Initialised NormFace (NSL) s={self.s:.3f}')

    def forward(self, x, label=None):

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats

        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        costh_s = self.s * costh
        loss    = self.ce(costh_s, label)
        prec1   = accuracy(costh_s.detach(), label.detach(), topk=(1,))[0]
        return loss, prec1
