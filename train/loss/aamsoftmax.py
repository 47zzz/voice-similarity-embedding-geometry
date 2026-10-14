# Adapted from https://github.com/wujiyang/Face_Pytorch (Apache License)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import accuracy

class LossFunction(nn.Module):
    def __init__(self, nOut, nClasses, margin=0.3, scale=15, easy_margin=False, **kwargs):
        super().__init__()

        self.test_normalize = True

        self.m = margin
        self.s = scale
        self.in_feats = nOut
        self.weight = torch.nn.Parameter(torch.FloatTensor(nClasses, nOut), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)

        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        self.norm_eps = 1e-8
        self.cosine_eps = 1e-6

        print(f'Initialised AAMSoftmax margin {self.m:.3f} scale {self.s:.3f}')

    def forward(self, x, label=None):

        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats

        with torch.autocast(device_type=x.device.type, enabled=False):
            cosine = F.linear(
                F.normalize(x.float(), p=2, dim=1, eps=self.norm_eps),
                F.normalize(self.weight.float(), p=2, dim=1, eps=self.norm_eps),
            )
            cosine = cosine.clamp(min=-1.0 + self.cosine_eps, max=1.0 - self.cosine_eps)
            sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp_min(self.cosine_eps))
            phi = cosine * self.cos_m - sine * self.sin_m

            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, label.view(-1, 1), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output = output * self.s

            loss    = self.ce(output, label)
            prec1   = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        return loss, prec1
