'''
This is the ECAPA-TDNN model.
This model is modified and combined based on the following three projects:
  1. https://github.com/clovaai/voxceleb_trainer/issues/86
  2. https://github.com/lawlict/ECAPA-TDNN/blob/master/ecapa_tdnn.py
  3. https://github.com/speechbrain/speechbrain/blob/96077e9a1afff89d3f5ff47cab4bca0202770e4f/speechbrain/lobes/models/ECAPA_TDNN.py

(attribution docstring of https://github.com/TaoRuijie/ECAPA-TDNN/blob/main/model.py, MIT License,
Copyright (c) 2022 Tao Ruijie; see ../NOTICE.md)
'''

import math
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x


class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        convs = []
        bns = []
        num_pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.se(out)
        out += residual
        return out


class ECAPA_TDNN(nn.Module):
    def __init__(self, C=1024, nOut=192, n_mels=80, log_input=True, **kwargs):
        super(ECAPA_TDNN, self).__init__()
        print(f'Embedding size is {nOut:d}, ECAPA-TDNN (C={C}).')

        self.n_mels = n_mels
        self.log_input = log_input

        self.torchfb = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=512, win_length=400, hop_length=160,
            window_fn=torch.hamming_window, n_mels=n_mels)

        self.conv1 = nn.Conv1d(n_mels, C, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, nOut)
        self.bn6 = nn.BatchNorm1d(nOut)

    def forward(self, x):
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=False):
                if x.dim() == 2:
                    x = self.torchfb(x) + 1e-6
                    if self.log_input:
                        x = x.log()
                x = (x - x.mean(dim=-1, keepdim=True)).detach()

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            t = x.size()[-1]
            global_x = torch.cat((
                x,
                torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t),
            ), dim=1)

            w = self.attention(global_x)

            mu = torch.sum(x * w, dim=2)
            sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))

            x = torch.cat((mu, sg), 1)
            x = self.bn5(x)
            x = self.fc6(x)
            x = self.bn6(x)

        return x


def MainModel(nOut=192, n_mels=80, log_input=True, C=1024, **kwargs):
    return ECAPA_TDNN(C=C, nOut=nOut, n_mels=n_mels, log_input=log_input)
