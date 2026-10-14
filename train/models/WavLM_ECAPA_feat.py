import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from models.ECAPA_TDNN_feat import Bottle2neck


class WavLM_ECAPA(nn.Module):
    def __init__(self, nOut=192, C=512, wavlm="WAVLM_BASE_PLUS", finetune=False,
                 n_mels=40, log_input=True, encoder_type="ASP", **kwargs):
        super().__init__()
        self.finetune = finetune

        bundle = getattr(torchaudio.pipelines, wavlm)
        self.sample_rate = bundle.sample_rate
        p = getattr(bundle, "_params", {}) or {}
        feat_dim = p.get("encoder_embed_dim")
        n_layers = p.get("encoder_num_layers")
        self.wavlm = bundle.get_model()
        if feat_dim is None or n_layers is None:
            with torch.no_grad():
                fl, _ = self.wavlm.extract_features(torch.zeros(1, self.sample_rate))
            n_layers, feat_dim = len(fl), fl[0].shape[-1]
        self.n_layers, self.feat_dim = n_layers, feat_dim
        print(f'WavLM-SV: {wavlm} ({"PHASE2 finetune-encoder" if finetune else "PHASE1 frozen"}, '
              f'{n_layers} layers × {feat_dim}-d) + learnable weighted-sum + ECAPA-TDNN(C={C}) -> emb {nOut}.')

        self._set_wavlm_trainable()

        self.layer_weights = nn.Parameter(torch.zeros(n_layers))
        self.instancenorm = nn.InstanceNorm1d(feat_dim)

        self.conv1 = nn.Conv1d(feat_dim, C, kernel_size=5, stride=1, padding=2)
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

    def _set_wavlm_trainable(self):
        for prm in self.wavlm.feature_extractor.parameters():
            prm.requires_grad_(False)
        for prm in self.wavlm.encoder.parameters():
            prm.requires_grad_(self.finetune)
        self.wavlm.feature_extractor.eval()
        if not self.finetune:
            self.wavlm.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.wavlm.feature_extractor.eval()
        if not self.finetune:
            self.wavlm.encoder.eval()
        return self

    def _wavlm_weighted_sum(self, wav):
        if self.finetune:
            feats, _ = self.wavlm.extract_features(wav)
        else:
            with torch.no_grad():
                feats, _ = self.wavlm.extract_features(wav)
        stack = torch.stack(feats, dim=0)
        w = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        return (stack * w).sum(dim=0)

    def forward(self, x):
        if x.dim() != 2:
            raise ValueError(
                f"WavLM_ECAPA is a raw-waveform backbone: forward() expects [N, samples] (2-D), got "
                f"{tuple(x.shape)}. Train it on raw wav (configs/wavlm.py sets USE_SHARDS = False).")
        with torch.amp.autocast("cuda", enabled=False):
            feat = self._wavlm_weighted_sum(x.float())
            feat = self.instancenorm(feat.transpose(1, 2))

        x = self.conv1(feat)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)

        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)

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


def MainModel(nOut=192, **kwargs):
    import os
    finetune = os.environ.get("WAVLM_FINETUNE", "0") == "1"
    return WavLM_ECAPA(nOut=nOut, C=kwargs.pop("C", 512), wavlm="WAVLM_BASE_PLUS", finetune=finetune, **kwargs)
