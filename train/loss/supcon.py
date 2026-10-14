import torch
import torch.nn as nn
import torch.nn.functional as F


class LossFunction(nn.Module):
    def __init__(self, temperature=0.1, **kwargs):
        super().__init__()

        self.test_normalize = True
        self.temperature = temperature

        print(f'Initialised SupCon (temperature={temperature})')

    def forward(self, x, label=None):

        assert x.size()[1] >= 2

        N, M, D = x.size()
        feat   = F.normalize(x.reshape(N * M, D), p=2, dim=1)
        labels = torch.arange(N, device=x.device).repeat_interleave(M)
        NM = N * M

        logits = torch.mm(feat, feat.t()) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

        self_mask = torch.eye(NM, dtype=torch.bool, device=x.device)
        pos_mask  = (labels.unsqueeze(0) == labels.unsqueeze(1)) & (~self_mask)

        exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
        log_prob   = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        pos_count  = pos_mask.sum(dim=1).clamp(min=1)
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_count
        nloss = -mean_log_prob_pos.mean()

        with torch.no_grad():
            sim = logits.masked_fill(self_mask, float('-inf'))
            top1 = sim.argmax(dim=1)
            prec1 = pos_mask[torch.arange(NM, device=x.device), top1].float().mean() * 100.0

        return nloss, prec1
