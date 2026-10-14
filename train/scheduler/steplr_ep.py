import torch


def Scheduler(optimizer, lr_decay, **kwargs):

    sche_fn = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay)

    lr_step = 'epoch'

    print(f'Initialised per-epoch StepLR scheduler (gamma={lr_decay} every epoch)')

    return sche_fn, lr_step
