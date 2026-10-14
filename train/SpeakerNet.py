import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import itertools
import math
import random

logger = logging.getLogger(__name__)

import numpy
from tqdm import tqdm

from DatasetLoader import test_dataset_loader
from torch.amp import autocast, GradScaler


def _snapshot_batchnorm_buffers(model):
    snapshots = []
    for module in model.modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue
        if module.running_mean is None or module.running_var is None:
            continue
        snapshots.append((
            module,
            module.running_mean.detach().clone(),
            module.running_var.detach().clone(),
            module.num_batches_tracked.detach().clone() if module.num_batches_tracked is not None else None,
        ))
    return snapshots


def _restore_batchnorm_buffers(snapshots):
    with torch.no_grad():
        for module, running_mean, running_var, batches in snapshots:
            module.running_mean.copy_(running_mean)
            module.running_var.copy_(running_var)
            if batches is not None:
                module.num_batches_tracked.copy_(batches)


def _nonfinite_tensor_names(value, prefix=""):
    if torch.is_tensor(value):
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            yield prefix
    elif isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _nonfinite_tensor_names(child, child_prefix)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _nonfinite_tensor_names(child, f"{prefix}[{index}]")


def _nonfinite_gradient_names(model):
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            yield name


class WrappedModel(nn.Module):


    def __init__(self, model):
        super().__init__()
        self.module = model

    def forward(self, x, label=None):
        return self.module(x, label)


class SpeakerNet(nn.Module):
    def __init__(self, model, optimizer, trainfunc, nPerSpeaker, **kwargs):
        super().__init__()

        SpeakerNetModel = importlib.import_module("models." + model).__getattribute__("MainModel")
        self.__S__ = SpeakerNetModel(**kwargs)

        LossFunction = importlib.import_module("loss." + trainfunc).__getattribute__("LossFunction")
        self.__L__ = LossFunction(**kwargs)

        self.nPerSpeaker = nPerSpeaker

    def forward(self, data, label=None):

        if data.dim() == 4:
            data = data.reshape(-1, data.size(-2), data.size(-1)).to(next(self.parameters()).device)
        else:
            data = data.reshape(-1, data.size()[-1]).to(next(self.parameters()).device)
        outp = self.__S__.forward(data)

        if label is None:
            return outp

        else:

            outp = outp.reshape(self.nPerSpeaker, -1, outp.size()[-1]).transpose(1, 0).squeeze(1)

            nloss, prec1 = self.__L__.forward(outp, label)

            return nloss, prec1


class ModelTrainer:
    def __init__(self, speaker_model, optimizer, scheduler, gpu, mixedprec, **kwargs):

        self.__model__ = speaker_model

        Optimizer = importlib.import_module("optimizer." + optimizer).__getattribute__("Optimizer")
        self.__optimizer__ = Optimizer(self.__model__.parameters(), **kwargs)

        Scheduler = importlib.import_module("scheduler." + scheduler).__getattribute__("Scheduler")
        self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__, **kwargs)

        self.scaler = GradScaler("cuda")

        self.gpu = gpu
        self.device = torch.device(f"cuda:{gpu}")

        self.mixedprec = mixedprec

        assert self.lr_step in ["epoch", "iteration"]


    def train_network(self, loader, verbose):

        self.__model__.train()

        counter = 0
        loss = 0
        top1 = 0
        skipped_nonfinite = 0

        it = iter(loader)
        total = len(loader)
        with tqdm(it, unit="batch", total=total, disable=not verbose) as tepoch:

            for batch_index, (data, data_label) in enumerate(tepoch, start=1):

                data = data.transpose(1, 0)

                self.__model__.zero_grad()

                label = torch.LongTensor(data_label).to(self.device)
                batchnorm_snapshot = _snapshot_batchnorm_buffers(self.__model__)

                if self.mixedprec:
                    with autocast("cuda"):
                        nloss, prec1 = self.__model__(data, label)
                    loss_value = float(nloss.detach().float().cpu().item())
                    if not math.isfinite(loss_value):
                        _restore_batchnorm_buffers(batchnorm_snapshot)
                        skipped_nonfinite += 1
                        logger.warning("Skipped non-finite training batch %d/%d (skipped=%d).",
                                       batch_index, total, skipped_nonfinite)
                        continue
                    self.scaler.scale(nloss).backward()
                    self.scaler.step(self.__optimizer__)
                    self.scaler.update()
                else:
                    nloss, prec1 = self.__model__(data, label)
                    loss_value = float(nloss.detach().float().cpu().item())
                    if not math.isfinite(loss_value):
                        _restore_batchnorm_buffers(batchnorm_snapshot)
                        skipped_nonfinite += 1
                        logger.warning("Skipped non-finite training batch %d/%d (skipped=%d).",
                                       batch_index, total, skipped_nonfinite)
                        continue
                    nloss.backward()
                    bad_gradients = list(_nonfinite_gradient_names(self.__model__))
                    if bad_gradients:
                        _restore_batchnorm_buffers(batchnorm_snapshot)
                        self.__model__.zero_grad()
                        skipped_nonfinite += 1
                        logger.warning("Skipped batch %d/%d with non-finite gradients (skipped=%d).",
                                       batch_index, total, skipped_nonfinite)
                        continue
                    self.__optimizer__.step()

                loss += loss_value
                top1 += prec1.detach().cpu().item()
                counter += 1

                tepoch.set_postfix(loss=loss / counter)

                if self.lr_step == "iteration":
                    self.__scheduler__.step()

        if counter == 0:
            raise FloatingPointError("all training batches produced non-finite loss")
        bad_state = list(_nonfinite_tensor_names(self.__model__.state_dict(), "model_state_dict"))
        bad_state.extend(_nonfinite_tensor_names(self.__optimizer__.state_dict(), "optimizer_state_dict"))
        if bad_state:
            raise FloatingPointError(f"training epoch produced {len(bad_state)} non-finite state tensors: "
                                     + ", ".join(bad_state[:8]))
        if self.lr_step == "epoch":
            self.__scheduler__.step()
        return (loss / counter, top1 / counter)


    def evaluateFromList(self, test_list, test_path, nDataLoaderThread, num_eval=10, **kwargs):

        self.__model__.eval()

        feats = {}

        with open(test_list) as f:
            lines = f.readlines()

        files = list(itertools.chain(*[x.strip().split()[-2:] for x in lines]))
        setfiles = list(set(files))
        setfiles.sort()

        test_dataset = test_dataset_loader(setfiles, test_path, num_eval=num_eval, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=nDataLoaderThread, drop_last=False)

        with tqdm(test_loader, unit="utt", desc="Extracting") as tepoch:
            for data in tepoch:
                inp1 = data[0][0].to(self.device)
                with torch.no_grad():
                    ref_feat = self.__model__(inp1).detach().cpu()
                feats[data[1][0]] = ref_feat

        all_scores = []
        all_labels = []
        all_trials = []

        with tqdm(lines, unit="trial", desc="Scoring") as tepoch:
            for line in tepoch:

                data = line.split()

                if len(data) == 2:
                    data = [random.randint(0, 1)] + data

                ref_feat = feats[data[1]].to(self.device)
                com_feat = feats[data[2]].to(self.device)

                if self.__model__.module.__L__.test_normalize:
                    ref_feat = F.normalize(ref_feat, p=2, dim=1)
                    com_feat = F.normalize(com_feat, p=2, dim=1)

                dist = torch.cdist(ref_feat.reshape(num_eval, -1), com_feat.reshape(num_eval, -1)).detach().cpu().numpy()

                score = -1 * numpy.mean(dist)

                all_scores.append(score)
                all_labels.append(int(data[0]))
                all_trials.append(data[1] + " " + data[2])

        return (all_scores, all_labels, all_trials)


    def saveParameters(self, path, epoch):

        tmp = f"{path}.tmp.{os.getpid()}"
        torch.save({
            "model_state_dict": self.__model__.module.state_dict(),
            "optimizer_state_dict": self.__optimizer__.state_dict(),
            "scheduler_state_dict": self.__scheduler__.state_dict(),
            "epoch": epoch,
        }, tmp)
        try:
            with open(tmp, "rb") as _f:
                os.fsync(_f.fileno())
        except OSError:
            pass
        os.replace(tmp, path)


    def loadParameters(self, path, restore_optim=True):

        self_state = self.__model__.module.state_dict()
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        if "model_state_dict" in checkpoint:
            loaded_state = checkpoint["model_state_dict"]
            if restore_optim:
                try:
                    self.__optimizer__.load_state_dict(checkpoint["optimizer_state_dict"])
                except ValueError:
                    logger.warning("Optimizer state could not be restored; resuming with the initial learning rate.")
                try:
                    self.__scheduler__.load_state_dict(checkpoint["scheduler_state_dict"])
                except ValueError:
                    logger.warning("Scheduler state not compatible with the current model; skipping restore.")
                epoch = checkpoint["epoch"]
            else:
                epoch = None
        else:
            logger.warning("Loading a checkpoint with model weights only; optimizer and scheduler states are not restored.")
            loaded_state = checkpoint
            epoch = None

        if len(loaded_state.keys()) == 1 and "model" in loaded_state:
            loaded_state = {"__S__." + name: param for name, param in loaded_state["model"].items()}

        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")

                if name not in self_state:
                    logger.warning(f"{origname} is not in the model.")
                    continue

            if self_state[name].size() != loaded_state[origname].size():
                logger.warning(f"Wrong parameter length: {origname}, model: {self_state[name].size()}, loaded: {loaded_state[origname].size()}")
                continue

            self_state[name].copy_(param)

        return epoch
