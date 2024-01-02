import os
import pickle

import torch
from tensorboardX import SummaryWriter


class SingleLoss:
    def __init__(self, name: str, writer: SummaryWriter):
        self.name = name
        self.loss_step = []
        self.loss_epoch = []
        self.loss_epoch_tmp = []
        self.loss_epoch_val = []
        self.writer = writer

    def add_scalar(self, val, step=None, validation=False):
        if validation:
            self.loss_epoch_val.append(val)
        else:
            if step is None:
                step = len(self.loss_step)
            self.loss_step.append(val)
            self.loss_epoch_tmp.append(val)
            self.writer.add_scalar('Loss/step_' + self.name, val, step)

    def current(self):
        return self.loss_epoch[-1]

    def epoch(self, step=None):
        if step is None:
            step = len(self.loss_epoch)
        if self.loss_epoch_tmp:
            loss_avg = sum(self.loss_epoch_tmp) / len(self.loss_epoch_tmp)
        else:
            loss_avg = 0
        self.loss_epoch_tmp = []
        if self.loss_epoch_val:
            loss_avg_val = sum(self.loss_epoch_val) / len(self.loss_epoch_val)
        else:
            loss_avg_val = 0
        self.loss_epoch_val = []
        self.loss_epoch.append([loss_avg, loss_avg_val])
        self.writer.add_scalars('Loss/epoch_' + self.name, {'train': loss_avg, 'val': loss_avg_val}, step)

    def save(self):
        return {
            'loss_step': self.loss_step,
            'loss_epoch': self.loss_epoch
        }


class LossRecorder:
    def __init__(self, writer: SummaryWriter):
        self.losses = {}
        self.writer = writer

    def add_scalar(self, name, val, step=None, validation=False):
        if isinstance(val, torch.Tensor):
            val = val.item()
        if name not in self.losses:
            self.losses[name] = SingleLoss(name, self.writer)
        self.losses[name].add_scalar(val, step, validation)

    def epoch(self, step=None):
        for loss in self.losses.values():
            loss.epoch(step)

    def save(self, path):
        data = {}
        for key, value in self.losses.items():
            data[key] = value.save()
        with open(os.path.join(path, 'loss.pkl'), 'wb') as f:
            pickle.dump(data, f)

    def load(self, path):
        with open(os.path.join(path, 'loss.pkl'), 'rb') as f:
            data = pickle.load(f)
        for key, value in data.items():
            self.losses[key] = SingleLoss(key, self.writer)
            self.losses[key].loss_step = value['loss_step']
            self.losses[key].loss_epoch = value['loss_epoch']
