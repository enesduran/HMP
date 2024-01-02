import math
import os
from abc import ABC, abstractmethod

import torch
import torch.nn.init
import torch.optim


def weights_init(init_type='default'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                torch.nn.init.normal_(m.weight.data)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                torch.nn.init.normal_(m.weight.data)
                weight_norm = m.weight.pow(2).sum(2, keepdim=True).sum(1, keepdim=True).add(1e-8).sqrt()
                m.weight.data.div_(weight_norm)
            else:
                raise NotImplementedError(f"Unsupported initialization: {init_type}")
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
    return init_fun


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
    """

    def __init__(self, args):
        self.args = args
        self.is_train = args.is_train
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_save_dir = os.path.join(args.save_dir, 'checkpoints')  # save all the checkpoints to save_dir

        if self.is_train and args.log:
            import time
            from tensorboardX import SummaryWriter
            from .loss_record import LossRecorder

            log_dir = os.path.join(args.save_dir, 'logs')
            if args.epoch_begin != 0:
                all = [d for d in os.listdir(log_dir)]
                if len(all) == 0:
                    raise RuntimeError(f'Empty logging path {log_dir}')
                timestamp = sorted(all)[-1]
            else:
                timestamp = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime(time.time()))
            self.log_path = os.path.join(log_dir, timestamp)
            os.makedirs(self.log_path, exist_ok=True)
            self.writer = SummaryWriter(self.log_path)
            self.loss_recorder = LossRecorder(self.writer)

        self.epoch_cnt = 0
        self.schedulers = []
        self.optimizers = []
        self.models = []

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def compute_test_result(self):
        """
        After forward, do something like output bvh, get error value
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def get_scheduler(self, optimizer):
        if self.args.scheduler.name == 'linear':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch - self.args.n_epochs_origin) / float(self.args.n_epochs_decay + 1)
                return lr_l
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        if self.args.scheduler.name == 'StepLR':
            print('StepLR scheduler set')
            return torch.optim.lr_scheduler.StepLR(optimizer, self.args.scheduler.step_size, self.args.scheduler.gamma, verbose=self.args.verbose)
        if self.args.scheduler.name == 'Plateau':
            print('ReduceLROnPlateau shceduler set')
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.args.scheduler.factor, threshold=self.args.scheduler.threshold, patience=5, verbose=True)
        if self.args.scheduler.name == 'MultiStep':
            print('MultiStep shceduler set')
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.scheduler.milestones, gamma=self.args.scheduler.gamma, verbose=self.args.verbose)
        else:
            print('No scheduler set')
            return None

    def setup(self):
        """Load and print networks; create schedulers
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.is_train:
            self.schedulers = [self.get_scheduler(optimizer) for optimizer in self.optimizers]

    def epoch(self):
        if self.args.verbose:
            self.loss_recorder.epoch()
        for scheduler in self.schedulers:
            if scheduler is not None:
                if self.args.scheduler.name == 'Plateau':
                    loss = self.loss_recorder.losses['total_loss'].current()
                    scheduler.step(loss[1])
                else:
                    scheduler.step()
        self.epoch_cnt += 1

    def test(self):
        """Forward function used in test time.
        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_test_result()

    def train(self):
        for model in self.models:
            model.train()
            for param in model.parameters():
                param.requires_grad = True

    def eval(self):
        for model in self.models:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

    def print(self):
        for model in self.models:
            print(model)

    def count_params(self):
        params = 0
        for model in self.models:
            params += sum(p.numel() for p in model.parameters() if p.requires_grad)

        return params
