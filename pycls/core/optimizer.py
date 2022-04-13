#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Optimizer."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from pycls.core.config import cfg
from collections import deque


def construct_optimizer(model):
    """Constructs the optimizer.

    Note that the momentum update in PyTorch differs from the one in Caffe2.
    In particular,

        Caffe2:
            V := mu * V + lr * g
            p := p - V

        PyTorch:
            V := mu * V + g
            p := p - lr * V

    where V is the velocity, mu is the momentum factor, lr is the learning rate,
    g is the gradient and p are the parameters.

    Since V is defined independently of the learning rate in PyTorch,
    when the learning rate is changed there is no need to perform the
    momentum correction by scaling V (unlike in the Caffe2 case).
    """
    # Split parameters into types and get weight decay for each type
    optim, wd, params = cfg.OPTIM, cfg.OPTIM.WEIGHT_DECAY, [[], [], [], []]
    for n, p in model.named_parameters():
        ks = [k for (k, x) in enumerate(["bn", "ln", "bias", ""]) if x in n]
        params[ks[0]].append(p)
    wds = [
        cfg.BN.CUSTOM_WEIGHT_DECAY if cfg.BN.USE_CUSTOM_WEIGHT_DECAY else wd,
        cfg.LN.CUSTOM_WEIGHT_DECAY if cfg.LN.USE_CUSTOM_WEIGHT_DECAY else wd,
        optim.BIAS_CUSTOM_WEIGHT_DECAY if optim.BIAS_USE_CUSTOM_WEIGHT_DECAY else wd,
        wd,
    ]
    param_wds = [{"params": p, "weight_decay": w} for (p, w) in zip(params, wds) if p]
    # Set up optimizer
    if optim.OPTIMIZER == "sgd":
        optimizer = torch.optim.SGD(
            param_wds,
            lr=optim.BASE_LR,
            momentum=optim.MOMENTUM,
            weight_decay=wd,
            dampening=optim.DAMPENING,
            nesterov=optim.NESTEROV,
        )
    elif optim.OPTIMIZER == "adam":
        optimizer = torch.optim.Adam(
            param_wds,
            lr=optim.BASE_LR,
            betas=(optim.BETA1, optim.BETA2),
            weight_decay=wd,
        )
    elif optim.OPTIMIZER == "adamw":
        optimizer = torch.optim.AdamW(
            param_wds,
            lr=optim.BASE_LR,
            betas=(optim.BETA1, optim.BETA2),
            weight_decay=wd,
        )
    elif optim.OPTIMIZER == "vanilla_sgd":
        optimizer = VanillaSGD(
            param_wds,
            lr=optim.BASE_LR,
            momentum=optim.MOMENTUM,
            weight_decay=wd,
            dampening=optim.DAMPENING,
            nesterov=optim.NESTEROV,
        )
    else:
        raise NotImplementedError
    return optimizer


def lr_fun_steps(cur_epoch):
    """Steps schedule (cfg.OPTIM.LR_POLICY = 'steps')."""
    ind = [i for i, s in enumerate(cfg.OPTIM.STEPS) if cur_epoch >= s][-1]
    return cfg.OPTIM.LR_MULT ** ind


def lr_fun_exp(cur_epoch):
    """Exponential schedule (cfg.OPTIM.LR_POLICY = 'exp')."""
    return cfg.OPTIM.MIN_LR ** (cur_epoch / cfg.OPTIM.MAX_EPOCH)


def lr_fun_cos(cur_epoch):
    """Cosine schedule (cfg.OPTIM.LR_POLICY = 'cos')."""
    lr = 0.5 * (1.0 + np.cos(np.pi * cur_epoch / cfg.OPTIM.MAX_EPOCH))
    return (1.0 - cfg.OPTIM.MIN_LR) * lr + cfg.OPTIM.MIN_LR


def lr_fun_lin(cur_epoch):
    """Linear schedule (cfg.OPTIM.LR_POLICY = 'lin')."""
    lr = 1.0 - cur_epoch / cfg.OPTIM.MAX_EPOCH
    return (1.0 - cfg.OPTIM.MIN_LR) * lr + cfg.OPTIM.MIN_LR


def get_lr_fun():
    """Retrieves the specified lr policy function"""
    lr_fun = "lr_fun_" + cfg.OPTIM.LR_POLICY
    assert lr_fun in globals(), "Unknown LR policy: " + cfg.OPTIM.LR_POLICY
    err_str = "exp lr policy requires OPTIM.MIN_LR to be greater than 0."
    assert cfg.OPTIM.LR_POLICY != "exp" or cfg.OPTIM.MIN_LR > 0, err_str
    return globals()[lr_fun]


def get_epoch_lr(cur_epoch):
    """Retrieves the lr for the given epoch according to the policy."""
    # Get lr and scale by by BASE_LR
    lr = get_lr_fun()(cur_epoch) * cfg.OPTIM.BASE_LR
    # Linear warmup
    if cur_epoch < cfg.OPTIM.WARMUP_EPOCHS:
        alpha = cur_epoch / cfg.OPTIM.WARMUP_EPOCHS
        warmup_factor = cfg.OPTIM.WARMUP_FACTOR * (1.0 - alpha) + alpha
        lr *= warmup_factor
    return lr

def set_lr(optimizer, new_lr):
    """Sets the optimizer lr to the specified value."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def plot_lr_fun():
    """Visualizes lr function."""
    epochs = list(range(cfg.OPTIM.MAX_EPOCH))
    lrs = [get_epoch_lr(epoch) for epoch in epochs]
    plt.plot(epochs, lrs, ".-")
    plt.title("lr_policy: {}".format(cfg.OPTIM.LR_POLICY))
    plt.xlabel("epochs")
    plt.ylabel("learning rate")
    plt.ylim(bottom=0)
    plt.show()

def get_lrs1():
    lr=cfg.OPTIM.BASE_LR
    if hasattr(get_lrs1,"prev_lr"):
        lr=get_lrs1.prev_lr
    num=5
    lrs=np.linspace(lr/2,lr*1.5,num=num)
    lrs=np.round(lrs,4)
    lrs=list(lrs)
    return lrs

def get_neighbour_lrs():
    # return [0,0.01,0.02,0.05,0.1,0.5,1.0]
    lr=cfg.OPTIM.BASE_LR
    num=5
    lrs=np.linspace(lr/2,lr*1.5,num=num)
    lrs=np.round(lrs,4)
    lrs=list(lrs)
    return lrs

class LR_Finder:
    def __init__(self,version=1,history_len=10):
        self.prev_lrs=deque(maxlen=history_len)
        self.version=version
    def get_prev_lr(self):
        if len(self.prev_lrs) != 0:
            return self.prev_lrs[-1]
        return cfg.OPTIM.BASE_LR
    def get_lrs(self):
        if self.version==1:
            lr=self.get_prev_lr()
            lrs=np.linspace(lr/2,lr*1.5,num=5)
        elif self.version==2:
            lrs=[0.0, 0.001,0.005,0.01,0.02,0.05,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        elif self.version==3:
            lr=self.get_prev_lr()
            lrs=np.linspace(lr/2,lr*2,num=10)
            lrs=[0]+list(lrs)
        elif self.version==4:
            lr=self.get_prev_lr()
            lrs=np.linspace(lr/3,lr*3,num=10)
            lrs=[0]+list(lrs)
        elif self.version==5:
            lr=self.get_prev_lr()
            lrs=np.linspace(lr/10,lr*10,num=20)
            lrs=[0]+list(lrs)
        elif self.version==6:
            mean=np.mean(self.prev_lrs)
            std=np.std(self.prev_lrs)
            lrs=list(np.linspace(mean-std*2,mean+std*2,num=10))
        else:
            raise NotImplementedError()
        lrs=[max(round(lr,4),0) for lr in list(lrs)]
        lrs=sorted(set(lrs))
        return lrs
    def determine_best_lr(self,lr_to_loss):
        best_lr = min(lr_to_loss, key=lr_to_loss.get)
        self.prev_lrs.append(best_lr)
        return best_lr


@torch.no_grad()
def setup_p_grad(optimizer):
    # incorporate weight decay into p.grad
    # also means the actual optimizer.step()
    # doesn't need to use weight decay again
    for group in optimizer.param_groups:
        weight_decay = group["weight_decay"]
        momentum = group['momentum']

        for p in group["params"]:
            state = optimizer.state[p]
            if p.grad is not None:

                # weight decay
                if weight_decay != 0:
                    p.grad.add_(p,alpha=weight_decay)

                if momentum != 0:
                    # update momentum_buffers in the state
                    if 'momentum_buffer' not in state:
                        # initializing momentum buffer
                        state['momentum_buffer'] = torch.clone(p.grad).detach()
                    # updating momentum buffer
                    else:
                        state['momentum_buffer'].mul_(momentum).add_(p.grad, alpha=1)
                    p.grad=torch.clone(state["momentum_buffer"])

@torch.no_grad()
def get_iter_lr_edward(model, loss_fun, image, target, optimizer, cur_epoch):
    lr_to_loss = dict()
    cur_lr = 0
    target_lr = lr_fun_cos(cur_epoch)
    lrs = np.linspace(target_lr/1.5, target_lr*1.5, num=10)
    lrs = np.round(lrs, 4)
    lrs = list(lrs)

    setup_p_grad(optimizer)

    # testing each learning rate
    for lr in lrs:
        # originally:   p <- p + prev_lr * p.grad - cur_lr * p.grad
        # simplify:     p <- p + (prev_lr - cur_lr) * p.grad
        #               p <- p + change_in_lr * p.grad
        change_in_lr = lr - cur_lr
        cur_lr = lr
        for p in model.parameters():
            p.add_(p.grad, alpha=-change_in_lr)

        output = model(image)
        loss = loss_fun(output, target)
        lr_to_loss[lr] = loss.item()

    # reverting the learning rate that is applied
    for p in model.parameters():
        p.add_(p.grad, alpha=cur_lr)

    best_lr = min(lr_to_loss, key=lr_to_loss.get)
    return best_lr, lr_to_loss


@torch.no_grad()
def get_iter_lr(model, loss_fun, image, target, optimizer):
    lr_to_loss={}
    cur_lr=0 # no previous lr, set to 0
    if hasattr(get_iter_lr,"lr_finder"):
        lr_finder=get_iter_lr.lr_finder
    else:
        lr_finder=LR_Finder(version=cfg.OPTIM.VERSION, history_len=10)
    lrs=lr_finder.get_lrs()
    setup_p_grad(optimizer)

    # testing each learning rate
    for lr in lrs:
        # originally:   p <- p + prev_lr * p.grad - cur_lr * p.grad
        # simplify:     p <- p + (prev_lr - cur_lr) * p.grad
        #               p <- p + change_in_lr * p.grad
        change_in_lr = lr - cur_lr
        cur_lr = lr
        for p in model.parameters():
            p.add_(p.grad, alpha=-change_in_lr)

        output = model(image)
        loss = loss_fun(output, target)
        lr_to_loss[lr] = loss.item()

    # reverting the learning rate that is applied
    for p in model.parameters():
        p.add_(p.grad,alpha=cur_lr)

    best_lr=lr_finder.determine_best_lr(lr_to_loss)
    return best_lr, lr_to_loss


class VanillaSGD(torch.optim.Optimizer):
    """Modified SGD only updating lr
    momentum and weight decay is separated into get_iter_lr
    """

    def __init__(self, params, lr=0.1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is not None:
                    p.add_(p.grad,alpha=-lr)

        return loss
