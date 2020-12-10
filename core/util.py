# coding=UTF-8
from itertools import product, combinations
import torch
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import logging
import os


def get_optimizer(name, parameters, lr, weight_decay=0):
    """initialize parameter optimizer
    """
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(parameters, lr=lr,
                                   weight_decay=weight_decay)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr,
                                   weight_decay=weight_decay)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1, min_ratio=0.0):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(min_ratio, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def change_lr(optimizer, new_lr):
    """change the learing rate in the optimizer
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def heuristic_pooling(x, step, method='mean'):
    def torch_average(x, dim=-2):
        return torch.mean(x, dim)

    def torch_max(x, dim=-2):
        return torch.max(x, dim)[0]

    def torch_min(x, dim=-2):
        return torch.min(x, dim)[0]

    def check_step(x, step):
        if isinstance(step, list):
            return x.size(0) == sum(step)
        else:
            return x.size(0) % step == 0

    assert check_step(x, step)

    if method.lower() == 'max':
        func = torch_max
    elif method.lower() == 'min':
        func = torch_min
    else:
        func = torch_average

    if isinstance(step, list):
        output = []
        step_start = 0
        for s in step:
            step_end = step_start + s
            if s == 0:
                value = torch.zeros(x.size(-1), device=x.device).type_as(x)
            else:
                value = func(x[step_start:step_end])
            output.append(value)
            step_start = step_end
        return torch.stack(output, dim=0)
    else:
        x = x.view(-1, step, x.size(-1))
        output = func(x)
        return output


def predict_confidence(predicts):
    entropy = -torch.mean(predicts * predicts.log(), dim=-1)
    max_entropy = entropy.max()
    confidence = 1 - entropy / max_entropy
    return confidence


def check_tensor(device, *args):
    results = []
    for arg in args:
        if arg is not None:
            if isinstance(arg, tuple) or isinstance(arg, list):
                arg = [p_arg.to(device) for p_arg in arg]
                results.append(arg)
            else:
                results.append(arg.to(device))
        else:
            results.append(arg)
    return results
