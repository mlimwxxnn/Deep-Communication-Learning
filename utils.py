"""
    A collection of utility functions for the experimental process.
"""

import random
import torch
import os
import numpy as np
import time


def seed_torch(seed=1029):
    """
    Set various random number seeds to allow for repeatable experimental results.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class TimeCount:
    """
    Class used to time the training process.
    """
    def __init__(self):
        self.start = time.time()

    def count(self):
        """
        Return the time since the last timing.
        """
        use_time = round(time.time() - self.start, 2)
        self.start = time.time()
        return use_time


def print_single(log_str, logfile=None, show_in_console=True):
    """
    Print the log to the console or to the input log file.
    """
    if show_in_console:
        print(log_str)
    if logfile is not None:
        file = open(logfile, 'a+')
        print(log_str, file=file)
        file.close()


def weight_communicate_by_abs(model_list, beta=0.5):
    """
    The exchange of model parameters is performed through the idea of regularization,
    i.e., these networks will learn more proportional knowledge from networks with
    smaller absolute values of model parameters.
    """
    epsilon = 1e-18
    device = model_list[0].device
    for w in zip(*[model.net.parameters() for model in model_list]):
        w_sum = 0
        numerator_dic = dict()
        for i in range(len(w)):
            start_ind = i
            end_ind = i + len(w) - 1
            w_sum_tmp = torch.ones(size=w[0].size(), dtype=torch.float32).to(device)
            for ind in range(start_ind, end_ind):
                ind %= len(w)
                w_sum_tmp *= w[ind]
            w_sum_tmp = torch.abs(w_sum_tmp)
            index = (i + len(w) - 1) % len(w)
            numerator_dic[index] = w_sum_tmp
            w_sum += w_sum_tmp
        w_final = 0
        for i in range(len(w)):
            w_final += (numerator_dic[i] / (w_sum + epsilon)) * w[i]
        for i in range(len(w)):
            w[i].data = beta * w_final + (1 - beta) * w[i]


def weight_communicate_by_mean(model_list, beta=0.5):
    """
    The ratio of communication parameters is the same between models, and in the
    communication, each network learns the same ratio of knowledge from the other
    K-1 networks.
    """
    for w in zip(*[model.net.parameters() for model in model_list]):
        w_final = 0
        for i in range(len(w)):
            w_final += w[i]
        w_final /= len(w)
        for i in range(len(w)):
            w[i].data = beta * w_final + (1 - beta) * w[i]


def communicate_weight(model_list, beta=0.5, combine_type='abs'):
    """
    Select the corresponding communication operation function for the network list
    according to the way of network communication.
    """
    if len(model_list) == 0:
        raise ValueError('model list is empty')
    elif combine_type == 'mean':
        weight_communicate_by_mean(model_list, beta)
    elif combine_type == 'abs':
        weight_communicate_by_abs(model_list, beta)
    else:
        raise ValueError('illegal communicate type!')


def get_lr(init_lr, epoch):
    """
    Tool function to perform learning rate decay.
    """
    times = 0
    if 60 < epoch <= 120:
        times = 1
    elif 120 < epoch <= 160:
        times = 2
    elif 160 < epoch:
        times = 3
    return init_lr * (.2 ** times)

