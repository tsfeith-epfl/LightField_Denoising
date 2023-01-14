# -*- coding: utf-8 -*-
import glob
import os
import re

import torch

'''
# --------------------------------------------
# Model
# --------------------------------------------
# Kai Zhang (github: https://github.com/cszn)
# 03/Mar/2019
# --------------------------------------------
'''


def find_last_checkpoint(save_dir, net_type='G', pretrained_path=None):
    """
    # ---------------------------------------
    # Kai Zhang (github: https://github.com/cszn)
    # 03/Mar/2019
    # ---------------------------------------
    Args:
        save_dir: model folder
        net_type: 'G' or 'D' or 'optimizerG' or 'optimizerD'
        pretrained_path: pretrained model path. If save_dir does not have any model, load from pretrained_path

    Return:
        init_iter: iteration number
        init_path: model path
    # ---------------------------------------
    """

    file_list = glob.glob(os.path.join(save_dir, '*_{}.pth'.format(net_type)))
    if file_list:
        iter_exist = []
        for file_ in file_list:
            iter_current = re.findall(r"(\d+)_{}.pth".format(net_type), file_)
            iter_exist.append(int(iter_current[0]))
        init_iter = max(iter_exist)
        init_path = os.path.join(save_dir, '{}_{}.pth'.format(init_iter, net_type))
    else:
        init_iter = 0
        init_path = pretrained_path
    return init_iter, init_path


'''
# --------------------------------------------
# print
# --------------------------------------------
'''


# --------------------------------------------
# print model
# --------------------------------------------
def print_model(model):
    msg = describe_model(model)
    print(msg)


# --------------------------------------------
# print params
# --------------------------------------------
def print_params(model):
    msg = describe_params(model)
    print(msg)


'''
# --------------------------------------------
# information
# --------------------------------------------
'''


# --------------------------------------------
# model inforation
# --------------------------------------------
def info_model(model):
    msg = describe_model(model)
    return msg


# --------------------------------------------
# params inforation
# --------------------------------------------
def info_params(model):
    msg = describe_params(model)
    return msg


'''
# --------------------------------------------
# description
# --------------------------------------------
'''


# --------------------------------------------
# model name and total number of parameters
# --------------------------------------------
def describe_model(model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    msg = '\n'
    msg += 'models name: {}'.format(model.__class__.__name__) + '\n'
    msg += 'Params number: {}'.format(sum(map(lambda x: x.numel(), model.parameters()))) + '\n'
    msg += 'Net structure:\n{}'.format(str(model)) + '\n'
    return msg


# --------------------------------------------
# parameters description
# --------------------------------------------
def describe_params(model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    msg = '\n'
    msg += ' | {:^6s} | {:^6s} | {:^6s} | {:^6s} || {:<20s}'.format('mean', 'min', 'max', 'std', 'shape',
                                                                    'param_name') + '\n'
    for name, param in model.state_dict().items():
        if not 'num_batches_tracked' in name:
            v = param.data.clone().float()
            msg += ' | {:>6.3f} | {:>6.3f} | {:>6.3f} | {:>6.3f} | {} || {:s}'.format(v.mean(), v.min(), v.max(),
                                                                                      v.std(), v.shape, name) + '\n'
    return msg
