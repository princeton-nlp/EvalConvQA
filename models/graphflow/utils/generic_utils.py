'''
Created on Nov, 2018

@author: hugo

'''
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None, device=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    sinusoid_table = torch.Tensor(sinusoid_table)
    return sinusoid_table.to(device) if device else sinusoid_table

def get_range_vector(size, device):
    """
    Returns a range vector with the desired size, starting at 0. The CUDA implementation
    is meant to avoid copy data from CPU to GPU.
    """
    if device.type == 'cuda':
        return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
    else:
        return torch.arange(0, size, dtype=torch.long)

def to_cuda(x, device=None):
    if device:
        x = x.to(device)
    return x

def batched_diag(x, device=None):
    # Input: a 2D tensor
    # Output: a 3D tensor
    x_diag = torch.zeros(x.size(0), x.size(1), x.size(1))
    _ = x_diag.as_strided(x.size(), [x_diag.stride(0), x_diag.size(2) + 1]).copy_(x)
    return to_cuda(x_diag, device)

def create_mask(x, N, device=None):
    x = x.data
    mask = np.zeros((x.size(0), N))
    for i in range(x.size(0)):
        mask[i, :x[i]] = 1
    return to_cuda(torch.Tensor(mask), device)

def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting)
    return config
