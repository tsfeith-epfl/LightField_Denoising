import torch
import numpy as np


def PSNR(orig, rec):
    # orig and rec are tensors
    # with values in the range [0, 1]
    # and shape [1, H, W]
    if len(orig.shape) != 3 or len(rec.shape) != 3:
        raise ValueError('PSNR expects 3D tensors with shape [1, H, W]')
    if orig.shape[0] != 1 or rec.shape[0] != 1:
        raise ValueError('PSNR expects 1-channel images, use `utilities.rgb_to_one_channel` to convert them to 1-channel')
    mse = ((orig - rec) ** 2).mean().item()
    if mse == 0.0:
        return 100
    else:
        return - 10 * np.log10(mse)
