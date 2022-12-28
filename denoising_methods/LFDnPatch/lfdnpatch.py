import os.path
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.append('denoising_methods/LFDnPatch')

from denoising_methods.LFDnPatch.utils import utils_image as util


def lfdnpatch(noisy_img):
    """
    Denoise images using LFDnCNN.
    Adapted from https://github.com/cszn/KAIR
    """
    img = np.load(noisy_img).astype(np.float32) / 255.0
    n_channels = img.shape[2]
    nb = 20

    # model_path = os.path.join('denoising_methods', 'DnCNN', 'weights', 'dncnn_color_blind.pth')
    model_path = os.path.join('denoising_methods', 'LFDnPatch', 'weights', 'lfdnpatch_color_blind.pth')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from models.network_dncnn import LFDnPatch as net
    model = net(in_nc=n_channels, out_nc=3, nc=64, nb=nb, act_mode='R')
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    model = model.double()

    img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    img = img.double()
    img = model(img)
    img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return img
