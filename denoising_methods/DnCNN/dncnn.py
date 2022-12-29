import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.append('denoising_methods/DnCNN')


def dncnn(noisy_imgs):
    """
    Denoise images using DnCNN.
    Adapted from https://github.com/cszn/KAIR
    """
    n_channels = 3
    nb = 20

    model_path = os.path.join('denoising_methods', 'DnCNN', 'weights', 'dncnn_color_blind.pth')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from model.network_dncnn import DnCNN as net
    model = net(in_nc=n_channels, out_nc=n_channels, nc=64, nb=nb, act_mode='R')
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    model = model.double()

    pbar = tqdm(noisy_imgs, total=len(noisy_imgs))

    noisy_imgs = np.array([np.array(Image.open(img), dtype=float) / 255.0 for img in noisy_imgs])

    for img in pbar:
        img = np.array(Image.open(img), dtype=float) / 255.0

        img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        img = model(img)
        img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        yield img
