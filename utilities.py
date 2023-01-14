import os

import numpy as np
import torch
from PIL import Image
from skimage.restoration import estimate_sigma


def check_BM3D():
    """Check if BM3D is properly set-up"""
    # check if ./denoising_methods/LFBM5D/ contains the executable
    if not os.path.isfile("./denoising_methods/LFBM5D/LFBM3Ddenoising"):
        # check if it is not in the bin directory
        if not os.path.isfile("./denoising_methods/LFBM5D/bin/LFBM3Ddenoising"):
            raise ValueError("LFBM5D is not properly set-up, please follow the instructions in the README.md")
        else:
            # move it to the root directory
            os.rename("./denoising_methods/LFBM5D/bin/LFBM3Ddenoising", "./denoising_methods/LFBM5D/LFBM3Ddenoising")


def check_LFBM5D():
    """Check if LFBM5D is properly set-up"""
    # check if ./denoising_methods/LFBM5D/ contains the executable
    if not os.path.isfile("./denoising_methods/LFBM5D/LFBM5Ddenoising"):
        # check if it is not in the bin directory
        if not os.path.isfile("./denoising_methods/LFBM5D/bin/LFBM5Ddenoising"):
            raise ValueError("LFBM5D is not properly set-up, please follow the instructions in the README.md")
        else:
            # move it to the root directory
            os.rename("./denoising_methods/LFBM5D/bin/LFBM5Ddenoising", "./denoising_methods/LFBM5D/LFBM5Ddenoising")


def check_imgs(imgs, scene_name):
    """Check if images are in the right format"""

    # check if any images exist
    if len(imgs) == 0:
        raise ValueError("No images found, generate the noisy images using script `./data/noise_generator.py`")

    # imgs should be in format `scene_name_x_y.png`
    min_x, max_x, min_y, max_y = float('inf'), -float('inf'), float('inf'), -float('inf')
    for img in imgs:
        img_name = os.path.basename(img)
        img_name = img_name.split("/")[-1]
        img_name = img_name.split("_")

        if len(img_name) < 3:
            raise ValueError("Images should be in format `scene_name_x_y.png` but found `%s`" % img)
        if len(img_name) > 3:
            img_name = ["_".join(img_name[:-2]), img_name[-2], img_name[-1]]
        if img_name[0] != scene_name:
            raise ValueError("Images should be in format `scene_name_x_y.png` but found `%s`" % img)
        if not img_name[1].isdigit():
            raise ValueError("Images should be in format `scene_name_x_y.png` but found `%s`" % img)
        if not img_name[2].split('.')[0].isdigit():
            raise ValueError("Images should be in format `scene_name_x_y.png` but found `%s`" % img)
        if not img_name[2].endswith(".png"):
            raise ValueError("Images should be in format `scene_name_x_y.png` but found `%s`" % img)
        x, y = int(img_name[1]), int(img_name[2].split('.')[0])
        min_x, max_x, min_y, max_y = min(min_x, x), max(max_x, x), min(min_y, y), max(max_y, y)

    return [min_x, max_x - min_x + 1], [min_y, max_y - min_y + 1]


def convert_imgs_to_tensor(img_path):
    """Convert list of path names to imgs, into a 4D pytorch tensor"""
    img = Image.open(img_path)
    # convert from PIL to tensor without transforms
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img)
    return img


def load_image(path):
    """Load image from path"""
    img = Image.open(path)
    img = np.array(img, dtype=np.float32) / 255.0
    return img


def rgb_to_one_channel(img):
    # img is a tensor of shape [3, H, W]
    # return a tensor of shape [1, H, W]
    if len(img.shape) != 3 or img.shape[0] != 3:
        raise ValueError("The input tensor must be of shape [3, H, W]")

    r = img[0, :, :]
    g = img[1, :, :]
    b = img[2, :, :]

    p1, p2, p3 = [0.2126, 0.7152, 0.0722]  # params taken from  ITU-R Rec. BT.709-6

    y_img = p1 * r + p2 * g + p3 * b
    # add a dummy dimension
    y_img = y_img.unsqueeze(0)
    return y_img


def estimate_noise(noisy_imgs, sigma_imgs_used):
    # Estimate noise level
    if sigma_imgs_used < 0:
        raise ValueError("sigma_imgs must be a positive integer")
    elif sigma_imgs_used == 0 or sigma_imgs_used > len(noisy_imgs):
        sigma_est = 0
        for img in noisy_imgs:
            try:
                sigma_est += estimate_sigma(load_image(img), channel_axis=-1, average_sigmas=True)
            except:
                sigma_est += estimate_sigma(load_image(img), multichannel=True, average_sigmas=True)
        sigma_est /= len(noisy_imgs)
    else:
        sigma_est = 0
        for img in noisy_imgs[:sigma_imgs_used]:
            try:
                sigma_est += estimate_sigma(load_image(img), channel_axis=-1, average_sigmas=True)
            except:
                sigma_est += estimate_sigma(load_image(img), multichannel=True, average_sigmas=True)
        sigma_est /= sigma_imgs_used
    return sigma_est
