import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image
from pytimedinput import timedInput
from skimage.restoration import estimate_sigma
from tqdm import tqdm

import blur_noise  # adapted from https://github.com/LeviBorodenko/motionblur/blob/master/motionblur.py
import phone_noise  # adapted from https://gitlab.com/wg1/jpeg-ai/jpeg-ai-anchors/-/tree/main/Denoising/noise_generator

sys.path.append('..')
from utilities import estimate_noise


def load_img_paths(scene_name=None):
    """Load all the image paths from a scene"""
    # check if the scene exists
    if scene_name is None:
        img_paths = glob.glob(os.path.join('./clean', '[!.]*/*'))
    else:
        img_paths = glob.glob(os.path.join('./clean', scene_name, '[!.]*'))

    if len(img_paths) == 0:
        raise ValueError("No images found.")

    return img_paths


def test_noise(img_path, noise_type, params, noise_strength=None):
    if noise_type == 'phone':
        img = np.asarray(Image.open(img_path), dtype=np.float64) / 255.0
        while True:
            a_r, b_r = phone_noise.sample_param_rgb(params[0], params[1], params[2], params[3], params[4], params[5],
                                                    params[6], params[7], params[8])
            noisy_img = phone_noise.add_noise(img, a_r, b_r)
            try:
                noise_estimate = estimate_sigma(noisy_img, channel_axis=-1, average_sigmas=True)
            except:
                noise_estimate = estimate_sigma(noisy_img, multichannel=True, average_sigmas=True)
            if noise_estimate < 0.035:
                category = 'soft'
            elif noise_estimate < 0.08:
                category = 'medium'
            else:
                category = 'hard'
            if noise_strength is not None:
                if noise_strength == category:
                    return a_r, b_r
                else:
                    continue

            accept, timed_out = timedInput(
                f"Estimated noise sigma is {noise_estimate * 255:.4f} ({category}). Is this acceptable? [y] / n\n\t",
                10)
            if timed_out:
                accept = 'y'
            if accept == 'y':
                return a_r, b_r
    elif noise_type == 'additive_gaussian':
        while True:
            strength = np.random.uniform(0.01, 0.1)
            if noise_strength is not None:
                if noise_strength == 'soft' and strength <= 0.03:
                    return strength
                if noise_strength == 'medium' and 0.07 > strength > 0.03:
                    return strength
                if noise_strength == 'hard' and strength > 0.07:
                    return strength
            else:
                accept, timed_out = timedInput(
                    f"Estimated noise sigma is {strength * 255:.4f}. Is this acceptable? [y] / n\n\t", 10)
                if timed_out:
                    accept = 'y'
                if accept == 'y':
                    return strength
    elif noise_type == 'motion_blur':
        raise NotImplementedError
    else:
        raise NotImplementedError


def generate_noisy_imgs(clean_imgs, noise_type, kargs):
    """Take a list of clean img paths, add noise and save them"""
    # create new folder for the noisy images
    noisy_paths = [path.replace("clean", "noisy/" + noise_type) for path in clean_imgs]
    paths = zip(clean_imgs, noisy_paths)

    # Create output directories if they don't exist
    for path in noisy_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    pbar = tqdm(paths, total=len(clean_imgs))

    if noise_type == 'phone':
        a_r, b_r = kargs
        for path in pbar:
            img = np.asarray(Image.open(path[0]), dtype=np.float64) / 255.0
            noisy_img = phone_noise.add_noise(img, a_r, b_r)
            phone_noise.to_image_from_array(noisy_img).save(path[1])
    elif noise_type == 'motion_blur':
        for path in pbar:
            k = blur_noise.Kernel()
            noisy_img = k.add_noise(path[0], keep_image_dim=True)
            noisy_img.save(path[1])
    elif noise_type == 'additive_gaussian':
        stregth = kargs
        for path in pbar:
            img = np.asarray(Image.open(path[0]), dtype=np.float64) / 255.0
            noisy_img = (img + np.random.normal(0, stregth, img.shape)).clip(0, 1)
            Image.fromarray((noisy_img * 255).astype(np.uint8)).save(path[1])
    else:
        raise NotImplementedError

    noise_estimate = estimate_noise(noisy_paths, 0) * 255
    # write noise estimate to file next to images
    with open(os.path.join(os.path.dirname(noisy_paths[0]), 'noise_estimate.txt'), 'w') as f:
        f.write("Estimated noise variance is " + str(noise_estimate) + ".")


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Argparser')
    parser.add_argument('--one_scene',
                        action='store_true',
                        help="Set this option to not use all scenes")
    parser.add_argument("--scene_name",
                        help="Name of scene to use (same name as images directory). Only used if --one_scene is set")
    parser.add_argument("--noise_type",
                        default="phone",
                        type=str,
                        help="Type of noise to add",
                        choices=["phone", "motion_blur", "additive_gaussian", "all"])
    parser.add_argument("--noise_strength",
                        type=str,
                        help="Strength of noise to add",
                        choices=["soft", "medium", "hard"])
    args = parser.parse_args()

    if args.one_scene and args.scene_name is None:
        raise ValueError("Please specify a scene name if you want to use only one scene.")

    img_paths = load_img_paths(args.scene_name)
    if len(img_paths) == 0:
        raise ValueError("No images found.")

    if args.noise_type == 'phone':
        a_r, b_r = test_noise(img_paths[0], args.noise_type, phone_noise.load_param(), args.noise_strength)
        generate_noisy_imgs(img_paths, args.noise_type, [a_r, b_r])
    elif args.noise_type == 'additive_gaussian':
        sigma = test_noise(img_paths[0], args.noise_type, None, args.noise_strength)
        generate_noisy_imgs(img_paths, args.noise_type, [sigma])
    else:
        raise NotImplementedError
