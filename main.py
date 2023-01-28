import argparse
import glob
import os
import time
from subprocess import Popen, PIPE

import numpy as np
from PIL import Image
from tqdm import tqdm

import utilities
from denoising_methods.DnCNN.dncnn import dncnn
from denoising_methods.LFDnPatch.lfdnpatch import lfdnpatch
from denoising_methods.Wavelet.wavelet import wavelet_denoising
from metrics_only import compute_metrics

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Argparser')
    parser.add_argument("--denoising_method",
                        required=True,
                        type=str,
                        help="Denoising method to use",
                        choices=["bm3d", "lfbm5d", "wavelets", "dncnn", "lfdnpatch"])
    parser.add_argument("--scene_name",
                        help="Name of scene to use (same name as images directory)",
                        required=True)
    parser.add_argument("--noise_type",
                        default="phone",
                        type=str,
                        help="Select noisy imgs to consider",
                        choices=["phone", "motion_blur", "additive_gaussian"])
    parser.add_argument("--input_dir",
                        type=str,
                        help="Custom noisy images directory. If not set, the default directory is used")
    parser.add_argument("--output_dir",
                        default='',
                        type=str,
                        help="Custom output directory for denoised images.")
    parser.add_argument("--quality_metric",
                        type=str,
                        help="Image Quality Assessment metric to use",
                        choices=["psnr", "ssim", "msssim", "vif", "dists", "all"])
    parser.add_argument("--IQA_full",
                        action="store_true",
                        help="If set, save the Image Quality Assessment metric for all images in the scene")
    parser.add_argument("--sigma_imgs",
                        type=int,
                        help="Choose how many imags to use to estimate the noise level. Use 0 to use all images",
                        default=0)
    parser.add_argument("--grid_limit",
                        help="Limit the dimensions of the image grid. Set to 0 to use all images. Give one or two "
                             "values.",
                        nargs='+',
                        type=int,
                        default=[0, 0])
    parser.add_argument("--patch_size",
                        help="Patch size for LFDnPatch",
                        type=int,
                        default=20)
    parser.add_argument("--patch_stride",
                        help="Patch stride for LFDnPatch",
                        type=int,
                        default=1)
    parser.add_argument("--number_patches",
                        help="Number of patches for LFDnPatch. Note: a different number of patches requires a new "
                             "trained model",
                        type=int,
                        default=6)
    parser.add_argument("--search_space",
                        help="Search space for LFDnPatch",
                        type=int,
                        default=10)

    args = parser.parse_args()

    # Print experiment description
    print(
        "Running denoising method " + args.denoising_method + " on scene " + args.scene_name + " with noise type " + args.noise_type + ".")

    # methods for blur noise are not implemented yet
    if args.noise_type in ["motion_blur"]:
        raise NotImplementedError("Methods for blur noise are not implemented yet")

    # Get noisy img
    if args.scene_name is None:
        raise ValueError("scene_name is not set")
    if args.input_dir is None:
        img_dir = os.path.join("./data/noisy", args.noise_type, args.scene_name)
    else:
        img_dir = args.input_dir
    # check if dir exists
    if not os.path.isdir(img_dir):
        raise ValueError(
            "Directory `%s` does not exist, generate the noisy images using script `./data/noise_generator.py`" % img_dir)
    noisy_imgs = glob.glob(os.path.join(img_dir, "[!.]*[!.txt]"))
    noisy_imgs = sorted(noisy_imgs)

    # Check if images are in the right format
    grid_x, grid_y = utilities.check_imgs(noisy_imgs, args.scene_name)
    full_size = [grid_x[1], grid_y[1]]
    # Set grid limit
    if len(args.grid_limit) == 1:
        grid_x[1] = min(grid_x[1], args.grid_limit[0])
        grid_y[1] = min(grid_y[1], args.grid_limit[0])
    elif len(args.grid_limit) == 2:
        grid_x[1] = min(grid_x[1], args.grid_limit[0])
        grid_y[1] = min(grid_y[1], args.grid_limit[1])
    else:
        raise ValueError("grid_limit must be a list of 1 or 2 values")

    if grid_x[1] < 0 or grid_y[1] < 0:
        raise ValueError("grid_limit must be positive")

    if grid_x[1] == 0:
        grid_x[1] = full_size[0]
    if grid_y[1] == 0:
        grid_y[1] = full_size[1]

    # remove the noisy imgs that fall outside the grid
    temp = []
    for i in range(grid_y[1]):
        for j in range(grid_x[1]):
            temp.append(noisy_imgs[i * full_size[0] + j])
    noisy_imgs = temp

    if args.output_dir == '':
        output_dir = os.path.join("./data/denoised", args.noise_type, args.scene_name, args.denoising_method)
    else:
        output_dir = os.path.join("./data/denoised", args.output_dir)
    # check if directory exists
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Estimate noise level
    sigma_est = utilities.estimate_noise(noisy_imgs, args.sigma_imgs)
    print("Estimated noise level: %f" % sigma_est)

    if args.denoising_method == "bm3d":
        # measure execution time
        start_time = time.time()

        utilities.check_BM3D()

        # run the executable ./denoising_methods/LFBM5D/LFBM3Ddenoisign
        # and show output
        p = Popen(("./denoising_methods/LFBM5D/LFBM3Ddenoising "
                   f"none {args.scene_name} _ {grid_x[1]} {grid_y[1]} {grid_x[0]} {grid_y[0]} 1 1 row {sigma_est * 255} "
                   f"2.7 {img_dir} none {output_dir} none 16 16 8 3 bior 0 32 16 8 3 dct 0 opp 8 none").split(),
                  stdout=PIPE)

        while p.poll() is None:
            line = p.stdout.readline()
            print(line.decode("utf-8"), end='')

        # measure execution time
        end_time = time.time()

    if args.denoising_method == "lfbm5d":
        # measure execution time
        start_time = time.time()

        utilities.check_LFBM5D()

        # run the executable ./denoising_methods/LFBM5D/LFBM5Ddenoisign
        # and show output
        p = Popen(("./denoising_methods/LFBM5D/LFBM5Ddenoising "
                   f"none {args.scene_name} _ {grid_x[1]} {grid_y[1]} {grid_x[0]} {grid_y[0]} 1 1 row "
                   f"{sigma_est * 255} 2.7 {img_dir} none {output_dir} none 8 18 6 16 4 id sadct haar 0 16 18 6 8 4 "
                   "dct sadct haar 0 opp 0. none").split(),
                  stdout=PIPE)

        while p.poll() is None:
            line = p.stdout.readline()
            print(line.decode("utf-8"), end='')

        # measure execution time
        end_time = time.time()

    if args.denoising_method == "wavelets":
        # measure execution time
        start_time = time.time()

        pbar = tqdm(noisy_imgs, total=len(noisy_imgs))
        for img in pbar:
            # run the wavelet denoising
            im_bayes = wavelet_denoising(img)
            # save the denoised image
            im_bayes = Image.fromarray((im_bayes * 255).astype(np.uint8))
            im_bayes.save(os.path.join(output_dir, os.path.basename(img)))

        # measure execution time
        end_time = time.time()

    if args.denoising_method == "dncnn":
        # measure execution time
        start_time = time.time()

        clean_imgs = dncnn(noisy_imgs)
        for i, img in enumerate(clean_imgs):
            # save the denoised image
            img = Image.fromarray((img * 255).astype(np.uint8))
            img.save(os.path.join(output_dir, os.path.basename(noisy_imgs[i])))

        # measure execution time
        end_time = time.time()

    if args.denoising_method == "lfdnpatch":
        if args.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if args.patch_stride <= 0:
            raise ValueError("patch_stride must be positive")
        if args.number_patches <= 0:
            raise ValueError("number_patches must be positive")
        if args.search_space <= 0:
            raise ValueError("search_space must be positive")
        # measure execution time
        start_time = time.time()

        # get the full path for the directory with the noisy imgs
        full_img_dir = os.path.abspath(img_dir)

        # check if inside the noisy images folder there is another folder called "frankenpatches"
        # if not, create it
        if not os.path.isdir(os.path.join(full_img_dir, "frankenpatches")):
            os.makedirs(os.path.join(full_img_dir, "frankenpatches"))

        # run the executable ./denoising_methods/LFDnPatch/LF-PatchMatch/build/PatchMatch
        print("Computing the Frankenpatches")
        p = Popen(
            f"./denoising_methods/LFDnPatch/LF-PatchMatch/build/PatchMatch {full_img_dir} {full_size[0]} {full_size[1]} {args.patch_size} {args.number_patches} {args.patch_stride} {args.search_space}",
            stdout=PIPE, shell=True)

        end_time = time.time()
        with open(os.path.join(output_dir, "execution_time.txt"), "w") as f:
            f.write(f"Patch matching time: {end_time - start_time} seconds\n")

        while p.poll() is None:
            line = p.stdout.readline()
            print(line.decode("utf-8"), end='')

        # modify the noisy image paths to take in the generated frankenpatches
        noisy_frankenpatches = []
        # old name is: ./data/noisy/noise_type/scene_name/scene_name_xx_yy.png
        # new name is: ./data/noisy/noise_type/scene_name/frankenpatches/scene_name_xx_yy.npy
        for img in noisy_imgs:
            new_name = os.path.join(os.path.dirname(img), "frankenpatches", os.path.basename(img))
            new_name = os.path.splitext(new_name)[0] + ".npy"
            noisy_frankenpatches.append(new_name)

        pbar = tqdm(noisy_frankenpatches, total=len(noisy_frankenpatches))
        for img in pbar:
            clean_img = lfdnpatch(img)
            clean_img = Image.fromarray((clean_img * 255).astype(np.uint8))
            output_name = os.path.join(output_dir, os.path.basename(img))
            output_name = os.path.splitext(output_name)[0] + ".png"
            clean_img.save(output_name)

    # write execution time to file
    with open(os.path.join(output_dir, "execution_time.txt"), "a") as f:
        f.write("Execution time: %s seconds" % (end_time - start_time))

    if args.quality_metric is not None:
        if args.quality_metric == 'all':
            metrics = ['psnr', 'ssim', 'msssim', 'vif', 'dists']
        else:
            metrics = [args.quality_metric]

        # compute quality metrics
        for metric in metrics:
            try:
                compute_metrics(args.denoising_method,
                                args.scene_name,
                                args.noise_type,
                                metric,
                                output_dir[16:],
                                args.IQA_full)
            except AssertionError as e:
                print("Error: %s couldn't be computed" % metric)
                print(e)

        print("Results save in the directory with the denoised images")
