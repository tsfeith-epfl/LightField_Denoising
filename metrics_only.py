import argparse
import glob
import os
from datetime import datetime

import numpy as np
from tqdm import tqdm

import utilities
from quality_assessment import PSNR, SSIM, MSSSIM, VIF, DISTS


def write_results(metrics, metric_name, output_dir, full_results):
    if full_results:
        if os.path.isfile(os.path.join(output_dir, "full_metrics_%s.txt" % metric_name)):
            # if file already exists, erase its content
            with open(os.path.join(output_dir, "full_metrics_%s.txt" % metric_name), "w") as f:
                f.write("")

        with open(os.path.join(output_dir, "full_metrics_%s.txt" % metric_name), "a") as f:
            f.write("----- FULL %s RESULTS FROM %s -----\n\n" % (metric_name, datetime.now()))
            for img_name, metric in metrics.items():
                f.write("%s : %.5f\n" % (img_name, metric))
            f.write("\n")

    metric = np.array(list(metrics.values()))
    with open(os.path.join(output_dir, "metrics.txt"), "a") as f:
        f.write("----- RESULTS FROM %s -----\n" % datetime.now())
        f.write("Average %s : %.5f\n" % (metric_name, metric.mean().item()))
        f.write("Standard deviation %s : %.5f\n" % (metric_name, metric.std().item()))
        f.write("Minimum %s : %.5f\n" % (metric_name, metric.min().item()))
        f.write("Maximum %s : %.5f\n" % (metric_name, metric.max().item()))
        f.write("\n")


def compute_metrics(denoising_method, scene_name, noise_type, iqa_metric, denoised_dir, full_results):
    # Get clean imgs
    img_dir = os.path.join("./data/clean", scene_name)
    # check if dir exists
    if not os.path.isdir(img_dir):
        raise ValueError(
            "Directory `%s` does not exist, generate the clean images using script `./data/clean_generator.py`" % img_dir)
    clean_imgs = glob.glob(os.path.join(img_dir, "[!.]*[!.txt]"))

    # Get denoised imgs
    # If set, use custom directory
    if denoised_dir is not None:
        img_dir = os.path.join("./data/denoised", denoised_dir)
    else:
        img_dir = os.path.join("./data/denoised", noise_type, scene_name, denoising_method)
    # check if dir exists
    if not os.path.isdir(img_dir):
        raise ValueError(
            "Directory `%s` does not exist, generate the denoised images using script `./main_denoising.py`" % img_dir)
    denoised_imgs = glob.glob(os.path.join(img_dir, "[!.]*[!.txt]"))

    # Remove the clean images that don't have a counterpart in the denoised
    base_denoised = set([os.path.basename(img) for img in denoised_imgs])
    clean_imgs = [img for img in clean_imgs if os.path.basename(img) in base_denoised]
    base_clean = ([os.path.basename(img) for img in clean_imgs])

    # Check if all denoised have a clean counterpart
    for img in denoised_imgs:
        if not os.path.basename(img) in base_clean:
            raise ValueError("Image `%s` has no clean counterpart" % img)

    # Sort the images
    clean_imgs.sort()
    denoised_imgs.sort()

    # generate the output directory
    if denoised_dir is None:
        output_dir = os.path.join("./data/denoised", noise_type, scene_name, denoising_method)
    else:
        output_dir = os.path.join("./data/denoised", denoised_dir)

    # Check if there is file `metrics.txt` in the output directory
    # If not, create it
    if not os.path.isfile(os.path.join(output_dir, "metrics.txt")):
        with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
            f.write("")

    print("\nComputing", iqa_metric.upper(), "for denoised images.")
    metrics = {}
    for img_clean, img_denoised in tqdm(zip(clean_imgs, denoised_imgs), total=len(clean_imgs)):
        # Convert images to numpy arrays
        img_clean_arr = utilities.convert_imgs_to_tensor(img_clean)
        img_denoised_arr = utilities.convert_imgs_to_tensor(img_denoised)

        # Compute the chose metric and save it
        # metrics has key: value with clean image name: metric value
        if iqa_metric == "psnr":
            img_clean_arr = utilities.rgb_to_one_channel(img_clean_arr)
            img_denoised_arr = utilities.rgb_to_one_channel(img_denoised_arr)
            metrics[img_clean] = PSNR.PSNR(img_clean_arr, img_denoised_arr)
        elif iqa_metric == "ssim":
            metrics[img_clean] = SSIM.SSIM(img_clean_arr, img_denoised_arr)
        elif iqa_metric == "msssim":
            metrics[img_clean] = MSSSIM.MS_SSIM(img_clean_arr, img_denoised_arr)
        elif iqa_metric == "vif":
            img_clean_arr = utilities.rgb_to_one_channel(img_clean_arr)
            img_denoised_arr = utilities.rgb_to_one_channel(img_denoised_arr)
            metrics[img_clean] = VIF.VIF(img_clean_arr, img_denoised_arr)
        elif iqa_metric == "dists":
            metrics[img_clean] = DISTS.DISTS(img_clean_arr, img_denoised_arr)
        else:
            raise ValueError("Invalid metric")

    # Write results
    write_results(metrics, iqa_metric.upper(), output_dir, full_results)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Argparser')
    parser.add_argument("--denoising_method",
                        default="bm3d",
                        type=str,
                        help="Denoising method used",
                        choices=["bm3d", "lfbm5d", "wavelets", "dncnn", "lfdncnn"])
    parser.add_argument("--scene_name",
                        help="Name of scene used (same name as images directory)",
                        required=True)
    parser.add_argument("--noise_type",
                        default="phone",
                        type=str,
                        help="Noise type used",
                        choices=["phone", "motion_blur", "additive_gaussian"])
    parser.add_argument("--metric",
                        default="psnr",
                        type=str,
                        help="Metric to use",
                        choices=["psnr", "ssim", "msssim", "vif", "dists", "all"])
    parser.add_argument("--denoised_dir",
                        type=str,
                        help="Custom denoised images directory. If not set, the default directory is used")
    parser.add_argument("--IQA_full",
                        action="store_true",
                        help="If set, write the full results to file")
    args = parser.parse_args()

    # methods for blur noise are not implemented yet
    if args.noise_type in ["motion_blur", "all"]:
        raise NotImplementedError("Methods for blur noise are not implemented yet")

    if args.metric == 'all':
        metrics = ['psnr', 'ssim', 'msssim', 'vif', 'dists']
    else:
        metrics = [args.metric]

    for metric in metrics:
        compute_metrics(args.denoising_method,
                        args.scene_name,
                        args.noise_type,
                        metric,
                        args.denoised_dir,
                        args.IQA_full)

    print("Results saved in the directory with the denoised images.")
