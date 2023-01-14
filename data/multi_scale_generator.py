import argparse
import os

from PIL import Image
from tqdm import tqdm

from noise_generator import load_img_paths, generate_noisy_imgs, test_noise, phone_noise


def rescale_imgs(imgs, scale):
    """Take a list of img paths, change their size and save them"""
    print("Generating rescaled images for linear scale factor:", scale)

    # create new folder for the rescaled images
    new_folder = os.path.join(os.path.dirname(imgs[0]) + "_" + f"{int(scale * 1000)}")
    os.makedirs(new_folder, exist_ok=True)

    rescaled_paths = []

    pbar = tqdm(imgs, total=len(imgs))

    for img_path in pbar:
        img = Image.open(img_path)
        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))
        # save new image as scene_name_scale_factor_x_y.png
        img_name = os.path.basename(img_path)
        img_name = img_name.split(".")[0]
        split_name = img_name.split("_")
        split_name.insert(-2, f"{int(scale * 1000)}")
        img_name = "_".join(split_name)
        img_name = img_name + ".png"
        new_path = os.path.join(new_folder, img_name)
        img.save(new_path)
        rescaled_paths.append(os.path.join(new_folder, os.path.basename(img_path)))

    return rescaled_paths


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Argparser')
    parser.add_argument("--n_scales",
                        default=5,
                        type=int,
                        help="Number of scales to generate")
    parser.add_argument("--min_scale",
                        default=0.1,
                        type=float,
                        help="Smallest linear rescale")
    parser.add_argument("--add_noise",
                        action="store_true",
                        help="If set, generate also the corresponding noisy images")
    parser.add_argument("--noise_type",
                        default="phone",
                        type=str,
                        help="Noise type used. Only used if add_noise is set",
                        choices=["phone", "motion_blur", "additive_gaussian"])
    parser.add_argument("--scene_name",
                        help="Name of scene used (same name as images directory)",
                        required=True)

    args = parser.parse_args()

    if args.n_scales < 1:
        raise ValueError("The number of scales needs to be positive.")
    if args.min_scale <= 0 or args.min_scale >= 1:
        raise ValueError("The minimum scale needs to be in (0, 1).")

    # Generate linear scales in range [min_scale, 1)
    if args.n_scales == 1:
        scales = [args.min_scale]
    else:
        scales = [args.min_scale + (1 - args.min_scale) * i / args.n_scales for i in range(args.n_scales + 1)][:-1]

    # Get clean images
    clean_imgs = load_img_paths(args.scene_name)

    # Generate rescaled images
    paths = []
    for scale in scales:
        paths += rescale_imgs(clean_imgs, scale)

    if args.add_noise:
        print("Generating noisy images", end="\n\n")
        params = test_noise(paths[0], args.noise_type, phone_noise.load_param())
        generate_noisy_imgs(paths, args.noise_type, params)
