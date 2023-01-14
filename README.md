# LightField Denoising

This repository contains the code for the student research project titled "Light field denoising using conventional and
learning-based methods", under EPFL's MMPG lab. It performs a comparison between the conventional and learning-based
methods for light field denoising. Two sources of noise are considered: Realistic Phone Camera Noise and Realistic
Motion Blur. The results obtained are shown in the report included in this repository.

## Data Set-up Instructions

First, clone the repository. Then, add your own light field scenes to the `./data` folder. Two example scenes are
already provided The clean images should be in the `./data/clean` folder, subvided into folders for each scene. For
example, with two scenes named `scene1` and `scene2` with a grid of dimensions (x1,y1) and (x2,y2) respectively, the
folder structure should be as follows:

```
./data
    ./clean
        ./scene1
            scene1_00_00.png
            scene1_00_01.png
            ...
            scene1_xx_yy.png
        ./scene2
            scene2_00_00.png
            scene2_00_11.png
            ...
            scene2_xx_yy.png
```

To generate the noisy images, do the following steps:

1. `cd data`
2. `python noise_generator.py`

By default, the noise generator will generate noisy images with the Realistic Phone Camera Noise model. For other kinds
of noise (and other parameters), run the `noise_generator.py` script with the appropriate arguments.
There are three noise categories, depending on the category selected, the images will be placed in the folder
`./data/noisy/<noise_category>/<scene_name>`.

## Methods Set-up Instructions

### Wavelet Denoising

No set-up is required for this method. The implemented method is simply a call to SciPy's `denoise_wavelet` function.

### BM3D / LFBM5D

This entire method was adapted from the original LFBM5D implementation [GitHub Repo](https://github.com/V-Sense/LFBM5D).
Go to the method folder via `cd ./denoising_methods/classical/LFBM5D`. Once in that folder, follow the instructions
[here](https://github.com/V-Sense/LFBM5D#source-code-compilation) to generate the necessary executables.

### DnCNN

No set-up is required for this method. The implemented method was taken from the [original DnCNN implementation]
(https://github.com/cszn/KAIR/).

### LFDnPatch

The method relies on an auxiliary repo created for light field patch matching. It is set-up as a submodule in this
repository. To set-up the submodule, run the following commands:

1. `cd denoising_methods/LFDnPatch`
2. `git clone https://github.com/tsfeith-epfl/LF-PatchMatch.git`
3. `cd LF-PatchMatch`
4. `git submodule init`
5. `git submodule update`

Once the submodule is set-up, it needs to be built. To do so, do the following (already inside the `LF-PatchMatch`
folder):

1. `mkdir build`
2. `cd build`
3. `cmake ..`
4. `make`

This will create the executable `PatchMatch` that will be called by the LFDnPatch method.

## Usage Instructions

To run the denoising methods on the noisy images, run the following command:

`python main.py --scene_name <scene_name> --denoising_method <denoising_method>`

There are other possible parameters, but these are the only required ones. To see all the possible parameters, run
`python main.py --help`.

## Replicate Results

To replicate the results shown in the report, 3 bash scripts are provided. They are located in the `./scripts` folder.
However, they are not ready to be run as-is, as the scenes called in the scripts are not all provided. Place your own
scenes in the `./data/clean` folder, and adjust the scripts accordingly. Use the `./data/noise_generator.py` file to
generate the noisy images with different noise strengths and types, and the `./data/multi_scale_generator.py` to create
the downscaled versions of each scene.
