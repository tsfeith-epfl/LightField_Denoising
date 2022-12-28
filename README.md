# LightField Denoising

This repository contains the code for the student research project titled "Light field denoising using conventional and 
learning-based  methods", under EPFL's MMPG lab. It performs a comparison between the conventional and learning-based
methods for light field denoising. Two sources of noise are considered: Realistic Phone Camera Noise and Realistic 
Motion Blur. The results obtained are shown in the report included in this repository.

## Data Set-up Instructions

First, clone the repository. Then, add your own light field scenes to the `./data` folder. The clean images should be
in the `./data/clean` folder, subvided into folders for each scene. For example, with two scenes named `scene1` and
`scene2` with a grid of dimensions (x1,y1) and (x2,y2) respectively, the folder structure should be as follows:

```
./data
    ./clean
        ./scene1
            scene1_0_0.png
            scene1_0_1.png
            ...
            scene1_x1_y1.png
        ./scene2
            scene2_0_0.png
            scene2_0_1.png
            ...
            scene2_x2_y2.png
```

To generate the noisy images, do the following steps:
1. `cd ./data`
2. `python noise_generator.py`

There are three noise categories, depending on the category selected, the images will be placed in the folder
`./data/noisy/<noise_category>/<scene_name>`.

## Methods Set-up Instructions

### BM3D / LFBM5D

This entire method was adapted from the original LFBM5D implementation [GitHub Repo](https://github.com/V-Sense/LFBM5D).
Go to the method folder via `cd ./denoising_methods/classical/LFBM5D`. Once in that folder, follow the instructions 
[here](https://github.com/V-Sense/LFBM5D#source-code-compilation) to generate the necessary executables.
