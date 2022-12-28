#!/bin/sh
# conda activate LFields

for SCALE in "_100" "_200" "_300" "_400" "_500" "_600" "_799" "_900" ""
do
    for METHOD in wavelets bm3d lfbm5d dncnn lfdnpatch
    do
        python main.py \
          --scene_name "flower${SCALE}" \
          --denoising_method $METHOD \
          --noise_type phone \
          --quality_metric all \
          --grid_limit 6 \
          --output_dir "scale_flower${SCALE}_${METHOD}"
    done
done
