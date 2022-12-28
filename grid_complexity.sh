#!/bin/sh
# conda activate LFields

for GRID in {1..17}
do
    for METHOD in wavelets bm3d lfbm5d dncnn lfdnpatch
    do
        python main.py \
          --scene_name "flower_300" \
          --denoising_method $METHOD \
          --noise_type phone \
          --quality_metric all \
          --grid_limit $GRID \
          --output_dir "grid_flower_${GRID}_${METHOD}"
    done
done
