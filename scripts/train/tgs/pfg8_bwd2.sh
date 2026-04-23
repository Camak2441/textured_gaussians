#!/usr/bin/bash

cd ../examples
export CUDA_VISIBLE_DEVICES=${2:-0}
python simple_trainer_textured_gaussians.py mcmc \
    --scene "$1" \
    --pretrained_path "../results/2dgs_fg8-2000_fgo2000/$1/ckpts/ckpt_5999.pt" \
    --result_dir_suffix "pfg8-2000" \
    --init_extent 1 \
    --init_type=pretrained \
    --background_mode "white" \
    --model_type=tgs \
    --init_num_pts=10000 \
    --strategy.cap-max=10000 \
    --strategy.refine-start-iter=1000000000000 \
    --filtering=bilinear_bwd2 \
    --alpha_loss \
    --textured_rgb \
    --textured_alpha \
    --texture_resolution 64 \
    --steps_scaler=0.2 \
    --port 6070
