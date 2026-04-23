#!/usr/bin/bash

cd ../examples
export CUDA_VISIBLE_DEVICES=${2:-0}
python simple_trainer_textured_gaussians.py mcmc \
    --scene "$1" \
    --ckpt "../results/2dss_g100/$1/ckpts/ckpt_29999.pt" \
    --viewer_only \
    --init_extent 1 \
    --init_type=random \
    --background_mode "white" \
    --model_type=2dss \
    --init_num_pts=100 \
    --alpha_loss \
    --textured_rgb \
    --textured_alpha \
    --camera_path "../examples/results/camera_paths/default.json" \
    --port 6070
