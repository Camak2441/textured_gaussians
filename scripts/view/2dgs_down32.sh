#!/usr/bin/bash

cd ../examples
export CUDA_VISIBLE_DEVICES=${2:-0}
python simple_trainer_textured_gaussians.py mcmc \
    --scene "$1" \
    --ckpt "../results/2dgs_df32/$1/ckpts/ckpt_29999.pt" \
    --viewer_only \
    --init_extent 1 \
    --init_type=pretrained \
    --background_mode "white" \
    --model_type=2dgs \
    --init_num_pts=10000 \
    --alpha_loss \
    --textured_rgb \
    --textured_alpha \
    --resume \
    --camera_path "../examples/results/camera_paths/default.json" \
    --data_factor 32 \
    --port 6070
