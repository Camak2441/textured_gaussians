#!/usr/bin/bash

cd ../examples
export CUDA_VISIBLE_DEVICES=${2:-0}
python simple_trainer_textured_gaussians.py mcmc \
    --scene "$1" \
    --result_dir_suffix "t2to64" \
    --init_extent 1 \
    --init_type=pretrained \
    --background_mode "white" \
    --model_type=tgs \
    --init_num_pts=10000 \
    --strategy.cap-max=10000 \
    --strategy.refine-start-iter=1000000000000 \
    --alpha_loss \
    --textured_rgb \
    --textured_alpha \
    --texture_resolution 2 \
    --saved_texture_resolution 64 \
    --texture_resize_steps 5000 10000 15000 20000 25000 \
    --texture_resize_values 4 8 16 32 64 \
    --port 6070
