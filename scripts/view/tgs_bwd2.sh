#!/usr/bin/bash

cd ../examples
export CUDA_VISIBLE_DEVICES=${2:-0}
python simple_trainer_textured_gaussians.py mcmc \
    --scene "$1" \
    --ckpt "../results/tgs_b2/$1/ckpts/ckpt_9999.pt" \
    --viewer_only \
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
    --filtering=bilinear_bwd2 \
    --resume \
    --camera_path "../examples/results/camera_paths/default.json" \
    --port 6070
