#!/usr/bin/bash

cd ../examples
export CUDA_VISIBLE_DEVICES=${2:-0}
python simple_trainer_textured_gaussians.py mcmc \
    --scene "$1" \
    --pretrained_path "../results/2dgs_g1000/$1/ckpts/ckpt_29999.pt" \
    --init_extent 1 \
    --init_type=pretrained \
    --background_mode "white" \
    --model_type=tgs \
    --init_num_pts=1000 \
    --strategy.cap-max=1000 \
    --strategy.refine-start-iter=1000000000000 \
    --alpha_loss \
    --textured_rgb \
    --textured_alpha \
    --port 6070
