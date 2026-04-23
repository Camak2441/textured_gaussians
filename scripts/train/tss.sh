#!/usr/bin/bash

cd ../examples
export CUDA_VISIBLE_DEVICES=${2:-0}
python simple_trainer_textured_gaussians.py mcmc \
    --scene "$1" \
    --init_extent 1 \
    --init_type=pretrained \
    --pretrained_path "../results/2dss/$1/ckpts/ckpt_6999.pt" \
    --background_mode "white" \
    --model_type=tss \
    --init_num_pts=10000 \
    --strategy.cap-max=10000 \
    --strategy.refine-start-iter=1000000000000 \
    --filtering=bilinear_bwd2 \
    --alpha_loss \
    --dist_loss \
    --normal_loss \
    --textured_rgb \
    --textured_alpha \
    --steps_scaler=1 \
    --port 6070
