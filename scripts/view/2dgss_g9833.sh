#!/usr/bin/bash

cd ../examples
export CUDA_VISIBLE_DEVICES=${3:-0}
python simple_trainer_textured_gaussians.py mcmc \
    --scene "$1" \
    --ckpt "../results/2dgss_g9833_oquad1-1000_swc08/$1/ckpts/ckpt_29999.pt" \
    --viewer_only \
    --init_extent 1 \
    --init_type=random \
    --background_mode "white" \
    --model_type=2dgss \
    --sigmoid_factor=c08 \
    --init_num_pts=9833 \
    --strategy.cap-max=9833 \
    --alpha_loss \
    --normal_loss \
    --schedule_scales_lr \
    --schedule_quats_lr \
    --steps_scaler=1 \
    --port 6070
