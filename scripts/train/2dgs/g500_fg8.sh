#!/usr/bin/bash

cd ../examples
export CUDA_VISIBLE_DEVICES=${2:-0}
python simple_trainer_textured_gaussians.py mcmc \
    --scene "$1" \
    --init_extent 1 \
    --init_type=random \
    --background_mode "white" \
    --model_type=2dgs \
    --init_num_pts=500 \
    --strategy.cap-max=500 \
    --alpha_loss \
    --freq_guidance \
    --freq_guidance_steps 15000 \
    --freq_guidance_orient \
    --freq_guidance_orient_steps 15000 \
    --normal_loss \
    --dist_loss \
    --steps_scaler=0.2 \
    --texture_resolution 64 \
    --port 6070
