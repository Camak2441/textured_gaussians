#!/usr/bin/bash

cd ../examples
export CUDA_VISIBLE_DEVICES=${2:-0}
python simple_trainer_textured_gaussians.py mcmc \
    --scene "$1" \
    --init_extent 1 \
    --init_type=random \
    --background_mode "white" \
    --model_type=2dgs \
    --init_num_pts=1000 \
    --strategy.cap-max=1000 \
    --alpha_loss \
    --dist_loss \
    --normal_loss \
    --freq_guidance \
    --freq_guidance_start_iter 15000 \
    --freq_guidance_use_upsampled \
    --freq_guidance_orient \
    --freq_guidance_orient_start_iter 15000 \
    --opac_loss \
    --opac_loss_fn="t02" \
    --opac_loss_start_iter 10000 \
    --steps_scaler=1 \
    --texture_resolution 64 \
    --resume \
    --port 6070
