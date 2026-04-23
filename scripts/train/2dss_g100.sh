#!/usr/bin/bash

cd ../examples
export CUDA_VISIBLE_DEVICES=${2:-0}
python simple_trainer_textured_gaussians.py mcmc \
    --scene "$1" \
    --init_extent 1 \
    --init_type=random \
    --background_mode "white" \
    --model_type=2dss \
    --init_num_pts=100 \
    --strategy.cap-max=100 \
    --alpha_loss \
    --normal_loss \
    --steepness_loss \
    --steps_scaler=1 \
    --schedule_scales_lr \
    --port 6070
