#!/usr/bin/bash

cd ../examples
export CUDA_VISIBLE_DEVICES=${2:-0}
python simple_trainer_textured_gaussians.py mcmc \
    --scene "$1" \
    --init_extent 1 \
    --init_type=random \
    --background_mode "white" \
    --model_type=2dgss \
    --sigmoid_factor=c08 \
    --init_num_pts=500 \
    --strategy.cap-max=500 \
    --alpha_loss \
    --normal_loss \
    --opac_loss \
    --opac_loss_fn="quad1" \
    --opac_loss_start_iter 1000 \
    --schedule_scales_lr \
    --schedule_quats_lr \
    --steps_scaler=1 \
    --port 6070
