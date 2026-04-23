#!/usr/bin/bash

cd ../examples
export CUDA_VISIBLE_DEVICES=${2:-0}
python simple_trainer_textured_gaussians.py mcmc \
    --scene "$1" \
    --init_extent 1 \
    --init_type=random \
    --background_mode "white" \
    --model_type=2dss \
    --init_num_pts=200 \
    --strategy.cap-max=200 \
    --alpha_loss \
    --dist_loss \
    --normal_loss \
    --steps_scaler=1 \
    --port 6070
