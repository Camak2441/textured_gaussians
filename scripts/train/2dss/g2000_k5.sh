#!/usr/bin/bash

cd ../examples
export CUDA_VISIBLE_DEVICES=${2:-0}
python simple_trainer_textured_gaussians.py mcmc \
    --scene "$1" \
    --init_extent 1 \
    --init_type=random \
    --background_mode "white" \
    --model_type=2dss \
    --result_dir_suffix "k5" \
    --init_num_pts=2000 \
    --strategy.cap-max=2000 \
    --alpha_loss \
    --dist_loss \
    --normal_loss \
    --freeze_steepnesses 0 \
    --init_steepnesses=3.98151 \
    --steps_scaler=1 \
    --port 6070
