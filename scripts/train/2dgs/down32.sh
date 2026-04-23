#!/usr/bin/bash

cd ../examples
export CUDA_VISIBLE_DEVICES=${2:-0}
python simple_trainer_textured_gaussians.py mcmc \
    --scene "$1" \
    --init_extent 1 \
    --init_type=random \
    --background_mode "white" \
    --model_type=2dgs \
    --init_num_pts=10000 \
    --strategy.cap-max=10000 \
    --alpha_loss \
    --data_factor 32 \
    --port 6070
