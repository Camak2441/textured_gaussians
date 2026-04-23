#!/usr/bin/bash

cd ../examples
export CUDA_VISIBLE_DEVICES=${2:-0}
python simple_trainer_textured_gaussians.py mcmc \
    --scene "$1" \
    --init_extent 1 \
    --init_type=random \
    --background_mode "white" \
    --model_type=2dgs \
    --init_num_pts=100 \
    --strategy.cap-max=100 \
    --alpha_loss \
    --dist_loss \
    --normal_loss \
    --opac_loss \
    --opac_loss_fn="t02" \
    --opac_loss_start_iter 10000 \
    --port 6070
