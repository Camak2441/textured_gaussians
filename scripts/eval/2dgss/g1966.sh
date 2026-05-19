#!/usr/bin/bash

cd ../examples
export CUDA_VISIBLE_DEVICES=${2:-0}
python simple_trainer_textured_gaussians.py mcmc \
    --scene "$1" \
    --ckpt "../results/2dgss_g1966_oquad1-1000_swc08/$1/ckpts/ckpt_29999.pt" \
    --init_extent 1 \
    --init_type=random \
    --background_mode "white" \
    --model_type=2dgss \
    --sigmoid_factor=c08 \
    --init_num_pts=1966 \
    --strategy.cap-max=1966 \
    --opac_loss \
    --opac_loss_fn="quad1" \
    --opac_loss_start_iter 1000 \
    --alpha_loss \
    --dist_loss \
    --normal_loss \
    --port 6070
