#!/usr/bin/bash

cd ../examples
export CUDA_VISIBLE_DEVICES=${2:-0}
python simple_trainer_textured_gaussians.py mcmc \
    --scene "$1" \
    --ckpt "../results/2dgs_g2000/$1/ckpts/ckpt_29999.pt" \
    --init_extent 1 \
    --init_type=random \
    --background_mode "white" \
    --model_type=2dgs \
    --init_num_pts=2000 \
    --strategy.cap-max=2000 \
    --alpha_loss \
    --dist_loss \
    --normal_loss \
    --port 6070
