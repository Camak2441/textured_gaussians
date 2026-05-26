#!/usr/bin/bash

cd ../examples
export CUDA_VISIBLE_DEVICES=${2:-0}
python simple_trainer_textured_gaussians.py mcmc \
    --scene "$1" \
    --ckpt "../results/tgs_g2000_to0_tgs_b2_g2000_abp/$1/ckpts/ckpt_2999.pt" \
    --result_dir_suffix "tgs_b2_g2000_abp" \
    --init_extent 1 \
    --init_type=pretrained \
    --background_mode "white" \
    --model_type=tgs \
    --init_num_pts=2000 \
    --strategy.cap-max=2000 \
    --strategy.refine-start-iter=1000000000000 \
    --alpha_loss \
    --textured_rgb \
    --textured_alpha \
    --filtering=bilinear \
    --freeze_geometry=0 \
    --resume \
    --port 6070