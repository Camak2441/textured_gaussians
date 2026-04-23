#!/usr/bin/bash

cd ../examples
export CUDA_VISIBLE_DEVICES=${2:-0}
python simple_trainer_textured_gaussians.py mcmc \
    --scene "$1" \
    --pretrained_path "../results/2dgs_g100/$1/ckpts/ckpt_29999.pt" \
    --init_extent 1 \
    --init_type=pretrained \
    --background_mode "white" \
    --model_type=tgs \
    --init_num_pts=100 \
    --strategy.cap-max=100 \
    --strategy.refine-start-iter=1000000000000 \
    --alpha_loss \
    --textured_rgb \
    --textured_alpha \
    --filtering=anisotropic_bilinear \
    --freeze_geometry=0 \
    --steps_scaler=0.1 \
    --port 6070
    