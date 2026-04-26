#!/usr/bin/bash

cd ../examples
export CUDA_VISIBLE_DEVICES=${3:-0}
python simple_trainer_textured_gaussians.py mcmc \
    --scene "$1" \
    --pretrained_path "../results/$2/$1/ckpts/ckpt_29999.pt" \
    --result_dir_suffix "${2}_abp" \
    --init_extent 1 \
    --init_type=pretrained \
    --background_mode "white" \
    --model_type=tgss \
    --gaussian_factor=c02 \
    --sigmoid_factor=c08 \
    --init_num_pts=10000 \
    --strategy.cap-max=10000 \
    --strategy.refine-start-iter=1000000000000 \
    --alpha_loss \
    --textured_rgb \
    --textured_alpha \
    --filtering=anisotropic_bilinear2 \
    --freeze_geometry=0 \
    --steps_scaler=0.1 \
    --port 6070
    