#!/usr/bin/bash

cd ../examples
export CUDA_VISIBLE_DEVICES=${2:-0}
python simple_trainer_textured_gaussians.py mcmc \
    --scene "$1" \
    --ckpt "../results/aniso_bilinear_tgs_to0_tgs_b2_psfm_abp/$1/ckpts/ckpt_2999.pt" \
    --result_dir_suffix "tgs_b2_psfm_abp" \
    --init_extent 1 \
    --init_type=pretrained \
    --background_mode "white" \
    --model_type=tgs \
    --init_num_pts=10000 \
    --strategy.cap-max=10000 \
    --strategy.refine-start-iter=1000000000000 \
    --alpha_loss \
    --textured_rgb \
    --textured_alpha \
    --filtering=anisotropic_bilinear \
    --freeze_geometry=0 \
    --resume \
    --port 6070