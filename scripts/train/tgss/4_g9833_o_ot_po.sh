#!/usr/bin/bash

cd ../examples
export CUDA_VISIBLE_DEVICES=${2:-0}
python simple_trainer_textured_gaussians.py mcmc \
    --scene "$1" \
    --init_extent 1 \
    --init_type=pretrained \
    --pretrained_path "../results/2dgss_g9833_oquad1-1000_swc08/$1/ckpts/ckpt_29999.pt" \
    --result_dir_suffix "po_pswc08" \
    --background_mode "white" \
    --model_type=tgss \
    --gaussian_factor=c02 \
    --sigmoid_factor=c08 \
    --init_num_pts=9833 \
    --strategy.cap-max=9833 \
    --strategy.refine-start-iter=1000000000000 \
    --filtering=bilinear4_bwd2 \
    --alpha_loss \
    --normal_loss \
    --opac_loss \
    --opac_loss_fn="t01" \
    --opac_loss_start_iter 0 \
    --tex_opac_loss \
    --tex_opac_loss_fn="t03" \
    --tex_opac_loss_start_iter 0 \
    --steps_scaler=1 \
    --textured_rgb \
    --textured_alpha \
    --port 6070
