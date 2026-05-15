#!/usr/bin/bash

cd ../examples
export CUDA_VISIBLE_DEVICES=${2:-0}
python simple_trainer_textured_gaussians.py mcmc \
    --scene "$1" \
    --ckpt "../results/tgss4_b2_g1999_ot01-0_ott03-0_sgc02_swc08_po_pswc08/$1/ckpts/ckpt_29999.pt" \
    --init_extent 1 \
    --init_type=random \
    --result_dir_suffix "po_pswc08" \
    --background_mode "white" \
    --model_type=tgss \
    --gaussian_factor=c02 \
    --sigmoid_factor=c08 \
    --init_num_pts=1999 \
    --strategy.cap-max=1999 \
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
    --resume \
    --port 6070
