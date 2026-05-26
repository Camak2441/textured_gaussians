#!/usr/bin/bash

cd ../examples
export CUDA_VISIBLE_DEVICES=${2:-0}
python simple_trainer_textured_gaussians.py mcmc \
    --scene "$1" \
    --ckpt "../results/tgss4_g1999_sgc02_swc08_to0_tgss4_b2_g1999_ot01-0_ott03-0_sgc02_swc08_po_pswc08_abp/$1/ckpts/ckpt_2999.pt" \
    --result_dir_suffix "tgss4_b2_g1999_ot01-0_ott03-0_sgc02_swc08_po_pswc08_abp" \
    --init_extent 1 \
    --init_type=pretrained \
    --background_mode "white" \
    --model_type=tgss \
    --gaussian_factor=c02 \
    --sigmoid_factor=c08 \
    --init_num_pts=1999 \
    --strategy.cap-max=1999 \
    --strategy.refine-start-iter=1000000000000 \
    --alpha_loss \
    --textured_rgb \
    --textured_alpha \
    --filtering=bilinear4 \
    --freeze_geometry=0 \
    --resume \
    --port 6070