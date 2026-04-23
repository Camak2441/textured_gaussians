#!/usr/bin/bash

cd ../examples
export CUDA_VISIBLE_DEVICES=${2:-0}
python simple_trainer_textured_gaussians.py mcmc \
    --scene "$1" \
    --pretrained_path "../results/2dgs_g1000_ot02-10000_fg8-15000-fgu_fgo15000/$1/ckpts/ckpt_29999.pt" \
    --result_dir_suffix "po_pfg8-fgu" \
    --init_extent 1 \
    --init_type=pretrained \
    --background_mode "white" \
    --model_type=tgs \
    --init_num_pts=1000 \
    --strategy.cap-max=1000 \
    --strategy.refine-start-iter=1000000000000 \
    --filtering=bilinear3_bwd2 \
    --alpha_loss \
    --dist_loss \
    --normal_loss \
    --textured_rgb \
    --textured_alpha \
    --opac_loss \
    --opac_loss_fn="t02" \
    --opac_loss_start_iter 7000 \
    --tex_opac_loss \
    --tex_opac_loss_fn="t02" \
    --tex_opac_loss_start_iter 7000 \
    --port 6070
