#!/usr/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd ../examples
export CUDA_VISIBLE_DEVICES=${3:-0}
results_dir="../results/textured_gaussians_rgba"
python simple_trainer_textured_gaussians.py mcmc \
    --scene "$1" \
    --ckpt "../results/2dgs/$1/ckpts/ckpt_29999.pt" \
    --init_extent 1 \
    --init_type=pretrained \
    --background_mode "white" \
    --model_type=2dgs \
    --init_num_pts=10000 \
    --alpha_loss \
    --textured_rgb \
    --textured_alpha \
    --camera_path $CAMERA_PATHS_ARG \
    --disable_viewer \
    --port 6070
