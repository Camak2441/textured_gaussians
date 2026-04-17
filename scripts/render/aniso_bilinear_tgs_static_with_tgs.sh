#!/usr/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd ../examples
export CUDA_VISIBLE_DEVICES=${3:-0}
python simple_trainer_textured_gaussians.py mcmc \
    --scene "$1" \
    --ckpt "../results/aniso_tgs_to0/$1/ckpts/ckpt_2999.pt" \
    --result_dir_suffix "with_tgs" \
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
    --camera_path $( $SCRIPT_DIR/camera_path_args.sh "$2" ) \
    --freeze_geometry=0 \
    --disable_viewer \
    --port 6070
