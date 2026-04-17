#!/usr/bin/bash

cd ../examples
export CUDA_VISIBLE_DEVICES=${4:-0}
python simple_trainer_textured_gaussians.py mcmc \
    --scene "$1" \
    --init_extent 1 \
    --init_type=pretrained \
    --background_mode "white" \
    --model_type=itgs \
    --texture_model "$2" \
    --num_texture_samples 16 \
    --sample_alpha_threshold 0.05 \
    --texture_grad_method="dev" \
    --world_sample_normalisation="unit_sphere_strict" \
    --base_color_factor="$3" \
    --init_num_pts=10000 \
    --strategy.cap-max=10000 \
    --strategy.refine-start-iter=1000000000000 \
    --alpha_loss \
    --textured_rgb \
    --textured_alpha \
    --port 6070
