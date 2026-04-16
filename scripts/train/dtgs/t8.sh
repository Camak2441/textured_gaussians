cd ../examples
export CUDA_VISIBLE_DEVICES=${2:-0}
results_dir="../results/dct_textured_gaussians_rgba"
python simple_trainer_textured_gaussians.py mcmc \
    --scene "$1" \
    --init_extent 1 \
    --init_type=pretrained \
    --background_mode "white" \
    --model_type=dtgs \
    --init_num_pts=10000 \
    --strategy.cap-max=10000 \
    --strategy.refine-start-iter=1000000000000 \
    --alpha_loss \
    --texture_resolution 8 \
    --saved_texture_resolution 64 \
    --textured_rgb \
    --textured_alpha \
    --freeze_geometry=20000 \
    --port 6070