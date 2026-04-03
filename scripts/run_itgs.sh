cd ../examples
export CUDA_VISIBLE_DEVICES=${1:-0}
# Ensure the JIT compiler uses the correct CUDA version
export TORCH_CUDA_ARCH_LIST="12.0"
results_dir="../results/implicit_textured_gaussians_rgba"
python simple_trainer_textured_gaussians.py mcmc \
    --scene "chair" \
    --init_extent 1 \
    --init_type "pretrained" \
    --background_mode "white" \
    --model_type=itgs \
    --texture_model "mix1" \
    --num_texture_samples 2 \
    --init_num_pts=10000 \
    --strategy.cap-max=10000 \
    --strategy.refine-start-iter=1000000000000 \
    --alpha_loss \
    --textured_rgb \
    --textured_alpha \
    --port 6070