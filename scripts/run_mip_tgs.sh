cd ../examples
export CUDA_VISIBLE_DEVICES=${2:-0}
# Ensure the JIT compiler uses the correct CUDA version
export TORCH_CUDA_ARCH_LIST="12.0"
python simple_trainer_textured_gaussians.py mcmc \
    --scene "$1" \
    --init_extent 1 \
    --init_type "pretrained" \
    --background_mode "white" \
    --model_type=tgs \
    --init_num_pts=10000 \
    --strategy.cap-max=10000 \
    --strategy.refine-start-iter=1000000000000 \
    --alpha_loss \
    --textured_rgb \
    --textured_alpha \
    --filtering=mipmapped \
    --port 6070