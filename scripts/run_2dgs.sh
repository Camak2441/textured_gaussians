cd ../examples
export CUDA_VISIBLE_DEVICES=${1:-0}
# Ensure the JIT compiler uses the correct CUDA version
export TORCH_CUDA_ARCH_LIST="12.0"
results_dir="../results/2dgs"
python simple_trainer_textured_gaussians.py mcmc \
    --data_dir "../data/nerf_synthetic/chair/" \
    --result_dir "../results/2dgs/chair" \
    --dataset "blender" \
    --init_extent 1 \
    --init_type "random" \
    --background_mode "white" \
    --model_type=2dgs \
    --init_num_pts=10000 \
    --strategy.cap-max=10000 \
    --alpha_loss \
    --port 6070