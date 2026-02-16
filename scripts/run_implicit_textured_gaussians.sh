cd ../examples
export CUDA_VISIBLE_DEVICES=${1:-0}
# Ensure the JIT compiler uses the correct CUDA version
export TORCH_CUDA_ARCH_LIST="12.0"
results_dir="../results/implicit_textured_gaussians_rgba"
python simple_trainer_textured_gaussians.py mcmc \
    --data_dir "../data/nerf_synthetic/chair/" \
    --pretrained_path "../results/2dgs/chair/ckpts/ckpt_29999.pt" \
    --result_dir "${results_dir}/chair" \
    --dataset "blender" \
    --init_extent 1 \
    --init_type "pretrained" \
    --background_mode "white" \
    --model_type=implicit_textured_gaussians \
    --texture_model "MLP(in_dim=3,hidden_dims=[64,64,64,64],out_dim=4)" \
    --init_num_pts=10000 \
    --strategy.cap-max=10000 \
    --strategy.refine-start-iter=1000000000000 \
    --alpha_loss \
    --texture_resolution 64 \
    --textured_rgb \
    --textured_alpha \
    --port 6070