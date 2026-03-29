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
    --texture_model \
        "FourierSelector(\
            num_freqs=15,hidden_dims=[32],\
            sub_models=[\
                \"SIREN(in_dim=3,hidden_dims=[64,48,32],out_dim=4,omega_0=30,hidden_omegas=1.00)\",\
                \"SIREN(in_dim=3,hidden_dims=[64,48,32],out_dim=4,omega_0=45,hidden_omegas=1.25)\",\
                \"SIREN(in_dim=3,hidden_dims=[64,48,32],out_dim=4,omega_0=60,hidden_omegas=1.50)\"\
            ]
        )" \
    --num_texture_samples 3 \
    --init_num_pts=10000 \
    --strategy.cap-max=10000 \
    --strategy.refine-start-iter=1000000000000 \
    --alpha_loss \
    --textured_rgb \
    --textured_alpha \
    --port 6070