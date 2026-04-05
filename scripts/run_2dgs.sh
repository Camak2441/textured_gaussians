cd ../examples
export CUDA_VISIBLE_DEVICES=${2:-0}
results_dir="../results/2dgs"
python simple_trainer_textured_gaussians.py mcmc \
    --scene "$1" \
    --init_extent 1 \
    --init_type "random" \
    --background_mode "white" \
    --model_type=2dgs \
    --init_num_pts=10000 \
    --strategy.cap-max=10000 \
    --alpha_loss \
    --port 6070
    
    # --data_dir "../data/nerf_synthetic/ficus" \
    # --result_dir "../results/2dgs/ficus" \
    # --dataset "blender" \