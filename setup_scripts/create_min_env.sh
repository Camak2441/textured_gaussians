conda create -n textured_gaussians python=3.12.7
conda activate textured_gaussians

pip install torch torchvision torchaudio

pip install ninja numpy jaxtyping rich nerfview
pip install pycolmap
pip install gsplat==1.4.0 nerfstudio tyro wandb tensorboard
pip install plyfile imageio pillow tqdm scipy scikit-image open3d trimesh