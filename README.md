
## Textured Gaussian-Sigmoid Splatting

This project extends [Textured Gaussian Splatting](https://textured-gaussians.github.io/), which itself extends [gsplat](https://github.com/nerfstudio-project/gsplat) to explore the following changes to the pipelines:

- Using a novel sigmoid kernel
- Using mipmapping and anisotropic filtering
- Using the discrete cosine transform to store the texture

Additionally, this project also evaluates the pipeline using Color Video VDP. 


## Setup

The code was tested on:
- **OS**: Ubuntu 22.04.5 LTS
- **GPU**: NVIDIA RTX 5080 and NVIDIA RTX V100
- **Driver Version**: 560.35.03 
- **CUDA Version**: 12.8
- **nvcc**: 12.8
- **Python Version**: 3.12.7
- **Torch Version**: 2.9.1

If you are running this project on a fresh Ubuntu 22.04 VM, then you can run `setup_scripts/setup_blank_ubuntu_vm.sh` to complete much of the installation process of installing CUDA and miniconda. 

Clone this repository and install the Textured Gaussians codebase by running:
```bash
git clone --recursive https://github.com/Camak2441/textured_gaussians
cd textured-gaussians
```

There is no need to use --recursive if you already have GLM correctly installed. 
Alternatively, if you already have the codebase, and need to pull in GLM, run
```bash
git submodule update --init
```

To create a conda environment with the same dependencies as the codebase, please run:
```bash
conda env create -f environment.yml
```

If the above command does not work, please create a new conda environment and install the dependencies manually (especially torch and torchvision).
```bash
conda create -n textured_gaussians python=3.12.7
conda activate textured_gaussians
```

```bash
python -m pip install -e . # install in editable mode for development
```

You also need to create a `cfg.yml` in the root directory. It contains:
- data_dir: The path to your datasets
- results_dir: The path to your results folder
- cuda_path: The path to where your cuda install is located
- max_jobs: The default MAX_JOBS you want to use (optional)


## Datasets

- You can download the NeRF synthetic dataset [here](https://drive.google.com/file/d/1OsiBs2udl32-1CqTXCitmov4NQCYdA9g/view?usp=share_link) and the Mip-NeRF 360 dataset [here](https://jonbarron.info/mipnerf360/).
- Custom data loaders that support NeRF synthetic dataset and COLMAP dataset formats are defined in `examples/datasets/`. You can easily extend the code to support your own dataset. 


## Optimization

Textured Gaussians are optimized in two stages: the 2DGS pre-training stage and the Textured Gaussians refinement stage. 

For example, to run the 2dgs model on the scene chair, run in `examples/`
```bash
python simple_trainer_textured_gaussians.py mcmc \
    --scene chair \
    --init_extent 1 \
    --init_type=random \
    --background_mode "white" \
    --model_type=2dgs \
    --init_num_pts=10000 \
    --strategy.cap-max=10000 \
    --alpha_loss \
    --dist_loss \
    --normal_loss \
    --steps_scaler=1 \
    --port 6070
```

Then to train the textures, run
```bash
python simple_trainer_textured_gaussians.py mcmc \
    --scene chair \
    --init_extent 1 \
    --init_type=pretrained \
    --background_mode "white" \
    --model_type=tgs \
    --init_num_pts=10000 \
    --strategy.cap-max=10000 \
    --strategy.refine-start-iter=1000000000000 \
    --alpha_loss \
    --dist_loss \
    --normal_loss \
    --textured_rgb \
    --textured_alpha \
    --resume \
    --port 6070

```

Please refer to the `scripts/` folder for further examples on how to run the code. The trainer Python script is `examples/simple_trainer_textured_gaussians.py` that supports both 2DGS and Textured Gaussians optimization. The script also automatically computes the image quality metrics (PSNR, SSIM, LPIPS, CVVDP) and saves them to a json file.


## License

This codebase is Apache 2.0 licensed. Please refer to the [LICENSE](LICENSE) file for more details.
