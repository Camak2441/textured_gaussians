
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

Clone this repository and install the Textured Gaussians codebase by running:
```bash
git clone https://github.com/Camak2441/textured_gaussians --recursive
cd textured-gaussians
```

There is no need to use --recursive if you already have GLM correctly installed. 


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


## Datasets

- You can download the NeRF synthetic dataset [here](https://drive.google.com/file/d/1OsiBs2udl32-1CqTXCitmov4NQCYdA9g/view?usp=share_link) and the Mip-NeRF 360 dataset [here](https://jonbarron.info/mipnerf360/).
- Custom data loaders that support NeRF synthetic dataset and COLMAP dataset formats are defined in `examples/datasets/`. You can easily extend the code to support your own dataset. 


## Optimization

Textured Gaussians are optimized in two stages: the 2DGS pre-training stage and the Textured Gaussians refinement stage. 

Please refer to the `scripts/` folder for examples on how to run the code. The default trainer Python script is `examples/simple_trainer_textured_gaussians.py` that supports both 2DGS and Textured Gaussians optimization. The script also automatically computes the image quality metrics (PSNR, SSIM, LPIPS, CVVDP) and saves them to a json file.


## License

This codebase is Apache 2.0 licensed. Please refer to the [LICENSE](LICENSE) file for more details.
