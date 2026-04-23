from dataclasses import dataclass, field
from typing import Literal, assert_never

from textured_gaussians.strategy.default import DefaultStrategy
from textured_gaussians.strategy.mcmc import MCMCStrategy


@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Open the viewer and block, without training or evaluation
    viewer_only: bool = False
    # Path to the .pt file. If provide, it will skip training and render a video
    ckpt: str | None = None
    # Path to a nerfview camera path JSON for offline rendering
    camera_path: list[str] | None = None

    scene: str | None = None
    result_dir_suffix: str | None = None

    # Dataset mode
    dataset_type: str = "colmap"

    # Whether to attempt to resume from a previous checkpoint
    resume: bool = False

    # Path to the Mip-NeRF 360 dataset
    data_dir: str | None = None
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str | None = None
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: int | None = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0

    # Port for the viewer server
    port: int = 8080

    schedule_means_lr: bool = True
    schedule_quats_lr: bool = False
    schedule_scales_lr: bool = False

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: list[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: list[int] | None = None
    save_every: int | None = 500
    freeze_geometry: int | None = None
    # Steps to save the model textures
    render_traj_steps: list[int] = field(
        default_factory=lambda: [
            100,
            1_000,
            2_000,
            4_000,
            7_000,
            10_000,
            20_000,
            30_000,
        ]
    )
    render_texture_steps: list[int] = field(
        default_factory=lambda: [
            100,
            1_000,
            2_000,
            4_000,
            7_000,
            10_000,
            20_000,
            30_000,
        ]
    )
    # The step to start from, mostly used for resume
    init_step: int = 0

    # Initialization strategy
    init_type: Literal["sfm", "pretrained", "random"] = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.2
    # Far plane clipping distance
    far_plane: float = 200

    # GSs with opacity below this value will be pruned
    prune_opa: float = 0.05
    # GSs with image plane gradient above this value will be split/duplicated
    grow_grad2d: float = 0.0002
    # GSs with scale below this value will be duplicated. Above will be split
    grow_scale3d: float = 0.01
    # GSs with scale above this value will be pruned.
    prune_scale3d: float = 0.1

    # Start refining GSs after this iteration
    refine_start_iter: int = 500
    # Stop refining GSs after this iteration
    refine_stop_iter: int = 15_000
    # Reset opacities every this steps
    reset_every: int = 3000
    # Refine GSs every this steps
    refine_every: int = 100

    min_opacity: float = 0.005

    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use absolute gradient for pruning. This typically requires larger --grow_grad2d, e.g., 0.0008 or 0.0006
    absgrad: bool = False
    # Normalise rotation grad by the scale of the Gaussian
    norm_rot_grad: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False
    # Whether to use revised opacity heuristic from arXiv:2404.06109 (experimental)
    revised_opacity: bool = False

    # Use random background for training to discourage transparency
    background_mode: str | None = None

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Enable normal consistency loss. (Currently for 2DGS only)
    normal_loss: bool = False
    # Weight for normal loss
    normal_lambda: float = 5e-2
    # Iteration to start normal consistency regulerization
    normal_start_iter: int = 10_000

    # Distortion loss (experimental)
    dist_loss: bool = False
    # Weight for distortion loss
    dist_lambda: float = 1e-2
    # Iteration to start distortion loss regulerization
    dist_start_iter: int = 7_000

    # Alpha loss
    alpha_loss: bool = False
    alpha_lambda: float = 1e-1

    # Scale loss
    scale_loss: bool = False
    scale_lambda: float = 1e-1

    # Opac loss
    opac_loss: bool = False
    opac_lambda: float = 2e-2
    opac_loss_fn: str = "t02"
    opac_loss_start_iter: int = 15_000

    # Texture opacity loss
    tex_opac_loss: bool = False
    tex_opac_lambda: float = 2e-2
    tex_opac_loss_fn: str = "t04"
    tex_opac_loss_start_iter: int = 15_000

    # Steepness loss
    steepness_loss: bool = False
    steepness_loss_lambda: float = 2e-2
    steepness_loss_fn: str = "stpns"
    steepness_loss_start_iter: int = 0

    # Frequency regularization for DCT textures — penalises high-frequency coefficients
    freq_loss: bool = False
    freq_lambda: float = 1e-3

    # Frequency-guided scale pre-training (Steps 1–5 of freq_guidance.py)
    freq_guidance: bool = False
    freq_guidance_lambda: float = 1e-2
    freq_guidance_start_iter: int = 7_000
    freq_guidance_f_target: float = 0.25
    freq_guidance_downsample: int = 8
    freq_guidance_block_size: int = 16
    freq_guidance_use_upsampled: bool = False

    # Frequency-guided orientation pre-training (UV wavelength vector alignment)
    freq_guidance_orient: bool = False
    freq_guidance_orient_lambda: float = 1e-3
    freq_guidance_orient_start_iter: int = 7_000

    # Model for splatting.
    model_type: Literal[
        "2dgs",
        "2dss",
        "2dgss",
        "tgs",
        "dtgs",
        "itgs",
        "tss",
    ] = "2dgs"
    texture_model: str | None = None
    num_texture_samples: int = 10
    sample_alpha_threshold: float = 0.1
    texture_batch_size: int | None = None
    texture_grad_method: Literal["dev", "cpu", "checkpoint"] = "checkpoint"
    texture_input_type: Literal["gaussian", "world", "world_and_view"] = "gaussian"
    world_sample_normalisation: Literal[
        "none", "unit_sphere", "unit_sphere_strict", "bbox"
    ] = "none"
    base_color_factor: str | None = None
    sigmoid_factor: str | None = None

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    # Strategy for GS densification
    strategy: DefaultStrategy | MCMCStrategy = field(default_factory=MCMCStrategy)

    # Pretrained checkpoints
    pretrained_path: str | None = None
    # Checkpoint for resuming training
    checkpoint_path: tuple[str, str] | None = None

    # textured gaussians
    texture_resolution: int = 64
    texture_height: int | None = None
    saved_texture_resolution: int | None = None
    saved_texture_width: int | None = None
    saved_texture_height: int | None = None
    textured_rgb: bool = False
    textured_rgb_clamp: Literal["none", "normalize", "clamp", "sigmoid"] = "clamp"
    textured_alpha: bool = False
    textured_alpha_clamp: Literal["none", "normalize", "clamp", "sigmoid"] = "normalize"

    texture_resize_steps: list[int] | None = None
    texture_resize_values: list[int] | None = None
    texture_resize_heights: list[int] | None = None

    filtering: Literal[
        "bilinear",
        "bilinear_bwd2",
        "bilinear2",
        "bilinear3",
        "bilinear3_bwd2",
        "mipmapped",
        "mipmapped2",
        "anisotropic",
        "anisotropic_bilinear",
    ] = "bilinear"

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        if self.save_steps is not None:
            self.save_steps = [int(i * factor) for i in self.save_steps]
        if self.freeze_geometry != None:
            self.freeze_geometry = int(self.freeze_geometry * factor)
        self.render_traj_steps = [int(i * factor) for i in self.render_traj_steps]
        self.render_texture_steps = [int(i * factor) for i in self.render_texture_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)
        self.opac_loss_start_iter = int(self.opac_loss_start_iter * factor)
        self.tex_opac_loss_start_iter = int(self.tex_opac_loss_start_iter * factor)
        self.steepness_loss_start_iter = int(self.steepness_loss_start_iter * factor)
        self.freq_guidance_start_iter = int(self.freq_guidance_start_iter * factor)
        self.freq_guidance_orient_start_iter = int(
            self.freq_guidance_orient_start_iter * factor
        )
        self.normal_start_iter = int(self.normal_start_iter * factor)
        self.dist_start_iter = int(self.dist_start_iter * factor)
        if self.texture_resize_steps is not None:
            self.texture_resize_steps = [
                int(i * factor) for i in self.texture_resize_steps
            ]

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
            strategy.min_opacity = float(self.min_opacity)
        else:
            assert_never(strategy)
