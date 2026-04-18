import io
import json
import math
import os
import time
import zipfile
from typing_extensions import assert_never

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import yaml
import viser
from datasets.colmap import Dataset, Parser, BlenderDataset
from datasets.traj import generate_interpolated_path
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from examples.config import Config
from examples.scene_args_loader import process_config
from texture_models import canonical_model_name, load_factor, load_model
from textured_gaussians.utils import Filtering, TextureGrads
from examples.coordinate_normalization import (
    compute_camera_unit_sphere_normalization,
    compute_scene_bbox_from_cameras,
    compute_bbox_normalization,
)
from utils import (
    AppearanceOptModule,
    CameraOptModule,
    apply_depth_colormap,
    colormap,
    knn,
    rgb_to_sh,
    set_random_seed,
    remove_from_kwargs,
)

from textured_gaussians.rendering import (
    rasterization_2dgs,
    rasterization_textured_gaussians,
    rasterization_dct_textured_gaussians,
    rasterization_implicit_textured_gaussians,
)
from textured_gaussians.strategy import DefaultStrategy, MCMCStrategy
from textured_gaussians.cuda._wrapper import (
    rasterize_dct_textures,
)


def create_splats_with_optimizers(
    parser: Parser,
    cfg: Config,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    batch_size: int = 1,
    feature_dim: int | None = None,
    device: str = "cuda",
) -> tuple[
    torch.nn.ParameterDict, dict[str, torch.optim.Optimizer], torch.nn.Module | None
]:
    match init_type:
        case "sfm":
            points = torch.from_numpy(parser.points).float()
            rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
            if init_num_pts < points.shape[0]:
                sampled_pts_idx = np.random.choice(
                    points.shape[0], init_num_pts, replace=False
                )
            else:
                sampled_pts_idx = np.arange(points.shape[0])
            # randomly sample points from the SfM points
            points = points[sampled_pts_idx]
            rgbs = rgbs[sampled_pts_idx]
        case "pretrained":
            assert cfg.pretrained_path is not None
            ckpt = torch.load(cfg.pretrained_path)["splats"]
            if init_num_pts < ckpt["means"].shape[0]:
                sampled_pts_idx = np.random.choice(
                    ckpt["means"].shape[0], init_num_pts, replace=False
                )
            else:
                sampled_pts_idx = np.arange(ckpt["means"].shape[0])
            points = ckpt["means"][sampled_pts_idx]
            rgbs = torch.rand((points.shape[0], 3))
        case "random":
            points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
            rgbs = torch.rand((init_num_pts, 3))
        case _:
            raise ValueError(
                "Please specify a correct init_type: sfm, pretrained, or random"
            )

    if init_type == "pretrained":
        scales = ckpt["scales"][sampled_pts_idx]
        quats = ckpt["quats"][sampled_pts_idx]
        opacities = ckpt["opacities"][sampled_pts_idx]
    else:
        N = points.shape[0]
        # Initialize the GS size to be the average dist of the 3 nearest neighbors
        dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
        dist_avg = torch.sqrt(dist2_avg)
        scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]
        quats = torch.rand((N, 4))  # [N, 4]
        opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
    ]

    # SH coefficients
    if feature_dim is None:
        # color is SH coefficients.
        if init_type == "pretrained":
            params.append(
                ("sh0", torch.nn.Parameter(ckpt["sh0"][sampled_pts_idx]), 2.5e-3)
            )
            params.append(
                ("shN", torch.nn.Parameter(ckpt["shN"][sampled_pts_idx]), 2.5e-3 / 20)
            )
        else:
            colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
            colors[:, 0, :] = rgb_to_sh(rgbs)
            params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
            params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

    texture_model = None
    match cfg.model_type:
        case "tgs":
            textures = torch.ones(
                points.shape[0], cfg.texture_height, cfg.texture_width, 4
            )
            textures[:, :, :, :3] = 0.1  # init color to low value
            textures[:, :, :, 3] = 1.0  # init alpha to 1.0
            params.append(("textures", torch.nn.Parameter(textures), 2.5e-3))
        case "dtgs":
            textures = torch.ones(
                points.shape[0], cfg.texture_height, cfg.texture_width, 4
            )
            textures[:, :, :, :] = 0.1  # init to having no frequencies
            textures[:, 0, 0, 3] = (
                cfg.texture_height * cfg.texture_width
            )  # init alpha to flat opaque
            params.append(("textures", torch.nn.Parameter(textures), 1.5e-3))
        case "itgs":
            texture_model_name = canonical_model_name(cfg.texture_model)
            print(f"Loading model {texture_model_name}")
            texture_model = load_model(texture_model_name)
            texture_model.to(device)

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    optimizers = {
        name: (torch.optim.SparseAdam if sparse_grad else torch.optim.Adam)(
            [{"params": splats[name], "lr": lr * math.sqrt(batch_size)}],
            eps=1e-15 / math.sqrt(batch_size),
            betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers, texture_model


class Runner:
    """Engine for training and testing."""

    def __init__(self, cfg: Config) -> None:
        set_random_seed(42)

        self.cfg = cfg
        self.device = "cuda"
        self.step = 0  # current optimization step

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        if cfg.dataset_type == "colmap":
            self.parser = Parser(
                data_dir=cfg.data_dir,
                factor=cfg.data_factor,
                normalize=True,
                test_every=cfg.test_every,
            )
            self.trainset = Dataset(
                self.parser,
                split="train",
                patch_size=cfg.patch_size,
                load_depths=cfg.depth_loss,
            )
            self.valset = Dataset(self.parser, split="val")
            self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        elif cfg.dataset_type == "blender":
            self.parser = None
            if cfg.background_mode == "white":
                bg_color = (255, 255, 255)
            else:
                bg_color = (0, 0, 0)
            self.trainset = BlenderDataset(
                data_dir=cfg.data_dir,
                split="train",
                bg_color=bg_color,
                factor=cfg.data_factor,
            )
            self.valset = BlenderDataset(
                data_dir=cfg.data_dir,
                split="val",
                bg_color=bg_color,
                factor=cfg.data_factor,
            )
            self.scene_scale = 1.0  # no scaling required
        else:
            raise ValueError(f"Dataset mode {cfg.dataset_type} not supported!")

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers, self.texture_model = (
            create_splats_with_optimizers(
                self.parser,
                self.cfg,
                init_type=cfg.init_type,
                init_num_pts=cfg.init_num_pts,
                init_extent=cfg.init_extent,
                init_opacity=cfg.init_opa,
                init_scale=cfg.init_scale,
                scene_scale=self.scene_scale,
                sh_degree=cfg.sh_degree,
                sparse_grad=cfg.sparse_grad,
                batch_size=cfg.batch_size,
                feature_dim=feature_dim,
                device=self.device,
            )
        )
        self.base_color_factor = None
        print("Model initialized. Number of GS:", len(self.splats["means"]))
        self.model_type = cfg.model_type
        self.coord_center, self.coord_scale = self._compute_coord_normalisation(cfg)

        if self.model_type in [
            "2dgs",
            "tgs",
            "dtgs",
            "itgs",
        ]:
            key_for_gradient = "gradient_2dgs"
        else:
            key_for_gradient = "means2d"

        # Densification Strategy
        self.strategy = DefaultStrategy(
            verbose=True,
            prune_opa=cfg.prune_opa,
            grow_grad2d=cfg.grow_grad2d,
            grow_scale3d=cfg.grow_scale3d,
            prune_scale3d=cfg.prune_scale3d,
            # refine_scale2d_stop_iter=4000, # splatfacto behavior
            refine_start_iter=cfg.refine_start_iter,
            refine_stop_iter=cfg.refine_stop_iter,
            reset_every=cfg.reset_every,
            refine_every=cfg.refine_every,
            absgrad=cfg.absgrad,
            revised_opacity=cfg.revised_opacity,
            key_for_gradient=key_for_gradient,
        )
        self.strategy.check_sanity(self.splats, self.optimizers)
        self.strategy_state = self.cfg.strategy.initialize_state()

        self.texture_optimizers = []
        if (
            self.texture_model is not None
            and len(list(self.texture_model.parameters())) > 0
        ):
            self.texture_optimizers = [
                torch.optim.Adam(
                    self.texture_model.parameters(),
                    lr=2.5e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15 / math.sqrt(cfg.batch_size),
                    betas=(
                        1 - cfg.batch_size * (1 - 0.9),
                        1 - cfg.batch_size * (1 - 0.999),
                    ),
                )
            ]

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)

        self.app_optimizers = []
        if cfg.app_opt:
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(
            self.device
        )

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )

    def get_textures(self):
        # textures: [N, L, L, 4]
        textures = self.splats["textures"]
        if not self.cfg.textured_rgb:
            rgb_textures = torch.zeros_like(textures[..., :3])  # [N, L, L, 3]
        else:
            rgb_textures = textures[..., :3]  # [N, L, L, 3]
            match self.cfg.textured_rgb_clamp:
                case "none":
                    pass
                case "normalize":
                    rgb_textures = rgb_textures / (
                        rgb_textures.amax(dim=[1, 2], keepdim=True) + 1e-6
                    )
                    rgb_textures = rgb_textures.clamp(0.0, 1.0)
                case "clamp":
                    rgb_textures = rgb_textures.clamp(0.0, 1.0)
                case "sigmoid":
                    rgb_textures = rgb_textures.sigmoid()
                case _:
                    print(f"Unknown clamp type {self.cfg.textured_rgb_clamp}")
        if not self.cfg.textured_alpha:
            alpha_textures = torch.ones_like(textures[..., 3:4])  # [N, L, L, 1]
        else:
            alpha_textures = textures[..., 3:4]  # [N, L, L, 1]
            match self.cfg.textured_alpha_clamp:
                case "none":
                    pass
                case "normalize":
                    alpha_textures = alpha_textures / (
                        alpha_textures.amax(dim=[1, 2], keepdim=True) + 1e-6
                    )  # normalize so that the max is 1
                    alpha_textures = alpha_textures.clamp(0.0, 1.0)
                case "clamp":
                    alpha_textures = alpha_textures.clamp(0.0, 0.9999)
                case "sigmoid":
                    alpha_textures = alpha_textures.sigmoid() * 0.9999
                case _:
                    print(f"Unknown clamp type {self.cfg.textured_alpha_clamp}")

        textures = torch.cat([rgb_textures, alpha_textures], dim=-1)  # [N, L, L, 4]
        return textures

    @torch.no_grad()
    def render_textures(self, width: int, height: int) -> Tensor | None:
        """Render textures for all Gaussians.

        Args:
            width: Width to render each texture at.
            height: Height to render each texture at.

        Returns:
            Tensor of shape [N, height, width, 4] (RGBA in [0, 1]), or None
            if the model type has no textures.
        """
        match self.model_type:
            case "tgs":
                if self.cfg.app_opt:
                    colors = self.splats["colors"]
                    colors = torch.sigmoid(colors)
                    colors = colors.unsqueeze(-1).unsqueeze(-1)  # [N, 4, 1, 1]
                else:
                    colors = self.splats["sh0"].squeeze(1)
                    colors = torch.cat(
                        [
                            colors,
                            torch.zeros(colors.shape[0], 1, device=colors.get_device()),
                        ],
                        dim=1,
                    )
                    colors = colors.unsqueeze(-1).unsqueeze(-1)  # [N, 4, 1, 1]

                textures = self.get_textures()  # [N, H, W, 4]
                texture_height = textures.shape[1]
                texture_width = textures.shape[2]
                textures = textures.permute(0, 3, 1, 2)  # [N, 4, H, W]
                if self.cfg.textured_rgb and self.cfg.textured_alpha:
                    textures += colors
                elif self.cfg.textured_rgb:
                    textures += colors[:, :3, :, :]
                elif self.cfg.textured_alpha:
                    textures += colors[:, 3:, :, :]

                if texture_height >= height and texture_width >= width:
                    textures = F.interpolate(
                        textures, size=(height, width), mode="nearest"
                    )
                else:
                    textures = F.interpolate(
                        textures,
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                    )
                return textures.permute(0, 2, 3, 1)  # [N, H, W, 4]
            case "dtgs":
                if self.cfg.app_opt:
                    colors = self.splats["colors"]
                    colors = torch.sigmoid(colors)
                    colors = colors.unsqueeze(-1).unsqueeze(-1)  # [N, 4, 1, 1]
                else:
                    colors = self.splats["sh0"].squeeze(1)
                    colors = torch.cat(
                        [
                            colors,
                            torch.zeros(colors.shape[0], 1, device=colors.get_device()),
                        ],
                        dim=1,
                    )
                    colors = colors.unsqueeze(-1).unsqueeze(-1)  # [N, 4, 1, 1]

                textures = self.get_textures()  # [N, H, W, 4]
                rendered = rasterize_dct_textures(textures, height, width, 16)

                rendered = rendered.permute(0, 3, 1, 2)  # [N, 4, H, W]
                if self.cfg.textured_rgb and self.cfg.textured_alpha:
                    rendered += colors
                elif self.cfg.textured_rgb:
                    rendered += colors[:, :3, :, :]
                elif self.cfg.textured_alpha:
                    rendered += colors[:, 3:, :, :]
                return rendered.permute(0, 2, 3, 1)  # [N, H, W, 4]
            case "itgs":
                match self.cfg.texture_input_type:
                    case "gaussian":
                        N = len(self.splats["means"])
                        device = self.device
                        v_coords = torch.linspace(0, 1, height, device=device)
                        u_coords = torch.linspace(0, 1, width, device=device)
                        grid_v, grid_u = torch.meshgrid(
                            v_coords, u_coords, indexing="ij"
                        )
                        uv_flat = torch.stack(
                            [grid_u.reshape(-1), grid_v.reshape(-1)], dim=-1
                        )  # [H*W, 2]

                        textures = torch.empty(N, height, width, 4, device=device)
                        batch_size = max(1, 1024 * 1024 // (height * width))
                        for start in range(0, N, batch_size):
                            end = min(start + batch_size, N)
                            B = end - start
                            g_indices = torch.arange(
                                start, end, device=device, dtype=torch.float32
                            ) / max(N - 1, 1)
                            g_coords = g_indices[:, None, None].expand(
                                B, height * width, 1
                            )
                            uv_expanded = uv_flat[None].expand(B, -1, -1)
                            inputs = torch.cat([g_coords, uv_expanded], dim=-1).reshape(
                                B * height * width, 3
                            )
                            outputs = self.texture_model(inputs)
                            textures[start:end] = outputs.reshape(B, height, width, 4)
                        return textures
        return None

    @torch.no_grad()
    def resize_textures(self, width: int, height: int):
        match self.model_type:
            case "tgs":
                # textures: [N, L, L, 4]
                textures = self.splats["textures"]
                textures = textures.permute(0, 3, 1, 2)  # [N, 4, L, L]
                textures = F.interpolate(
                    textures, size=(height, width), mode="bicubic", align_corners=False
                )
                textures = textures.permute(0, 2, 3, 1)  # [N, H, W, 4]
                self.splats["textures"] = textures
                # The ParameterDict creates a new Parameter object on assignment,
                # so the optimizer must be updated to reference it. The optimizer
                # state is also cleared because the tensor shape has changed.
                opt = self.optimizers["textures"]
                opt.param_groups[0]["params"] = [self.splats["textures"]]
                opt.state.clear()
            case "dtgs":
                print("DCT texture resizing is not yet supported.")

    @torch.no_grad()
    def render_textures_video(self, width: int, height: int, step: int):
        """Render all Gaussian textures into a video (RGB only).

        Args:
            width: Width to render each texture at.
            height: Height to render each texture at.
            step: Training step, used for the output filename.
        """
        if self.model_type not in (
            "tgs",
            "dtgs",
            "itgs",
        ):
            print("No textures to render for model type:", self.model_type)
            return

        print("Rendering texture video...")
        N = len(self.splats["means"])
        video_dir = f"{self.cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        video_path = f"{video_dir}/textures{width}x{height}_{step}.mp4"
        writer = imageio.get_writer(video_path, fps=30)

        if self.model_type == "tgs":
            all_textures = self.get_textures()  # [N, L, L, 4]
            batch_size = 4096
            for start in tqdm.trange(0, N, batch_size, desc="Rendering texture video"):
                end = min(start + batch_size, N)
                batch = all_textures[start:end].permute(0, 3, 1, 2)  # [B, 4, L, L]
                batch = F.interpolate(
                    batch, size=(height, width), mode="bilinear", align_corners=False
                )
                batch = batch.permute(0, 2, 3, 1)  # [B, H, W, 4]
                for j in range(end - start):
                    frame = (batch[j, :, :, :3].clamp(0, 1).cpu().numpy() * 255).astype(
                        np.uint8
                    )
                    writer.append_data(frame)
        elif self.model_type == "dtgs":
            all_textures = self.get_textures()  # [N, L, L, 4]
            batch_size = 4096
            for start in tqdm.trange(0, N, batch_size, desc="Rendering texture video"):
                end = min(start + batch_size, N)
                batch = all_textures[start:end]  # [B, L, L, 4]
                rendered = rasterize_dct_textures(
                    batch, height, width, 16
                )  # [B, H, W, 4]
                for j in range(end - start):
                    frame = (
                        rendered[j, :, :, :3].clamp(0, 1).cpu().numpy() * 255
                    ).astype(np.uint8)
                    writer.append_data(frame)
        elif self.model_type == "itgs":
            match self.cfg.texture_input_type:
                case "gaussian":
                    device = self.device
                    v_coords = torch.linspace(0, 1, height, device=device)
                    u_coords = torch.linspace(0, 1, width, device=device)
                    grid_v, grid_u = torch.meshgrid(v_coords, u_coords, indexing="ij")
                    uv_flat = torch.stack(
                        [grid_u.reshape(-1), grid_v.reshape(-1)], dim=-1
                    )  # [H*W, 2]

                    batch_size = max(1, 1024 * 1024 // (height * width))
                    for start in tqdm.trange(
                        0, N, batch_size, desc="Rendering texture video"
                    ):
                        end = min(start + batch_size, N)
                        B = end - start
                        g_indices = torch.arange(
                            start, end, device=device, dtype=torch.float32
                        ) / max(N - 1, 1)
                        g_coords = g_indices[:, None, None].expand(B, height * width, 1)
                        uv_expanded = uv_flat[None].expand(B, -1, -1)
                        inputs = torch.cat([g_coords, uv_expanded], dim=-1).reshape(
                            B * height * width, 3
                        )
                        outputs = self.texture_model(inputs).reshape(
                            B, height, width, 4
                        )
                        for j in range(B):
                            frame = (
                                outputs[j, :, :, :3].clamp(0, 1).cpu().numpy() * 255
                            ).astype(np.uint8)
                            writer.append_data(frame)

        writer.close()
        print(f"Texture video saved to {video_path}")

    @torch.no_grad()
    def save_texture_images(self, width: int, height: int, step: int):
        """Save each Gaussian's texture as an individual image.

        Args:
            width: Width to render each texture at.
            height: Height to render each texture at.
            step: Training step, used for the output directory name.
        """
        if self.model_type not in (
            "tgs",
            "dtgs",
            "itgs",
        ):
            return

        print("Saving texture images...")

        texture_zip = f"{self.cfg.result_dir}/textures{width}x{height}/step_{step}.zip"
        os.makedirs(os.path.dirname(texture_zip), exist_ok=True)

        textures = self.render_textures(width, height)  # [N, H, W, 4]
        if textures is None:
            return

        N = textures.shape[0]
        with zipfile.ZipFile(texture_zip, "w", compression=zipfile.ZIP_STORED) as zf:
            for i in tqdm.trange(N, desc="Saving texture images"):
                rgba = (textures[i].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                buf = io.BytesIO()
                imageio.imwrite(buf, rgba, format="png")
                zf.writestr(f"{i:06d}.png", buf.getvalue())

        print(f"Saved {N} texture images to {texture_zip}")

    def _compute_coord_normalisation(
        self, cfg: "Config"
    ) -> tuple[Tensor | None, Tensor | None]:
        """Compute (coord_center, coord_scale) from the training cameras.

        Returns (None, None) when strategy is "none" or texture_input_type is
        "gaussian" (world coords are not used so normalisation is irrelevant).
        """
        if (
            cfg.world_sample_normalisation == "none"
            or cfg.texture_input_type == "gaussian"
        ):
            return None, None

        if cfg.dataset_type == "colmap":
            camtoworlds = torch.from_numpy(self.parser.camtoworlds).float()
        else:  # blender
            camtoworlds = torch.from_numpy(self.trainset.camtoworlds).float()

        match cfg.world_sample_normalisation:
            case "unit_sphere":
                center, scale = compute_camera_unit_sphere_normalization(
                    camtoworlds, strict=False
                )
            case "unit_sphere_strict":
                center, scale = compute_camera_unit_sphere_normalization(
                    camtoworlds, strict=True
                )
            case "bbox":
                bbox_min, bbox_max = compute_scene_bbox_from_cameras(camtoworlds)
                center, scale = compute_bbox_normalization(bbox_min, bbox_max)

        device = torch.device(self.device)
        return center.to(device), scale.to(device)

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        **kwargs,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, dict]:
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]

        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        assert self.cfg.antialiased is False, "Antialiased is not supported for 2DGS"

        match self.model_type:
            case "2dgs":
                remove_from_kwargs(
                    kwargs,
                    {
                        "num_texture_samples",
                        "filtering",
                        "sample_alpha_threshold",
                        "texture_batch_size",
                        "texture_grad_method",
                        "texture_input_type",
                        "coord_center",
                        "coord_scale",
                    },
                )
                (
                    render_colors,
                    render_alphas,
                    render_normals,
                    normals_from_depth,
                    render_distort,
                    render_median,
                    _,
                    _,
                    info,
                ) = rasterization_2dgs(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=colors,
                    viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
                    Ks=Ks,  # [C, 3, 3]
                    width=width,
                    height=height,
                    packed=self.cfg.packed,
                    absgrad=self.cfg.absgrad,
                    sparse_grad=self.cfg.sparse_grad,
                    **kwargs,
                )
            case "tgs":
                remove_from_kwargs(
                    kwargs,
                    {
                        "num_texture_samples",
                        "sample_alpha_threshold",
                        "texture_batch_size",
                        "texture_grad_method",
                        "texture_input_type",
                        "coord_center",
                        "coord_scale",
                    },
                )
                textures = self.get_textures()
                (
                    render_colors,
                    render_alphas,
                    render_normals,
                    normals_from_depth,
                    render_distort,
                    render_median,
                    _,
                    _,
                    info,
                ) = rasterization_textured_gaussians(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=colors,
                    textures=textures,
                    viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
                    Ks=Ks,  # [C, 3, 3]
                    width=width,
                    height=height,
                    packed=self.cfg.packed,
                    absgrad=self.cfg.absgrad,
                    sparse_grad=self.cfg.sparse_grad,
                    **kwargs,
                )
            case "dtgs":
                remove_from_kwargs(
                    kwargs,
                    {
                        "num_texture_samples",
                        "sample_alpha_threshold",
                        "filtering",
                        "texture_batch_size",
                        "texture_grad_method",
                        "texture_input_type",
                        "coord_center",
                        "coord_scale",
                    },
                )
                textures = self.get_textures()
                (
                    render_colors,
                    render_alphas,
                    render_normals,
                    normals_from_depth,
                    render_distort,
                    render_median,
                    _,
                    _,
                    info,
                ) = rasterization_dct_textured_gaussians(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=colors,
                    textures=textures,
                    viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
                    Ks=Ks,  # [C, 3, 3]
                    width=width,
                    height=height,
                    packed=self.cfg.packed,
                    absgrad=self.cfg.absgrad,
                    sparse_grad=self.cfg.sparse_grad,
                    **kwargs,
                )
            case "itgs":
                remove_from_kwargs(kwargs, {"filtering"})
                (
                    render_colors,
                    render_alphas,
                    render_normals,
                    normals_from_depth,
                    render_distort,
                    render_median,
                    _,
                    _,
                    info,
                ) = rasterization_implicit_textured_gaussians(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=colors,
                    textures=self.texture_model,
                    viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
                    Ks=Ks,  # [C, 3, 3]
                    width=width,
                    height=height,
                    packed=self.cfg.packed,
                    absgrad=self.cfg.absgrad,
                    sparse_grad=self.cfg.sparse_grad,
                    **kwargs,
                )
        return (
            render_colors,
            render_alphas,
            render_normals,
            normals_from_depth,
            render_distort,
            render_median,
            _,
            _,
            info,
        )

    def train(self):
        cfg = self.cfg
        device = self.device

        # Dump cfg.
        # with open(f"{cfg.result_dir}/cfg.json", "w") as f:
        #     json.dump(vars(cfg), f)

        with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
            yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = cfg.init_step

        base_color_factor = None
        if cfg.base_color_factor is not None and cfg.model_type == "itgs":
            base_color_factor = load_factor(cfg.base_color_factor)

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        prev_ckpt_step = None
        max_mem = 0

        if cfg.checkpoint_path is not None:
            _, train_state_path = cfg.checkpoint_path
            train_state = torch.load(
                train_state_path, map_location=device, weights_only=False
            )
            for name, state in train_state["optimizers"].items():
                self.optimizers[name].load_state_dict(state)
            for opt, state in zip(self.pose_optimizers, train_state["pose_optimizers"]):
                opt.load_state_dict(state)
            if cfg.pose_opt:
                self.pose_adjust.load_state_dict(train_state["pose_adjust"])
            for opt, state in zip(self.app_optimizers, train_state["app_optimizers"]):
                opt.load_state_dict(state)
            if cfg.app_opt:
                self.app_module.load_state_dict(train_state["app_module"])
            for opt, state in zip(
                self.texture_optimizers, train_state["texture_optimizers"]
            ):
                opt.load_state_dict(state)
            for s, state in zip(schedulers, train_state["schedulers"]):
                s.load_state_dict(state)
            self.strategy_state = train_state["strategy"]
            if base_color_factor is not None:
                base_color_factor.load_state_dict(train_state["base_color_factor"])
            global_tic -= train_state["elapsed_time"]
            max_mem = train_state["max_mem"]

        for step in pbar:

            self.step = step

            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]
            if cfg.alpha_loss:
                alphas_gt = data["alpha"].to(device)  # [1, H, W]

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            if self.cfg.pretrained_path is not None:
                sh_degree_to_use = cfg.sh_degree
            else:
                sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            opt_kwargs = {}
            if base_color_factor is not None:
                self.base_color_factor = base_color_factor.get_value(step)
                opt_kwargs["base_color_factor"] = self.base_color_factor

            # forward
            (
                renders,
                alphas,
                normals,
                normals_from_depth,
                render_distort,
                render_median,
                _,
                _,
                info,
            ) = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB+D",
                distloss=self.cfg.dist_loss,
                filtering=self.cfg.filtering,
                num_texture_samples=self.cfg.num_texture_samples,
                sample_alpha_threshold=self.cfg.sample_alpha_threshold,
                texture_batch_size=self.cfg.texture_batch_size,
                texture_grad_method=self.cfg.texture_grad_method,
                texture_input_type=self.cfg.texture_input_type,
                coord_center=self.coord_center,
                coord_scale=self.coord_scale,
                **opt_kwargs,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.background_mode is not None:
                match cfg.background_mode:
                    case "random":
                        bkgd = torch.rand(1, 3, device=device)
                        colors = colors + bkgd * (1.0 - alphas)
                    case "white":
                        colors = colors + 1.0 * (1.0 - alphas)
                    case "black":
                        colors = colors + 0.0 * (1.0 - alphas)
                    case _:
                        raise ValueError(
                            f"Background mode {cfg.background_mode} not supported!"
                        )

            self.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - self.ssim(
                pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2)
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            if cfg.depth_loss:
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt  # [1, M]
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda

            if cfg.normal_loss:
                if step > cfg.normal_start_iter:
                    curr_normal_lambda = cfg.normal_lambda
                else:
                    curr_normal_lambda = 0.0
                # normal consistency loss
                normals = normals.squeeze(0).permute((2, 0, 1))
                normals_from_depth *= alphas.squeeze(0).detach()
                if len(normals_from_depth.shape) == 4:
                    normals_from_depth = normals_from_depth.squeeze(0)
                normals_from_depth = normals_from_depth.permute((2, 0, 1))
                normal_error = (1 - (normals * normals_from_depth).sum(dim=0))[None]
                normalloss = curr_normal_lambda * normal_error.mean()
                loss += normalloss

            if cfg.dist_loss:
                if step > cfg.dist_start_iter:
                    curr_dist_lambda = cfg.dist_lambda
                else:
                    curr_dist_lambda = 0.0
                distloss = render_distort.mean()
                loss += distloss * curr_dist_lambda

            if cfg.alpha_loss:
                alphas = alphas.squeeze(-1)  # [1, H, W]
                alpha_error = (alphas - alphas_gt).abs().mean()
                alpha_loss = cfg.alpha_lambda * alpha_error
                loss += alpha_loss

            if cfg.scale_loss:
                max_scale = torch.exp(self.splats["scales"]).amax(dim=-1)
                scale_loss = cfg.scale_lambda * max_scale.mean()
                loss += scale_loss

            if cfg.freq_loss and cfg.model_type == "dtgs":
                textures = self.splats["textures"]  # [N, L_y, L_x, 4]
                L_y, L_x = textures.shape[1], textures.shape[2]
                freq_i = torch.arange(L_x, device=textures.device, dtype=textures.dtype)
                freq_j = torch.arange(L_y, device=textures.device, dtype=textures.dtype)
                # weight[j, i] = i + j — higher for high-frequency coefficients
                freq_weight = (
                    (freq_i[None, :] + freq_j[:, None]).unsqueeze(0).unsqueeze(-1)
                )
                freqloss = cfg.freq_lambda * (textures * freq_weight).pow(2).mean()
                loss += freqloss

            loss.backward()

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.dist_loss:
                desc += f"dist loss={distloss.item():.6f}"
            if cfg.normal_loss:
                desc += f"normal loss={normalloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            if cfg.alpha_loss:
                desc += f"alpha loss={alpha_loss.item():.6f}| "
            if cfg.scale_loss:
                desc += f"scale loss={scale_loss.item():.6f}| "
            if cfg.freq_loss and cfg.model_type == "dtgs":
                desc += f"freq loss={freqloss.item():.6f}| "
            pbar.set_description(desc)

            if cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = max(max_mem, torch.cuda.max_memory_allocated() / 1024**3)
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.normal_loss:
                    self.writer.add_scalar("train/normalloss", normalloss.item(), step)
                if cfg.dist_loss:
                    self.writer.add_scalar("train/distloss", distloss.item(), step)
                if cfg.freq_loss and cfg.model_type == "dtgs":
                    self.writer.add_scalar("train/freqloss", freqloss.item(), step)
                if cfg.tb_save_image:
                    canvas = (
                        torch.cat([pixels, colors[..., :3]], dim=2)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # Run post-backward steps after backward and optimizer
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            # optimize
            if self.cfg.freeze_geometry is None or step < self.cfg.freeze_geometry:
                for optimizer in self.optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            else:
                for opt_key in [
                    "opacities",
                    "sh0",
                    "shN",
                    "features",
                    "colors",
                    "textures",
                ]:
                    if opt_key in self.optimizers:
                        optimizer = self.optimizers[opt_key]
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.texture_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            if step + 1 in cfg.texture_resize_points:
                width, height = cfg.texture_resize_points[step + 1]
                print(f"Resizing textures to {width}x{height}")
                self.resize_textures(width, height)

            # save checkpoint
            if (
                (cfg.save_every is not None and (step + 1) % cfg.save_every == 0)
                or (cfg.save_steps is not None and step + 1 in cfg.save_steps)
                or step + 1 == max_steps
            ):
                if step + 1 not in cfg.eval_steps and step + 1 != max_steps:
                    mem = max(max_mem, torch.cuda.max_memory_allocated() / 1024**3)
                    stats = {
                        "mem": mem,
                        "ellipse_time": time.time() - global_tic,
                        "num_GS": len(self.splats["means"]),
                    }
                    # print("Step: ", step, stats)
                    with open(f"{self.stats_dir}/train_step{step:04d}.json", "w") as f:
                        json.dump(stats, f)
                    ckpt_data = {
                        "step": step,
                        "splats": self.splats.state_dict(),
                    }
                    if self.texture_model is not None:
                        ckpt_data["texture_model"] = self.texture_model.state_dict()
                    if base_color_factor is not None:
                        ckpt_data["base_color_factor"] = opt_kwargs["base_color_factor"]
                    torch.save(ckpt_data, f"{self.ckpt_dir}/ckpt_{step}.pt")
                train_state_data = {
                    "optimizers": {
                        name: opt.state_dict() for name, opt in self.optimizers.items()
                    },
                    "pose_optimizers": [
                        opt.state_dict() for opt in self.pose_optimizers
                    ],
                    "app_optimizers": [opt.state_dict() for opt in self.app_optimizers],
                    "texture_optimizers": [
                        opt.state_dict() for opt in self.texture_optimizers
                    ],
                    "schedulers": [s.state_dict() for s in schedulers],
                    "strategy": self.strategy_state,
                    "max_mem": mem,
                    "elapsed_time": time.time() - global_tic,
                }
                if cfg.pose_opt:
                    train_state_data["pose_adjust"] = self.pose_adjust.state_dict()
                if cfg.app_opt:
                    train_state_data["app_module"] = self.app_module.state_dict()
                if base_color_factor:
                    train_state_data["base_color_factor"] = (
                        base_color_factor.state_dict()
                    )
                torch.save(train_state_data, f"{self.ckpt_dir}/train_state_{step}.pt")
                if prev_ckpt_step is not None:
                    for fname in (
                        f"{self.ckpt_dir}/ckpt_{prev_ckpt_step}.pt",
                        f"{self.ckpt_dir}/train_state_{prev_ckpt_step}.pt",
                    ):
                        if os.path.exists(fname):
                            os.remove(fname)

                if (
                    step not in [i - 1 for i in cfg.eval_steps]
                    and step != max_steps - 1
                ):
                    prev_ckpt_step = step

            # eval the full set
            if step + 1 in cfg.eval_steps or step + 1 == max_steps:
                mem = max(max_mem, torch.cuda.max_memory_allocated() / 1024**3)
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(f"{self.stats_dir}/train_step{step:04d}.json", "w") as f:
                    json.dump(stats, f)
                ckpt_data = {
                    "step": step,
                    "splats": self.splats.state_dict(),
                }
                if self.texture_model is not None:
                    ckpt_data["texture_model"] = self.texture_model.state_dict()
                if base_color_factor is not None:
                    ckpt_data["base_color_factor"] = opt_kwargs["base_color_factor"]
                torch.save(ckpt_data, f"{self.ckpt_dir}/ckpt_{step}.pt")
                self.eval(step)

            if step + 1 in cfg.render_traj_steps or step + 1 == max_steps:
                self.render_traj(step)

            if step + 1 in cfg.render_texture_steps or step + 1 == max_steps:
                self.render_textures_video(
                    width=cfg.saved_texture_width,
                    height=cfg.saved_texture_height,
                    step=step,
                )
                self.save_texture_images(
                    width=cfg.saved_texture_width,
                    height=cfg.saved_texture_height,
                    step=step,
                )

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_rays_per_sec
                )
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step: int):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=0
        )
        ellipse_time = 0
        metrics = {"psnr": [], "ssim": [], "lpips": []}
        render_zip_path = f"{self.render_dir}/step_{step}.zip"
        render_zip = zipfile.ZipFile(
            render_zip_path, "w", compression=zipfile.ZIP_STORED
        )
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]

            opt_kwargs = {}
            if self.base_color_factor is not None:
                opt_kwargs["base_color_factor"] = self.base_color_factor

            torch.cuda.synchronize()
            tic = time.time()
            (
                colors,
                alphas,
                normals,
                normals_from_depth,
                render_distort,
                render_median,
                _,
                _,
                _,
            ) = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
                filtering=self.cfg.filtering,
                num_texture_samples=self.cfg.num_texture_samples,
                sample_alpha_threshold=self.cfg.sample_alpha_threshold,
                texture_batch_size=self.cfg.texture_batch_size,
                texture_grad_method=self.cfg.texture_grad_method,
                texture_input_type=self.cfg.texture_input_type,
                coord_center=self.coord_center,
                coord_scale=self.coord_scale,
                **opt_kwargs,
            )  # [1, H, W, 3]
            colors = colors[..., :3]  # Take RGB channels

            if cfg.background_mode is not None:
                if cfg.background_mode == "random":
                    bkgd = torch.rand(1, 3, device=device)
                    colors = colors + bkgd * (1.0 - alphas)
                elif cfg.background_mode == "white":
                    colors = colors + 1.0 * (1.0 - alphas)
                elif cfg.background_mode == "black":
                    colors = colors + 0.0 * (1.0 - alphas)
                else:
                    raise ValueError(
                        f"Background mode {cfg.background_mode} not supported!"
                    )
            colors = torch.clamp(colors, 0.0, 1.0)

            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            # write images
            canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
            buf = io.BytesIO()
            imageio.imwrite(buf, (canvas * 255).astype(np.uint8), format="png")
            render_zip.writestr(f"val_{i:04d}.png", buf.getvalue())

            # write median depths
            render_median = (render_median - render_median.min()) / (
                render_median.max() - render_median.min()
            )
            # render_median = render_median.detach().cpu().squeeze(0).unsqueeze(-1).repeat(1, 1, 3).numpy()
            render_median = (
                render_median.detach().cpu().squeeze(0).repeat(1, 1, 3).numpy()
            )

            buf = io.BytesIO()
            imageio.imwrite(buf, (render_median * 255).astype(np.uint8), format="png")
            render_zip.writestr(f"val_{i:04d}_median_depth_{step}.png", buf.getvalue())

            # write normals
            normals = (normals * 0.5 + 0.5).squeeze(0).cpu().numpy()
            normals_output = (normals * 255).astype(np.uint8)
            buf = io.BytesIO()
            imageio.imwrite(buf, normals_output, format="png")
            render_zip.writestr(f"val_{i:04d}_normal_{step}.png", buf.getvalue())

            # write normals from depth
            normals_from_depth *= alphas.squeeze(0).detach()
            normals_from_depth = (normals_from_depth * 0.5 + 0.5).cpu().numpy()
            normals_from_depth = (normals_from_depth - np.min(normals_from_depth)) / (
                np.max(normals_from_depth) - np.min(normals_from_depth)
            )
            normals_from_depth_output = (normals_from_depth * 255).astype(np.uint8)
            if len(normals_from_depth_output.shape) == 4:
                normals_from_depth_output = normals_from_depth_output.squeeze(0)
            buf = io.BytesIO()
            imageio.imwrite(buf, normals_from_depth_output, format="png")
            render_zip.writestr(
                f"val_{i:04d}_normals_from_depth_{step}.png", buf.getvalue()
            )

            # write distortions

            render_dist = render_distort
            dist_max = torch.max(render_dist)
            dist_min = torch.min(render_dist)
            render_dist = (render_dist - dist_min) / (dist_max - dist_min)
            render_dist = (
                colormap(render_dist.cpu().numpy()[0])
                .permute((1, 2, 0))
                .numpy()
                .astype(np.uint8)
            )
            buf = io.BytesIO()
            imageio.imwrite(buf, render_dist, format="png")
            render_zip.writestr(f"val_{i:04d}_distortions_{step}.png", buf.getvalue())

            pixels = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
            colors = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
            metrics["psnr"].append(self.psnr(colors, pixels))
            metrics["ssim"].append(self.ssim(colors, pixels))
            metrics["lpips"].append(self.lpips(colors, pixels))

        render_zip.close()
        print(f"Saved render images to {render_zip_path}")

        ellipse_time /= len(valloader)

        psnr = torch.stack(metrics["psnr"]).mean()
        ssim = torch.stack(metrics["ssim"]).mean()
        lpips = torch.stack(metrics["lpips"]).mean()
        print(
            f"PSNR: {psnr.item():.3f}, SSIM: {ssim.item():.4f}, LPIPS: {lpips.item():.3f} "
            f"Time: {ellipse_time:.3f}s/image "
            f"Number of GS: {len(self.splats['means'])}"
        )
        # save stats as json
        stats = {
            "psnr": psnr.item(),
            "ssim": ssim.item(),
            "lpips": lpips.item(),
            "ellipse_time": ellipse_time,
            "num_GS": len(self.splats["means"]),
        }
        with open(f"{self.stats_dir}/val_step{step:04d}.json", "w") as f:
            json.dump(stats, f)
        # save stats to tensorboard
        for k, v in stats.items():
            self.writer.add_scalar(f"val/{k}", v, step)
        self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device
        if cfg.dataset_type == "colmap":
            camtoworlds = self.parser.camtoworlds[5:-5]
            camtoworlds = generate_interpolated_path(camtoworlds, 1)  # [N, 3, 4]
            camtoworlds = np.concatenate(
                [
                    camtoworlds,
                    np.repeat(
                        np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds), axis=0
                    ),
                ],
                axis=1,
            )  # [N, 4, 4]

            camtoworlds = torch.from_numpy(camtoworlds).float().to(device)
            K = (
                torch.from_numpy(list(self.parser.Ks_dict.values())[0])
                .float()
                .to(device)
            )
            width, height = list(self.parser.imsize_dict.values())[0]
        elif cfg.dataset_type == "blender":
            camtoworlds = np.stack(self.trainset.camtoworlds)  # [N, 4, 4]
            camtoworlds = generate_interpolated_path(camtoworlds, 1)  # [N, 3, 4]
            camtoworlds = np.concatenate(
                [
                    camtoworlds,
                    np.repeat(
                        np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds), axis=0
                    ),
                ],
                axis=1,
            )  # [N, 4, 4]
            camtoworlds = torch.from_numpy(camtoworlds).float().to(device)
            K = torch.from_numpy(self.trainset.K).float().to(device)
            width, height = self.trainset.image_size, self.trainset.image_size

        canvas_all = []
        for i in tqdm.trange(len(camtoworlds), desc="Rendering trajectory"):
            opt_kwargs = {}
            if self.base_color_factor is not None:
                opt_kwargs["base_color_factor"] = self.base_color_factor

            renders, _, _, surf_normals, _, _, _, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds[i : i + 1],
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
                filtering=self.cfg.filtering,
                num_texture_samples=self.cfg.num_texture_samples,
                sample_alpha_threshold=self.cfg.sample_alpha_threshold,
                texture_batch_size=self.cfg.texture_batch_size,
                texture_grad_method=self.cfg.texture_grad_method,
                texture_input_type=self.cfg.texture_input_type,
                coord_center=self.coord_center,
                coord_scale=self.coord_scale,
                **opt_kwargs,
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[0, ..., 0:3], 0.0, 1.0)  # [H, W, 3]
            depths = renders[0, ..., 3:4]  # [H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())

            surf_normals = (surf_normals - surf_normals.min()) / (
                surf_normals.max() - surf_normals.min()
            )

            # write images
            canvas = torch.cat(
                [colors, depths.repeat(1, 1, 3)], dim=1 if width > height else 1
            )
            canvas = (canvas.cpu().numpy() * 255).astype(np.uint8)
            canvas_all.append(canvas)

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for canvas in canvas_all:
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def render_camera_path(self, step: int, camera_path_file: str):
        """Render a video from a nerfview camera path JSON file."""
        import json as _json

        print(f"Rendering camera path from {camera_path_file}...")
        cfg = self.cfg
        device = self.device

        with open(camera_path_file) as f:
            path_data = _json.load(f)

        render_width = int(path_data["render_width"])
        render_height = int(path_data["render_height"])
        fps = float(path_data["fps"])
        frames = path_data["camera_path"]

        # nerfview applies two transforms when saving the camera path JSON:
        #   1. right-multiplies rotation by R_x180 (180 deg about X)
        #   2. divides translation by scale_ratio (default 10.0)
        # We must undo both to get back to the coordinate frame rasterize_splats expects.
        R_x180 = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
        scale_ratio = 10.0

        canvas_all = []
        for frame in tqdm.tqdm(frames, desc="Rendering camera path"):
            # Reconstruct c2w (4x4) from flat row-major list, then undo nerfview transforms
            c2w = np.array(frame["camera_to_world"], dtype=np.float32).reshape(4, 4)
            c2w[:3, :3] = c2w[:3, :3] @ R_x180  # undo right-multiply by R_x180
            c2w[:3, 3] *= scale_ratio  # undo division by scale_ratio
            camtoworld = torch.from_numpy(c2w).float().to(device).unsqueeze(0)

            # Reconstruct K from vertical FOV (degrees)
            fov_rad = np.deg2rad(frame["fov"])
            focal = render_height / 2.0 / np.tan(fov_rad / 2.0)
            K = np.array(
                [
                    [focal, 0.0, render_width / 2.0],
                    [0.0, focal, render_height / 2.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            Ks = torch.from_numpy(K).float().to(device).unsqueeze(0)

            opt_kwargs = {}
            if self.base_color_factor is not None:
                opt_kwargs["base_color_factor"] = self.base_color_factor

            renders, _, _, _, _, _, _, _, _ = self.rasterize_splats(
                camtoworlds=camtoworld,
                Ks=Ks,
                width=render_width,
                height=render_height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB",
                filtering=cfg.filtering,
                num_texture_samples=cfg.num_texture_samples,
                sample_alpha_threshold=cfg.sample_alpha_threshold,
                texture_batch_size=cfg.texture_batch_size,
                texture_grad_method=cfg.texture_grad_method,
                texture_input_type=cfg.texture_input_type,
                coord_center=self.coord_center,
                coord_scale=self.coord_scale,
                **opt_kwargs,
            )  # [1, H, W, 3]
            colors = torch.clamp(renders[0, ..., :3], 0.0, 1.0)
            canvas_all.append((colors.cpu().numpy() * 255).astype(np.uint8))

        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        video_name = os.path.splitext(os.path.basename(camera_path_file))[0]
        out_path = f"{video_dir}/camera_path_{video_name}_{step}.mp4"
        writer = imageio.get_writer(out_path, fps=fps)
        for canvas in canvas_all:
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {out_path}")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        opt_kwargs = {}
        if self.base_color_factor is not None:
            opt_kwargs["base_color_factor"] = self.base_color_factor

        render_colors, _, _, _, _, _, _, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
            filtering=self.cfg.filtering,
            num_texture_samples=self.cfg.num_texture_samples,
            sample_alpha_threshold=self.cfg.sample_alpha_threshold,
            texture_batch_size=self.cfg.texture_batch_size,
            texture_grad_method=self.cfg.texture_grad_method,
            texture_input_type=self.cfg.texture_input_type,
            coord_center=self.coord_center,
            coord_scale=self.coord_scale,
            **opt_kwargs,
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()


def main(cfg: Config):
    runner = Runner(cfg)

    if cfg.viewer_only:
        if cfg.ckpt is not None:
            ckpt = torch.load(cfg.ckpt, map_location=runner.device)
            for k in runner.splats.keys():
                runner.splats[k].data = ckpt["splats"][k]
            if runner.texture_model is not None and "texture_model" in ckpt:
                runner.texture_model.load_state_dict(ckpt["texture_model"])
            if "base_color_factor" in ckpt:
                runner.base_color_factor = ckpt["base_color_factor"]
        input("Viewer running... Press enter to exit: ")
        exit(0)
    elif cfg.ckpt is not None and cfg.camera_path is not None:
        ckpt = torch.load(cfg.ckpt, map_location=runner.device)
        for k in runner.splats.keys():
            runner.splats[k].data = ckpt["splats"][k]
        if runner.texture_model is not None and "texture_model" in ckpt:
            runner.texture_model.load_state_dict(ckpt["texture_model"])
        if "base_color_factor" in ckpt:
            runner.base_color_factor = ckpt["base_color_factor"]
        for camera_path_file in cfg.camera_path:
            try:
                runner.render_camera_path(
                    step=ckpt["step"], camera_path_file=camera_path_file
                )
            except Exception as e:
                print(f"Encountered exception whilst rendering {camera_path_file}")
                print(e)
    elif cfg.ckpt is not None:
        # run eval only
        ckpt = torch.load(cfg.ckpt, map_location=runner.device)
        for k in runner.splats.keys():
            runner.splats[k].data = ckpt["splats"][k]
        if runner.texture_model is not None and "texture_model" in ckpt:
            runner.texture_model.load_state_dict(ckpt["texture_model"])
        if "base_color_factor" in ckpt:
            runner.base_color_factor = ckpt["base_color_factor"]
        runner.eval(step=ckpt["step"])
        runner.render_traj(step=ckpt["step"])
        runner.render_textures_video(
            width=cfg.saved_texture_width,
            height=cfg.saved_texture_height,
            step=ckpt["step"],
        )
        runner.save_texture_images(
            width=cfg.saved_texture_width,
            height=cfg.saved_texture_height,
            step=ckpt["step"],
        )
    else:
        if cfg.checkpoint_path is not None:
            ckpt_path, _ = cfg.checkpoint_path
            ckpt = torch.load(ckpt_path, map_location=runner.device)
            for k in runner.splats.keys():
                runner.splats[k].data = ckpt["splats"][k]
            if runner.texture_model is not None and "texture_model" in ckpt:
                runner.texture_model.load_state_dict(ckpt["texture_model"])
            if "base_color_factor" in ckpt:
                runner.base_color_factor = ckpt["base_color_factor"]
        runner.train()

    # if not cfg.disable_viewer:
    #     print("Viewer running... Ctrl+C to exit.")
    #     time.sleep(1000000)


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)

    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    # cfg = tyro.cli(Config)
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)
    process_config(cfg)
    main(cfg)
