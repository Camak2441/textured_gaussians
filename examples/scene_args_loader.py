import os

import yaml

from examples.config import Config
from examples.texture_models import (
    canonical_model_name,
    add_arg_to_model_name,
    pop_arg_from_model_name,
)
from examples.factor_fs import canonical_factor_name
from examples.loss_fs import canonical_loss_name
from textured_gaussians.utils import TEXTURE_INPUT_SIZES
from utils import get_file_with_max_int

_PATHS_FILE = os.path.join(os.path.dirname(__file__), "..", "cfg.yml")
_PATHS_FILE = os.path.normpath(_PATHS_FILE)


def _load_paths():
    if os.path.exists(_PATHS_FILE):
        with open(_PATHS_FILE) as f:
            return yaml.safe_load(f)
    return {}


_paths = _load_paths()
_DATA_DIR = _paths.get("data_dir", "../data")
_RESULTS_DIR = _paths.get("results_dir", "../results")

MODEL_SHORTHANDS = {
    "mix1": """FourierSelector(
            num_freqs=15,hidden_dims=[24, 24],
            sub_models=[
                "SIREN(hidden_dims=[48,48],out_dim=4,omega_0=20,hidden_omegas=1.00)",
                "SIREN(hidden_dims=[48,48],out_dim=4,omega_0=30,hidden_omegas=1.25)",
                "SIREN(hidden_dims=[48,48],out_dim=4,omega_0=40,hidden_omegas=1.50)",
                "SIREN(hidden_dims=[48,48],out_dim=4,omega_0=50,hidden_omegas=1.75)",
                "SIREN(hidden_dims=[48,48],out_dim=4,omega_0=60,hidden_omegas=2.00)",
                "FourierMLP(hidden_dims=[48,48],out_dim=4,num_frequencies=128,sigma=[1000,1,1])",
                "FourierMLP(hidden_dims=[48,48],out_dim=4,num_frequencies=128,sigma=[1000,2,2])",
                "FourierMLP(hidden_dims=[48,48],out_dim=4,num_frequencies=128,sigma=[1000,3,3])"
            ]
        )""",
    "mix2": """FourierSelector2(
            num_freqs=15,hidden_dims=[24, 24],
            sub_models=[
                "SIREN(hidden_dims=[48,48],out_dim=4,omega_0=20,hidden_omegas=1.00)",
                "SIREN(hidden_dims=[48,48],out_dim=4,omega_0=30,hidden_omegas=1.25)",
                "SIREN(hidden_dims=[48,48],out_dim=4,omega_0=40,hidden_omegas=1.50)",
                "SIREN(hidden_dims=[48,48],out_dim=4,omega_0=50,hidden_omegas=1.75)",
                "SIREN(hidden_dims=[48,48],out_dim=4,omega_0=60,hidden_omegas=2.00)",
                "FourierMLP(hidden_dims=[48,48],out_dim=4,num_frequencies=128,sigma=[1000,1,1])",
                "FourierMLP(hidden_dims=[48,48],out_dim=4,num_frequencies=128,sigma=[1000,2,2])",
                "FourierMLP(hidden_dims=[48,48],out_dim=4,num_frequencies=128,sigma=[1000,3,3])"
            ]
        )""",
    "fourier1": """FourierMLP(
        input_type="gaussian",hidden_dims=[48,48,48],out_dim=4,num_frequencies=48,sigma=[1000,3,3]
    )""",
    "fourier2": """FourierMLP(
        input_type="world_and_view",sample_norm="bbox",hidden_dims=[48,48,48],out_dim=4,num_frequencies=48,sigma=[5,5,5,3,3,3]
    )""",
    "fourier3": """FourierMLP(
        input_type="world_and_view",sample_norm="bbox",hidden_dims=[48,48,48],out_dim=4,num_frequencies=48,sigma=[20,20,20,3,3,3]
    )""",
    "siren1": """SIREN(
        input_type="gaussian",hidden_dims=[48,48,48],out_dim=4,omega_0=30,hidden_omegas=1
    )""",
    "siren2": """SIREN(
        input_type="world_and_view",sample_norm="bbox",hidden_dims=[36,36,36],out_dim=4,omega_0=60,hidden_omegas=1.5
    )""",
    "red": "ConstColor(out_color=[1,0,0,1])",
    "blue": "ConstColor(out_color=[0,0,1,1])",
}

MODEL_BATCH_SIZE = {"mix2": 917504}

FACTOR_SHORTHANDS = {
    "zero": """Constant(value=0.0)""",
    "half": """Constant(value=0.5)""",
    "c06": """Constant(value=0.6)""",
    "c07": """Constant(value=0.7)""",
    "c08": """Constant(value=0.8)""",
    "c09": """Constant(value=0.9)""",
    "one": """Constant(value=1.0)""",
    "exp": """Exponential(start_value=1.0,limit_value=0.0,half_life=1000)""",
    "exphalf": """Exponential(start_value=1.0,limit_value=0.5,half_life=1000)""",
    "expos001": """ExpOneAtStep(start_value=0.01,one_step=30000)""",
    "expos01": """ExpOneAtStep(start_value=0.1,one_step=30000)""",
    "expos02": """ExpOneAtStep(start_value=0.2,one_step=30000)""",
    "expos05": """ExpOneAtStep(start_value=0.5,one_step=30000)""",
    "lin": """LinearInterpolate(key_steps=[0, 15000],key_values=[1.0,0.0])""",
    "lin2": """LinearInterpolate(key_steps=[0, 30000],key_values=[0.0,1.0])""",
    "linhalf": """LinearInterpolate(key_steps=[0, 15000],key_values=[1.0,0.5])""",
    "quad": "Quadratic(one_step=30000)",
    "sqrt": "SquareRoot(one_step=30000)",
}

LOSS_SHORTHANDS = {
    "t00": """Triangle0""",
    "t02": """Triangle(peak=0.2)""",
    "t03": """Triangle(peak=0.3)""",
    "t04": """Triangle(peak=0.4)""",
    "t05": """Triangle(peak=0.5)""",
    "t06": """Triangle(peak=0.6)""",
    "t10": """Triangle1""",
    "stpns": """Steepness""",
    "quad1": """Quadratic1""",
    "quad01": """Quadratic01""",
    "hquad06": """HalfQuadratic(valley=0.6)""",
    "hquad07": """HalfQuadratic(valley=0.7)""",
    "hquad08": """HalfQuadratic(valley=0.8)""",
    "hquad09": """HalfQuadratic(valley=0.9)""",
}

NERF_SYNTHETIC = {
    "data_dir": "nerf_synthetic/{path_name}",
    "pretrained_dir": "2dgs/{name}/ckpts",
    "result_dir": "{name}",
    "dataset_type": "blender",
    "alpha": True,
}
CUSTOM = {
    "data_dir": "custom/{path_name}",
    "pretrained_dir": "2dgs/{name}/ckpts",
    "result_dir": "{name}",
    "dataset_type": "blender",
    "alpha": True,
}
MIP_NERF_360 = {
    "data_dir": "mip_nerf_360/{path_name}",
    "pretrained_dir": "2dgs/{name}/ckpts",
    "result_dir": "{name}",
    "dataset_type": "colmap",
    "alpha": False,
}


def make_args(base: dict[str, any], name: str, path_name: str | None = None):
    if path_name is None:
        path_name = name
    args = base.copy()
    args["data_dir"] = args["data_dir"].format(path_name=path_name)
    args["pretrained_dir"] = args["pretrained_dir"].format(name=name)
    args["result_dir"] = args["result_dir"].format(name=name)
    return args


SCENES = {
    **{
        name: make_args(NERF_SYNTHETIC, name)
        for name in {
            "chair",
            "drums",
            "ficus",
            "hotdog",
            "lego",
            "materials",
            "mic",
            "ship",
        }
    },
    **{
        name: make_args(MIP_NERF_360, name)
        for name in {
            "bicycle",
            "bonsai",
            "counter",
            "flowers",
            "garden",
            "kitchen",
            "room",
            "stump",
            "treehill",
        }
    },
    **{
        name: make_args(CUSTOM, name)
        for name in {
            "dct_cube",
            "6_color_cube",
            "s_cube",
            "s_cube2",
            "s_cube3",
            "f_cube",
        }
    },
}


def process_config(cfg: Config):

    # Load texture dimensions
    cfg.texture_width = cfg.texture_resolution
    if cfg.texture_height is None:
        cfg.texture_height = cfg.texture_resolution

    # Load saved texture dimensions
    if cfg.saved_texture_resolution is None:
        if cfg.saved_texture_width is None:
            cfg.saved_texture_width = cfg.texture_width
        if cfg.saved_texture_height is None:
            cfg.saved_texture_height = cfg.texture_height
    else:
        cfg.saved_texture_width = cfg.saved_texture_resolution
        cfg.saved_texture_height = cfg.saved_texture_resolution

    if cfg.texture_model is not None:
        if cfg.texture_model not in MODEL_SHORTHANDS:
            cfg.texture_model = canonical_model_name(cfg.texture_model)

    if cfg.base_color_factor is not None:
        if cfg.base_color_factor not in FACTOR_SHORTHANDS:
            cfg.base_color_factor = canonical_factor_name(cfg.base_color_factor)

    if cfg.sigmoid_factor is not None:
        if cfg.sigmoid_factor not in FACTOR_SHORTHANDS:
            cfg.sigmoid_factor = canonical_factor_name(cfg.sigmoid_factor)

    if cfg.opac_loss:
        if cfg.opac_loss_fn not in LOSS_SHORTHANDS:
            cfg.opac_loss_fn = canonical_loss_name(cfg.opac_loss_fn)

    if cfg.tex_opac_loss:
        if cfg.tex_opac_loss_fn not in LOSS_SHORTHANDS:
            cfg.tex_opac_loss_fn = canonical_loss_name(cfg.tex_opac_loss_fn)

    if cfg.steepness_loss:
        if cfg.steepness_loss_fn not in LOSS_SHORTHANDS:
            cfg.steepness_loss_fn = canonical_loss_name(cfg.steepness_loss_fn)

    # Configure information based on the selected scene
    if cfg.scene is not None:
        assert cfg.scene in SCENES
        scene_args = SCENES[cfg.scene]
        cfg.data_dir = _DATA_DIR + "/" + scene_args["data_dir"]

        args = []

        def create_args_suffix():
            nonlocal args, cfg
            local_args = args
            if cfg.result_dir_suffix:
                local_args += [cfg.result_dir_suffix]
            args_suffix = "_".join(local_args)
            if len(args_suffix) != 0 and args_suffix[0] != "_":
                args_suffix = "_" + args_suffix
            return args_suffix

        if cfg.init_num_pts != 10_000:
            args.append(f"g{cfg.init_num_pts}")

        if cfg.data_factor != 4:
            args.append(f"df{cfg.data_factor}")

        if cfg.opac_loss:
            args.append(f"o{cfg.opac_loss_fn}-{cfg.opac_loss_start_iter}")

        if cfg.tex_opac_loss:
            args.append(f"ot{cfg.tex_opac_loss_fn}-{cfg.tex_opac_loss_start_iter}")

        if cfg.freq_guidance:
            if cfg.freq_guidance_use_upsampled:
                args.append(
                    f"fg{cfg.freq_guidance_downsample}-{cfg.freq_guidance_start_iter}-fgu"
                )
            else:
                args.append(
                    f"fg{cfg.freq_guidance_downsample}-{cfg.freq_guidance_start_iter}"
                )

        if cfg.freq_guidance_orient:
            args.append(f"fgo{cfg.freq_guidance_orient_start_iter}")

        match cfg.model_type:
            case "tgs":
                if cfg.texture_width != 64 or cfg.texture_height != 64:
                    if cfg.texture_width == cfg.texture_height:
                        args.append(f"t{cfg.texture_width}")
                    else:
                        args.append(f"t{cfg.texture_width}x{cfg.texture_height}")
                if cfg.textured_rgb_clamp != "clamp":
                    args.append(f"rgb{cfg.textured_rgb_clamp}")
                if cfg.textured_alpha_clamp != "normalize":
                    args.append(f"a{cfg.textured_alpha_clamp}")
                if cfg.freeze_geometry != None:
                    args.append(f"to{cfg.freeze_geometry}")
                args_suffix = create_args_suffix()
                match cfg.filtering:
                    case "bilinear":
                        if cfg.result_dir is None:
                            cfg.result_dir = f"{_RESULTS_DIR}/tgs{args_suffix}/{scene_args["result_dir"]}"
                    case "bilinear_bwd2":
                        if cfg.result_dir is None:
                            cfg.result_dir = f"{_RESULTS_DIR}/tgs_b2{args_suffix}/{scene_args["result_dir"]}"
                    case "bilinear2":
                        if cfg.result_dir is None:
                            cfg.result_dir = f"{_RESULTS_DIR}/tgs2{args_suffix}/{scene_args["result_dir"]}"
                    case "bilinear3":
                        if cfg.result_dir is None:
                            cfg.result_dir = f"{_RESULTS_DIR}/tgs3{args_suffix}/{scene_args["result_dir"]}"
                    case "bilinear3_bwd2":
                        if cfg.result_dir is None:
                            cfg.result_dir = f"{_RESULTS_DIR}/tgs3_b2{args_suffix}/{scene_args["result_dir"]}"
                    case "mipmapped":
                        if cfg.result_dir is None:
                            cfg.result_dir = f"{_RESULTS_DIR}/mip_tgs{args_suffix}/{scene_args["result_dir"]}"
                    case "mipmapped2":
                        if cfg.result_dir is None:
                            cfg.result_dir = f"{_RESULTS_DIR}/mip2_tgs{args_suffix}/{scene_args["result_dir"]}"
                    case "anisotropic":
                        if cfg.result_dir is None:
                            cfg.result_dir = f"{_RESULTS_DIR}/aniso_tgs{args_suffix}/{scene_args["result_dir"]}"
                    case "anisotropic_bilinear":
                        if cfg.result_dir is None:
                            cfg.result_dir = f"{_RESULTS_DIR}/aniso_bilinear_tgs{args_suffix}/{scene_args["result_dir"]}"

            case "itgs":
                if cfg.base_color_factor is not None:
                    args = [f"{cfg.base_color_factor}"] + args
                args = [f"{cfg.texture_model}"] + args
                args_suffix = create_args_suffix()
                if cfg.result_dir is None:
                    cfg.result_dir = (
                        f"{_RESULTS_DIR}/itgs{args_suffix}/{scene_args["result_dir"]}"
                    )

            case "dtgs":
                if cfg.texture_width != 16 or cfg.texture_height != 16:
                    if cfg.texture_width == cfg.texture_height:
                        args.append(f"t{cfg.texture_width}")
                    else:
                        args.append(f"t{cfg.texture_width}x{cfg.texture_height}")
                if cfg.freeze_geometry != None:
                    args.append(f"to{cfg.freeze_geometry}")

                args_suffix = create_args_suffix()
                if cfg.result_dir is None:
                    cfg.result_dir = f"{_RESULTS_DIR}/{cfg.model_type}{args_suffix}/{scene_args["result_dir"]}"
                cfg.textured_rgb_clamp = "none"
                cfg.textured_alpha_clamp = "none"

            case "2dgss":
                args.append(f"sw{cfg.sigmoid_factor}")
                args_suffix = create_args_suffix()
                if cfg.result_dir is None:
                    cfg.result_dir = f"{_RESULTS_DIR}/{cfg.model_type}{args_suffix}/{scene_args["result_dir"]}"

            case _:
                args_suffix = create_args_suffix()
                if cfg.result_dir is None:
                    cfg.result_dir = f"{_RESULTS_DIR}/{cfg.model_type}{args_suffix}/{scene_args["result_dir"]}"

        cfg.dataset_type = scene_args["dataset_type"]

        if cfg.alpha_loss and not scene_args["alpha"]:
            print("Dataset does not have alpha channels. Disabling alpha loss.")
            cfg.alpha_loss = False

        if cfg.pretrained_path is None and cfg.init_type == "pretrained":
            pretrained_dir = f"{_RESULTS_DIR}/{scene_args["pretrained_dir"]}"
            _, ckpt_path = get_file_with_max_int(pretrained_dir, "ckpt_", ".pt")
            if ckpt_path is None:
                print(f"Unable to find pretrained data in {pretrained_dir}.")
                cfg.pretrained_path = None
            else:
                cfg.pretrained_path = f"{pretrained_dir}/{ckpt_path}"

    if cfg.resume:
        ckpt_dir = cfg.result_dir + "/ckpts"
        os.makedirs(cfg.result_dir, exist_ok=True)
        if os.path.isdir(ckpt_dir):
            start, _ = get_file_with_max_int(
                ckpt_dir, "ckpt_", ".pt", limit=cfg.max_steps
            )
            if start is None:
                print(f"No checkpoints to resume.")
                cfg.checkpoint_path = None
            else:
                print(f"Resuming from checkpoint ckpt_{start}.pt")
                cfg.init_step = start + 1
                cfg.checkpoint_path = (
                    f"{ckpt_dir}/ckpt_{start}.pt",
                    f"{ckpt_dir}/train_state_{start}.pt",
                )

    if cfg.texture_model is not None:
        if cfg.texture_model in MODEL_SHORTHANDS:
            if cfg.texture_batch_size == 0:
                if cfg.texture_model in MODEL_BATCH_SIZE:
                    cfg.texture_batch_size = MODEL_BATCH_SIZE[cfg.texture_model]
                else:
                    print(
                        f"No batch size available for {cfg.texture_model}, setting to None"
                    )
                    cfg.texture_batch_size = None
            cfg.texture_model = canonical_model_name(
                MODEL_SHORTHANDS[cfg.texture_model]
            )

        cfg.texture_model, texture_input_type = pop_arg_from_model_name(
            cfg.texture_model, "input_type"
        )
        if texture_input_type is not None:
            cfg.texture_input_type = texture_input_type
            cfg.texture_model = add_arg_to_model_name(
                cfg.texture_model, "in_dim", TEXTURE_INPUT_SIZES[texture_input_type]
            )

        if cfg.texture_input_type in ("world", "world_and_view"):
            cfg.texture_model, sample_norm = pop_arg_from_model_name(
                cfg.texture_model, "sample_norm"
            )
            if sample_norm is not None:
                cfg.world_sample_normalisation = sample_norm

    if cfg.base_color_factor is not None:
        if cfg.base_color_factor in FACTOR_SHORTHANDS:
            cfg.base_color_factor = canonical_factor_name(
                FACTOR_SHORTHANDS[cfg.base_color_factor]
            )

    if cfg.sigmoid_factor is not None:
        if cfg.sigmoid_factor in FACTOR_SHORTHANDS:
            cfg.sigmoid_factor = canonical_factor_name(
                FACTOR_SHORTHANDS[cfg.sigmoid_factor]
            )

    if cfg.opac_loss is not None:
        if cfg.opac_loss_fn in LOSS_SHORTHANDS:
            cfg.opac_loss_fn = canonical_loss_name(LOSS_SHORTHANDS[cfg.opac_loss_fn])

    if cfg.tex_opac_loss is not None:
        if cfg.tex_opac_loss_fn in LOSS_SHORTHANDS:
            cfg.tex_opac_loss_fn = canonical_loss_name(
                LOSS_SHORTHANDS[cfg.tex_opac_loss_fn]
            )

    if cfg.steepness_loss is not None:
        if cfg.steepness_loss_fn in LOSS_SHORTHANDS:
            cfg.steepness_loss_fn = canonical_loss_name(
                LOSS_SHORTHANDS[cfg.steepness_loss_fn]
            )

    cfg.texture_resize_points = {}
    if cfg.texture_resize_steps is not None and cfg.texture_resize_values is not None:
        points = list(zip(cfg.texture_resize_values, cfg.texture_resize_values))
        if cfg.texture_resize_heights is not None:
            points = list(zip(cfg.texture_resize_values, cfg.texture_resize_heights))
        assert len(cfg.texture_resize_steps) == len(
            points
        ), "Texture steps and values are not the same length."
        resize_points = zip(cfg.texture_resize_steps, points)
        for key, (width, height) in resize_points:
            if height is None:
                height = width
            cfg.texture_resize_points[key] = width, height
    else:
        assert (
            cfg.texture_resize_steps is None and cfg.texture_resize_values is None
        ), "Incomplete texture resize steps. The textures will not be resized."
