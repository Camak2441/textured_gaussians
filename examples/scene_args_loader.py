import os
from typing import Optional

from examples.texture_models import canonical_model_name
from utils import get_file_with_max_int

MODEL_SHORTHANDS = {
    "mix1": """FourierSelector(
            num_freqs=15,hidden_dims=[24, 24],
            sub_models=[
                "SIREN(in_dim=3,hidden_dims=[48,48],out_dim=4,omega_0=20,hidden_omegas=1.00)",
                "SIREN(in_dim=3,hidden_dims=[48,48],out_dim=4,omega_0=30,hidden_omegas=1.25)",
                "SIREN(in_dim=3,hidden_dims=[48,48],out_dim=4,omega_0=40,hidden_omegas=1.50)",
                "SIREN(in_dim=3,hidden_dims=[48,48],out_dim=4,omega_0=50,hidden_omegas=1.75)",
                "SIREN(in_dim=3,hidden_dims=[48,48],out_dim=4,omega_0=60,hidden_omegas=2.00)",
                "FourierMLP(in_dim=3,hidden_dims=[48,48],out_dim=4,num_frequencies=128,sigma=[1000,1,1])",
                "FourierMLP(in_dim=3,hidden_dims=[48,48],out_dim=4,num_frequencies=128,sigma=[1000,2,2])",
                "FourierMLP(in_dim=3,hidden_dims=[48,48],out_dim=4,num_frequencies=128,sigma=[1000,3,3])"
            ]
        )""",
    "fourier1": """FourierMLP(
        in_dim=3,hidden_dims=[128,128,128],out_dim=4,num_frequencies=128,sigma=[1000,3,3]
    )""",
}

NERF_SYNTHETIC = {
    "data_dir": "nerf_synthetic/{path_name}",
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


def make_args(base: dict[str, any], name: str, path_name: Optional[str] = None):
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
}


def process_config(cfg):

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

    # Configure information based on the selected scene
    if cfg.scene is not None:
        assert cfg.scene in SCENES
        scene_args = SCENES[cfg.scene]
        cfg.data_dir = "../data/" + scene_args["data_dir"]

        match cfg.model_type:
            case "tgs":
                args = []

                if cfg.texture_width != 64 or cfg.texture_height != 64:
                    if cfg.texture_width == cfg.texture_height:
                        args.append(f"t{cfg.texture_width}")
                    else:
                        args.append(f"t{cfg.texture_width}x{cfg.texture_height}")
                args_suffix = "_".join(args)
                if len(args_suffix) != 0 and args_suffix[0] != "_":
                    args_suffix = "_" + args_suffix
                match cfg.filtering:
                    case "bilinear":
                        cfg.result_dir = (
                            f"../results/tgs{args_suffix}/{scene_args["result_dir"]}"
                        )
                    case "mipmapped":
                        cfg.result_dir = f"../results/mip_tgs{args_suffix}/{scene_args["result_dir"]}"
                    case "mipmapped2":
                        cfg.result_dir = f"../results/mip2_tgs{args_suffix}/{scene_args["result_dir"]}"
                    case "anisotropic":
                        cfg.result_dir = f"../results/aniso_tgs{args_suffix}/{scene_args["result_dir"]}"
            case "itgs":
                cfg.result_dir = (
                    f"../results/itgs_{cfg.texture_model}/{scene_args["result_dir"]}"
                )
            case _:
                cfg.result_dir = (
                    f"../results/{cfg.model_type}/{scene_args["result_dir"]}"
                )

        cfg.dataset_type = scene_args["dataset_type"]

        if cfg.alpha_loss and not scene_args["alpha"]:
            print("Dataset does not have alpha channels. Disabling alpha loss.")
            cfg.alpha_loss = False

        if cfg.pretrained_path is None:
            pretrained_dir = f"../results/{scene_args["pretrained_dir"]}"
            _, ckpt_path = get_file_with_max_int(pretrained_dir, "ckpt_", ".pt")
            if ckpt_path is None:
                print(f"Unable to find pretrained data in {pretrained_dir}.")
                cfg.pretrained_path = None
            else:
                cfg.pretrained_path = f"{pretrained_dir}/{ckpt_path}"

    if cfg.resume:
        ckpt_dir = cfg.result_dir + "/ckpts"
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
            cfg.texture_model = canonical_model_name(
                MODEL_SHORTHANDS[cfg.texture_model]
            )
