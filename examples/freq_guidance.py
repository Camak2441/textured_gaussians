"""
Frequency-guided scale pre-training (Steps 1–5).

Pipeline:
  Step 1  compute_freq_map_full   — downsampled image, DCT, spectral moment tensor
  Cache   load_or_compute_freq_cache — pack all images into one NPZ per (dataset, settings)
  Step 3  compute_j_unit          — scale-free projection Jacobian  [C, N, 2, 2]
  Step 4  lookup_freq_map         — nearest-neighbour block lookup   [C, N, 2, 2]
          pullback_freq           — pull Σ_ref into 3-D local space  [C, N, 2, 2]
  Step 5  target_covariance_3d    — target Gaussian 3-D covariance   [C, N, 2, 2]
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import Tensor

if TYPE_CHECKING:
    from datasets.colmap import Parser

# Absolute path to data/cache, anchored at the project root (parent of examples/).
_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


# ══════════════════════════════════════════════════════════════════════════════
# Internal DCT implementation (Type-II, via FFT — no extra dependencies)
# ══════════════════════════════════════════════════════════════════════════════


def _dct_1d(x: Tensor) -> Tensor:
    """Type-II DCT along the last dimension."""
    N = x.shape[-1]
    # Reorder: even-indexed samples forward, odd-indexed samples reversed.
    # This converts the DCT into a real-to-complex DFT problem.
    v = torch.cat([x[..., ::2], x[..., 1::2].flip(-1)], dim=-1)
    Vc = torch.fft.fft(v.float())
    k = torch.arange(N, device=x.device, dtype=torch.float32)
    W = torch.exp(-1j * math.pi * k / (2.0 * N))
    return (2.0 * (Vc * W).real).to(x.dtype)


def _dct_2d(x: Tensor) -> Tensor:
    """2-D Type-II DCT along the last two dimensions."""
    # Apply 1-D DCT along last dim, transpose, apply again, transpose back.
    return _dct_1d(_dct_1d(x).transpose(-2, -1)).transpose(-2, -1)


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — compute downsampled image, DCT blocks, and spectral moment tensor
# ══════════════════════════════════════════════════════════════════════════════


def compute_freq_map_full(
    img: Tensor,  # [H, W, C]  float32 in [0, 1]
    downsample: int = 8,
    block_size: int = 16,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Returns three artefacts for a single image:

    downsampled  [H_ds, W_ds, C]             float32   H_ds = H // downsample
    dct          [H_out, W_out, B, B, C]     float32   B = block_size, H_out = H_ds
    covariance   [H_out, W_out, 2, 2]        float32   spectral second-moment tensor

    Block layout
    ────────────
    stride = downsample (one block per downsampled pixel)
    block_size = 2 × downsample  →  50 % overlap between adjacent blocks
    pad = block_size // 2 - stride // 2  =  4  (with default 8/16)

    Each block is centred at the *centre* of the corresponding 8 × 8 downsampling
    cell (i.e. at original pixel (8i + 3.5, 8j + 3.5)), not at its corner.
    With pad = 4 the first block covers original rows [−4, 12) and is centred
    at row 4 ≈ 3.5, matching the first downsampled pixel.

    Windowing
    ─────────
    A separable Hamming window is applied to each block before the DCT to
    suppress spectral leakage from the implicit block-boundary discontinuity.
    With 50 % overlap every original pixel contributes to exactly two blocks,
    approximately equalising each pixel's influence on the frequency estimate.
    """
    H, W, C = img.shape
    stride = downsample  # = 8
    pad = block_size // 2 - stride // 2  # = 4  (centres on cell midpoint)
    H_out = H // stride
    W_out = W // stride

    # ── Level 1: average-pool downsampling ──────────────────────────────────
    img_4d = img.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    downsampled = F.avg_pool2d(img_4d, downsample).squeeze(0).permute(1, 2, 0)
    # [H_ds, W_ds, C]

    # ── Level 2: overlapping windowed blocks ────────────────────────────────
    img_padded = F.pad(img_4d, (pad, pad, pad, pad), mode="reflect")  # [1, C, H+8, W+8]

    # unfold extracts every (block_size × block_size) patch at stride `stride`.
    # Shape after unfold: [1, C, N_h, N_w, block_size, block_size]
    blocks = img_padded.unfold(2, block_size, stride).unfold(3, block_size, stride)
    blocks = blocks[:, :, :H_out, :W_out, :, :]  # exact grid count
    blocks = blocks.squeeze(0).permute(1, 2, 3, 4, 0)  # [H_out, W_out, B, B, C]

    # Separable 2-D Hamming window
    n = torch.arange(block_size, dtype=torch.float32, device=img.device)
    hamming_1d = 0.54 - 0.46 * torch.cos(2.0 * math.pi * n / (block_size - 1))
    hamming_2d = hamming_1d[:, None] * hamming_1d[None, :]  # [B, B]
    blocks = blocks * hamming_2d[None, None, :, :, None]

    # ── Level 2 (cont.): 2-D DCT ────────────────────────────────────────────
    # _dct_2d acts on the last two dimensions, so move B, B to the back.
    x = blocks.permute(0, 1, 4, 2, 3)  # [H_out, W_out, C, B, B]
    dct = _dct_2d(x).permute(0, 1, 3, 4, 2)  # [H_out, W_out, B, B, C]

    # ── Level 3: spectral second-moment tensor ───────────────────────────────
    # Average energy across channels; DC term removed so the tensor captures
    # only the AC (texture) frequency content.
    energy = dct.pow(2).mean(-1)  # [H_out, W_out, B, B]
    energy[..., 0, 0] = 0.0  # remove DC

    u = torch.arange(block_size, dtype=torch.float32, device=img.device).view(
        1, 1, -1, 1
    )
    v = torch.arange(block_size, dtype=torch.float32, device=img.device).view(
        1, 1, 1, -1
    )
    total = energy.sum((-2, -1), keepdim=True) + 1e-8  # normalisation

    Suu = (u**2 * energy).sum((-2, -1)) / total.squeeze((-2, -1))
    Svv = (v**2 * energy).sum((-2, -1)) / total.squeeze((-2, -1))
    Suv = (u * v * energy).sum((-2, -1)) / total.squeeze((-2, -1))

    covariance = torch.stack(
        [torch.stack([Suu, Suv], dim=-1), torch.stack([Suv, Svv], dim=-1)],
        dim=-2,
    )  # [H_out, W_out, 2, 2]

    return downsampled, dct, covariance


# ══════════════════════════════════════════════════════════════════════════════
# Cache management
# ══════════════════════════════════════════════════════════════════════════════


def _cache_path(data_dir: str, downsample: int, block_size: int) -> Path:
    """
    Returns the path to the NPZ cache file for the given dataset + settings.

    The filename encodes a human-readable dataset name and a 12-character
    SHA-256 prefix that uniquely identifies (data_dir, downsample, block_size,
    derived pad).  Changing any parameter produces a different file.
    """
    params = {
        "data_dir": str(Path(data_dir).resolve()),
        "downsample": downsample,
        "block_size": block_size,
        "pad": block_size // 2 - downsample // 2,
    }
    key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
    name = Path(data_dir).name
    return _CACHE_DIR / f"{name}_{key}.npz"


def _cache_valid(cache: "np.lib.npyio.NpzFile", image_keys: list[str]) -> bool:
    """True iff the stored image-key list matches the provided list."""
    return list(cache["image_keys"].astype(str)) == image_keys


def _hsv_to_rgb(h: Tensor, s: Tensor, v: Tensor) -> Tensor:
    """Vectorised HSV → RGB. All inputs in [0, 1], output shape [..., 3] in [0, 1]."""
    h6 = h * 6.0
    hi = h6.long() % 6
    f = h6 - h6.floor()
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    sectors = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
    rgb = torch.zeros(*h.shape, 3, device=h.device, dtype=h.dtype)
    for idx, (r, g, b) in enumerate(sectors):
        mask = (hi == idx).unsqueeze(-1)
        rgb = torch.where(mask, torch.stack([r, g, b], dim=-1), rgb)
    return rgb


def _save_freq_debug_images(
    covariance: Tensor,  # [N, H_ds, W_ds, 2, 2]
    cache_path: Path,
    image_keys: list[str],
) -> None:
    """
    For each image write a PNG using HSV encoding of the spectral covariance:
      H (hue)        = direction of dominant eigenvector, angle in [0, π) → [0, 1]
      S (saturation) = smaller eigenvalue / max(larger eigenvalue), i.e. λ1 / max(λ2)
      V (value)      = larger eigenvalue / max(larger eigenvalue), i.e. λ2 / max(λ2)

    Black pixel → no frequency information (λ2 ≈ 0).
    Grey pixel  → strong single-direction frequency (λ1 ≈ 0), brightness = strength.
    Coloured    → frequency energy in both directions; hue = dominant orientation.
    """
    debug_dir = cache_path.parent / "debug" / cache_path.stem
    debug_dir.mkdir(parents=True, exist_ok=True)

    for i, (cov, key) in enumerate(zip(covariance, image_keys)):
        # eigh returns eigenvalues ascending: index 0 = λ1 (smaller), 1 = λ2 (larger)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov.float())
        lambda1 = eigenvalues[..., 0].clamp(min=0.0)  # [H_ds, W_ds]
        lambda2 = eigenvalues[..., 1].clamp(min=0.0)  # [H_ds, W_ds]
        e2 = eigenvectors[..., :, 1]  # [H_ds, W_ds, 2]

        scale = lambda2.max() + 1e-8
        V = (lambda2 / scale).clamp(0.0, 1.0)
        S = (1 - lambda1 / scale).clamp(0.0, 1.0)

        # Eigenvectors are sign-ambiguous; fold angle into [0, π) then map to [0, 1]
        angle = torch.atan2(e2[..., 1], e2[..., 0])  # [-π, π]
        H = (angle % math.pi) / math.pi  # [0, 1)

        rgb = _hsv_to_rgb(H, S, V)
        img = (rgb.clamp(0.0, 1.0) * 255).byte().cpu().numpy()
        stem = Path(key).stem or f"{i:04d}"
        imageio.imwrite(str(debug_dir / f"{i:04d}_{stem}.png"), img)

    print(f"[freq_guidance] Saved {len(image_keys)} debug images to {debug_dir}")


def _build_and_save_cache(
    cache_path: Path,
    images_float: "list[np.ndarray]",  # [H, W, 3] float32 in [0, 1] each
    image_keys: list[str],
    downsample: int,
    block_size: int,
    device: str,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute freq tensors from a list of float32 images, write NPZ, return tensors."""
    ds_list, dct_list, cov_list = [], [], []
    for img_np in tqdm.tqdm(images_float, desc="freq cache", unit="img"):
        img = torch.from_numpy(img_np)
        ds, dct, cov = compute_freq_map_full(img, downsample, block_size)
        ds_list.append(ds.half())
        dct_list.append(dct.half())
        cov_list.append(cov)

    downsampled = torch.stack(ds_list)
    dct_all = torch.stack(dct_list)
    covariance = torch.stack(cov_list)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        str(tmp),
        downsampled=downsampled.numpy(),
        dct=dct_all.numpy(),
        covariance=covariance.numpy(),
        image_keys=np.array(image_keys),
    )
    tmp.rename(cache_path)
    print(f"[freq_guidance] Saved cache to {cache_path}")

    _save_freq_debug_images(covariance, cache_path, image_keys)

    return (
        downsampled.float().to(device),
        dct_all.float().to(device),
        covariance.to(device),
    )


def _load_cache(cache_path: Path, device: str) -> tuple[Tensor, Tensor, Tensor]:
    cache = np.load(str(cache_path), allow_pickle=False)
    return (
        torch.from_numpy(cache["downsampled"].astype(np.float32)).to(device),
        torch.from_numpy(cache["dct"].astype(np.float32)).to(device),
        torch.from_numpy(cache["covariance"]).to(device),
    )


def load_or_compute_freq_cache(
    parser: "Parser",
    downsample: int = 8,
    block_size: int = 16,
    device: str = "cpu",
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Returns (downsampled, dct, covariance) for *all* images in the parser,
    indexed by parser image index.

    downsampled  [N, H_ds, W_ds, C]             float32
    dct          [N, H_ds, W_ds, B, B, C]       float32   B = block_size
    covariance   [N, H_ds, W_ds, 2, 2]          float32
    """
    cache_path = _cache_path(parser.data_dir, downsample, block_size)
    image_keys = [str(Path(p).resolve()) for p in parser.image_paths]

    if cache_path.exists():
        cache = np.load(str(cache_path), allow_pickle=False)
        if _cache_valid(cache, image_keys):
            print(f"[freq_guidance] Loaded cache from {cache_path}")
            return _load_cache(cache_path, device)
        print(f"[freq_guidance] Cache at {cache_path} is stale — recomputing.")

    print(f"[freq_guidance] Building frequency cache for '{parser.data_dir}' ...")
    images_float = [
        imageio.imread(p)[..., :3].astype(np.float32) / 255.0
        for p in parser.image_paths
    ]
    return _build_and_save_cache(
        cache_path, images_float, image_keys, downsample, block_size, device
    )


def load_or_compute_freq_cache_from_images(
    images: "list[np.ndarray]",  # [H, W, 3] uint8 per image
    data_dir: str,
    image_ids: "list[str]",
    downsample: int = 8,
    block_size: int = 16,
    device: str = "cpu",
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Variant of load_or_compute_freq_cache for datasets whose images are
    already loaded into memory (e.g. BlenderDataset).

    images     list of [H, W, 3] uint8 numpy arrays
    image_ids  list of string identifiers used for cache validation
    """
    cache_path = _cache_path(data_dir, downsample, block_size)

    if cache_path.exists():
        cache = np.load(str(cache_path), allow_pickle=False)
        if _cache_valid(cache, image_ids):
            print(f"[freq_guidance] Loaded cache from {cache_path}")
            return _load_cache(cache_path, device)
        print(f"[freq_guidance] Cache at {cache_path} is stale — recomputing.")

    print(f"[freq_guidance] Building frequency cache for '{data_dir}' ...")
    images_float = [img[..., :3].astype(np.float32) / 255.0 for img in images]
    return _build_and_save_cache(
        cache_path, images_float, image_ids, downsample, block_size, device
    )


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — scale-free projection Jacobian
# ══════════════════════════════════════════════════════════════════════════════


def compute_j_unit(
    quats: Tensor,  # [N, 4]
    means: Tensor,  # [N, 3]
    viewmats: Tensor,  # [C, 4, 4]  world-to-cam
    Ks: Tensor,  # [C, 3, 3]
    detach_quats: bool = True,
) -> Tensor:  # [C, N, 2, 2]
    """
    Compute J_unit: the Jacobian that maps *unit* displacements in the
    Gaussian's 3-D local tangent frame (along R[:,0] and R[:,1]) to
    screen-pixel displacements.

        J_unit[c, n] = (1 / depth[c, n]) * K[:2, :2] * R_cw[:2, :] * R_world[:, :2]

    When detach_quats=True (default), quaternions are detached so gradients
    flow only to scales.  Set detach_quats=False for the orientation loss,
    which needs gradients to flow through J_unit to quats.
    """
    from textured_gaussians.cuda._torch_impl import _quat_to_rotmat

    R_world = _quat_to_rotmat(
        F.normalize(quats.detach() if detach_quats else quats, dim=-1)
    )  # [N, 3, 3]

    R_cw = viewmats[:, :3, :3]  # [C, 3, 3]
    t_cw = viewmats[:, :3, 3]  # [C, 3]

    # Camera-frame depth for each Gaussian × camera pair.
    means_c = torch.einsum("cij,nj->cni", R_cw, means) + t_cw[:, None, :]  # [C, N, 3]
    depths = means_c[:, :, 2].clamp(min=1e-4)  # [C, N]

    # Project the first two (tangent) columns of R_world into camera frame.
    # tangents_world: [N, 3, 2]  — columns are R_world[:,0] and R_world[:,1]
    tangents_cam = torch.einsum(
        "cij,njk->cnik", R_cw, R_world[:, :, :2]
    )  # [C, N, 3, 2]

    # Apply the 2 × 2 focal-length block of K (rows x, y; columns x, y of K).
    # J_unit[c,n] = K_2[c] @ tangents_cam[c,n,:2,:] / depth[c,n]
    K_2 = Ks[:, :2, :2]  # [C, 2, 2]
    J_unit = (
        torch.einsum("cij,cnjk->cnik", K_2, tangents_cam[:, :, :2, :])
        / depths[:, :, None, None]
    )  # [C, N, 2, 2]
    return J_unit


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — look up and pull back reference frequency into 3-D local space
# ══════════════════════════════════════════════════════════════════════════════


def lookup_freq_map(
    covariance_all: Tensor,  # [N_images, H_map, W_map, 2, 2]
    means2d: Tensor,  # [C, N, 2]  screen-pixel (x, y) — should be detached
    parser_ids: Tensor,  # [C]  global parser image index for each camera
) -> Tensor:  # [C, N, 2, 2]
    """
    Nearest-neighbour lookup of the pre-computed spectral moment tensor.

    The block at output position (i, j) is centred at original pixel
    (8j + 3.5, 8i + 3.5), so the block boundary falls between pixels.
    Integer division by 8 maps each pixel to the block whose centre is
    closest, matching the stride used in compute_freq_map_full.

    Out-of-bounds pixel positions are clamped to the nearest valid block.
    """
    H_map = covariance_all.shape[1]
    W_map = covariance_all.shape[2]

    px = means2d[:, :, 0]  # [C, N]
    py = means2d[:, :, 1]  # [C, N]

    bj = px.long().div(8, rounding_mode="floor").clamp(0, W_map - 1)  # [C, N]
    bi = py.long().div(8, rounding_mode="floor").clamp(0, H_map - 1)  # [C, N]

    C, N = means2d.shape[:2]
    out = torch.empty(
        C, N, 2, 2, device=covariance_all.device, dtype=covariance_all.dtype
    )
    for c in range(C):
        img_cov = covariance_all[parser_ids[c]]  # [H_map, W_map, 2, 2]
        out[c] = img_cov[bi[c], bj[c]]  # [N, 2, 2]
    return out


def pullback_freq(
    sigma_ref_screen: Tensor,  # [C, N, 2, 2]
    J_unit: Tensor,  # [C, N, 2, 2]
) -> Tensor:  # [C, N, 2, 2]
    """
    Pull the screen-space spectral moment tensor into the Gaussian's 3-D
    local tangent frame via the covariant frequency transformation:

        Σ_local = J_unit^T  Σ_screen  J_unit

    This accounts for foreshortening, focal-length scaling, and the
    Gaussian's 3-D orientation without requiring the Gaussian to be face-on.
    The result is expressed in the coordinate system defined by the
    Gaussian's first two rotation axes (R[:,0], R[:,1]).
    """
    Jt = J_unit.transpose(-1, -2)
    return Jt @ sigma_ref_screen @ J_unit


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — target 3-D Gaussian covariance
# ══════════════════════════════════════════════════════════════════════════════


def target_covariance_3d(
    sigma_ref_local: Tensor,  # [C, N, 2, 2]
    texture_size: int,
    f_target: float = 0.25,
    eps: float = 1e-6,
) -> Tensor:  # [C, N, 2, 2]
    """
    Compute the target Gaussian 3-D covariance matrix (in local tangent space)
    such that the texture will represent the reference frequency.

    Derivation
    ──────────
    For a Gaussian with local scales (s_x, s_y) and a T × T texture:
      • UV coordinate u = 3D_local_a / s_x
      • texture Nyquist in UV space: T/2 cycles per UV unit
      • desired operating point: f_target · T  cycles per UV unit

    Setting Σ_UV = diag(s_x, s_y) · Σ_local · diag(s_x, s_y) = (f_target · T)² · I
    and solving gives:

        Σ_target = (f_target · T)² · Σ_local⁻¹

    High reference frequency (large Σ_local eigenvalue)
      → small target σ  (tight Gaussian, texture detail preserved)
    Large texture T
      → larger target σ  (more texels can represent the same frequency on a
        bigger Gaussian, so the Gaussian does not need to be as small)
    """
    I = torch.eye(2, device=sigma_ref_local.device, dtype=sigma_ref_local.dtype)
    reg = sigma_ref_local + eps * I  # numerical stability
    return float((f_target * texture_size) ** 2) * torch.linalg.inv(reg)


# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — per-Gaussian per-pixel frequency loss (pullback in CUDA kernel)
# ══════════════════════════════════════════════════════════════════════════════


def _get_C() -> object:
    from textured_gaussians.cuda._backend import _C

    return _C


class _FreqAccumulate(torch.autograd.Function):
    """
    Differentiable per-Gaussian per-pixel frequency loss (eigenvector wavelength approach).

    For each pixel-Gaussian intersection the kernel:
      1. Eigendecomposes sigma_ref_screen at pixel (i,j) → (λ₁,e₁), (λ₂,e₂)
      2. Maps each eigenvector to UV space: e_uv = J_full⁻¹ @ e  (J_full = J_unit @ diag(s))
      3. Loss per eigenvector: (||e_uv||² * block_T_sq / λ  −  1)²
         Target condition: one screen wavelength = one texel in UV

    Forward:  returns freq_loss [C, N].
    Backward: propagates gradients only to scales (J_unit is detached).
    """

    @staticmethod
    def forward(
        ctx,
        means2d,  # [C, N, 2]    detached
        ray_transforms,  # [C, N, 9]    detached
        opacities,  # [C*N]        detached
        J_unit,  # [C*N, 4]     detached — no rotation gradients
        scales,  # [N, 2]       requires_grad
        freq_map,  # [C, H, W, 3] detached — (Suu, Suv, Svv) spectral covariance
        block_T_sq: float,
        image_width: int,
        image_height: int,
        freq_downsample: int,
        tile_size: int,
        tile_offsets,  # [C, th, tw]
        flatten_ids,  # [n_isects]
    ):
        C, N = means2d.shape[0], means2d.shape[1]

        freq_loss, render_alphas, last_ids = _get_C().freq_accumulate_fwd(
            means2d.contiguous(),
            ray_transforms.reshape(-1, 9).contiguous(),
            opacities.reshape(-1).contiguous(),
            J_unit.contiguous(),
            scales.contiguous(),
            freq_map.contiguous(),
            block_T_sq,
            image_width,
            image_height,
            freq_downsample,
            tile_size,
            tile_offsets.contiguous(),
            flatten_ids.contiguous(),
        )

        ctx.save_for_backward(
            means2d,
            ray_transforms,
            opacities,
            J_unit,
            scales,
            freq_map,
            render_alphas,
            last_ids,
            tile_offsets,
            flatten_ids,
        )
        ctx.block_T_sq = block_T_sq
        ctx.image_width = image_width
        ctx.image_height = image_height
        ctx.freq_downsample = freq_downsample
        ctx.tile_size = tile_size
        ctx.C, ctx.N = C, N

        return freq_loss.reshape(C, N)

    @staticmethod
    def backward(ctx, v_freq_loss):
        (
            means2d,
            ray_transforms,
            opacities,
            J_unit,
            scales,
            freq_map,
            render_alphas,
            last_ids,
            tile_offsets,
            flatten_ids,
        ) = ctx.saved_tensors

        v_scales = _get_C().freq_accumulate_bwd(
            means2d.contiguous(),
            ray_transforms.reshape(-1, 9).contiguous(),
            opacities.reshape(-1).contiguous(),
            J_unit.contiguous(),
            scales.contiguous(),
            freq_map.contiguous(),
            ctx.block_T_sq,
            ctx.image_width,
            ctx.image_height,
            ctx.freq_downsample,
            ctx.tile_size,
            tile_offsets.contiguous(),
            flatten_ids.contiguous(),
            render_alphas,
            last_ids,
            v_freq_loss.reshape(-1).contiguous(),
        )  # [N, 2]

        # Return gradients for: means2d, ray_transforms, opacities, J_unit, scales,
        #   freq_map, block_T_sq, image_width, image_height, freq_downsample,
        #   tile_size, tile_offsets, flatten_ids
        return (
            None,
            None,
            None,
            None,
            v_scales,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def compute_freq_loss(
    scales: Tensor,  # [N, 3]      Gaussian scales (first two used)
    quats: Tensor,  # [N, 4]      Gaussian quaternions (detached in J_unit)
    means: Tensor,  # [N, 3]      Gaussian means
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]   at downsampled resolution
    means2d: Tensor,  # [C, N, 2]   from rendering forward pass
    ray_transforms: Tensor,  # [C, N, 3, 3] from rendering forward pass
    opacities: Tensor,  # [C, N]      from rendering forward pass
    tile_offsets: Tensor,  # [C, th, tw]
    flatten_ids: Tensor,  # [n_isects]
    freq_covariance: Tensor,  # [N_imgs, H, W, 2, 2]
    camera_indices: Tensor,  # [C]  parser image index for each camera
    image_width: int,
    image_height: int,
    tile_size: int,
    texture_size: int,
    block_size: int,
    downsample: int,
    f_target: float = 0.25,
    freq_downsample: int = 1,
) -> Tensor:  # scalar loss
    """
    Per-Gaussian per-pixel frequency loss via eigenvector wavelength matching.

    For each pixel-Gaussian intersection the CUDA kernel eigendecomposes the
    pixel's spectral covariance, converts each eigenvector to the Gaussian's UV
    space via J_full⁻¹ = diag(1/s) @ J_unit⁻¹, and penalises the log-ratio
    between the UV-space texel size and the reference screen-space wavelength:

        loss_i = log(||e_uv_i||² * block_T_sq / λ_i)²

    where  block_T_sq = (block_size_screen * f_target * texture_size / 6)²
           block_size_screen = block_size / downsample  (DCT block in screen px)

    Log-space residual keeps the loss bounded at initialisation when scales are
    far from the target, avoiding the ∝ 1/s⁴ blow-up of the linear residual.

    f_target controls cycles per texel: f_target=1 means one wavelength per
    texel; f_target=0.5 means two texels per wavelength (Nyquist-limited).
    Smaller f_target allows the Gaussian to be larger relative to the wavelength.

    Gradients flow only to `scales`; quaternions are detached inside
    compute_j_unit, preventing the loss from driving Gaussians face-on.
    """
    # Select per-camera frequency maps and pack as (Suu, Suv, Svv)
    cov_C = freq_covariance[camera_indices]  # [C, H, W, 2, 2]
    freq_map = torch.stack(
        [cov_C[..., 0, 0], cov_C[..., 0, 1], cov_C[..., 1, 1]], dim=-1
    )  # [C, H, W, 3]

    # Scale-free Jacobian with detached quaternions → no rotation gradients
    J = compute_j_unit(quats, means, viewmats, Ks)  # [C, N, 2, 2]
    C, N = J.shape[:2]
    J_flat = J.reshape(C * N, 4)  # [C*N, 4]

    # Sanity-check tensor shapes before launching CUDA kernel.
    assert freq_map.shape[1] == image_height // freq_downsample, (
        f"freq_map height {freq_map.shape[1]} != image_height // freq_downsample "
        f"({image_height} // {freq_downsample} = {image_height // freq_downsample})"
    )
    assert freq_map.shape[2] == image_width // freq_downsample, (
        f"freq_map width {freq_map.shape[2]} != image_width // freq_downsample "
        f"({image_width} // {freq_downsample} = {image_width // freq_downsample})"
    )
    assert scales.shape[0] == N, (
        f"scales N={scales.shape[0]} != J N={N}"
    )
    assert means2d.shape[0] == C and means2d.shape[1] == N, (
        f"means2d {tuple(means2d.shape)} != ({C},{N},2)"
    )
    if flatten_ids.numel() > 0:
        max_id = int(flatten_ids.max())
        assert max_id < C * N, (
            f"flatten_ids max={max_id} >= C*N={C*N}"
        )

    # block_size_screen: DCT block in downsampled (screen) pixels
    # UV space is [-3, 3], so one texel spans 6/T UV units — divide by 6
    block_T_sq = float((block_size * f_target * texture_size / (downsample * 6)) ** 2)

    freq_loss = _FreqAccumulate.apply(
        means2d.detach(),
        ray_transforms.detach(),
        opacities.detach(),
        J_flat.detach(),  # no grad through J → no rotation update
        scales[:, :2].contiguous(),  # grad flows here → scales
        freq_map.detach(),
        block_T_sq,
        image_width,
        image_height,
        freq_downsample,
        tile_size,
        tile_offsets.detach(),
        flatten_ids.detach(),
    )  # [C, N]

    return freq_loss.sum() / image_width / image_height


# ══════════════════════════════════════════════════════════════════════════════
# Step 7 — per-Gaussian orientation loss (UV wavelength vector alignment)
# ══════════════════════════════════════════════════════════════════════════════


class _FreqOrient(torch.autograd.Function):
    """
    Differentiable per-Gaussian orientation loss using UV-space wavelength vectors.

    For each pixel-Gaussian intersection:
      1. Eigendecompose sigma_ref → dominant eigenvector e2, eigenvalue lambda2
      2. Compute w_screen = J_unit @ diag(s) @ freq_vec  (covariant transform)
      3. Reference: w_ref = e2 / sqrt(lambda2)           (screen-space wavelength)
      4. Loss: lambda2 * ||w_screen - w_ref||²            (dimensionless)

    Gradients flow to:
      - freq_vec: adjusts the UV-space wavelength direction/magnitude
      - J_flat:   flows back through compute_j_unit to quats (rotation)
    Scales are treated as constants (supervised by the scale loss).
    """

    @staticmethod
    def forward(
        ctx,
        means2d,  # [C, N, 2]   detached
        ray_transforms,  # [C, N, 3, 3] detached
        opacities,  # [C*N]       detached
        J_flat,  # [C*N, 4]    requires_grad (quats path)
        scales,  # [N, 2]      detached
        freq_vec,  # [N, 2]      requires_grad
        freq_map,  # [C, H, W, 3] detached
        image_width: int,
        image_height: int,
        freq_downsample: int,
        tile_size: int,
        tile_offsets,  # [C, th, tw]
        flatten_ids,  # [n_isects]
    ):
        C, N = means2d.shape[0], means2d.shape[1]

        orient_loss, render_alphas, last_ids = _get_C().freq_orient_fwd(
            means2d.contiguous(),
            ray_transforms.reshape(-1, 9).contiguous(),
            opacities.reshape(-1).contiguous(),
            J_flat.contiguous(),
            scales.contiguous(),
            freq_vec.contiguous(),
            freq_map.contiguous(),
            image_width,
            image_height,
            freq_downsample,
            tile_size,
            tile_offsets.contiguous(),
            flatten_ids.contiguous(),
        )

        ctx.save_for_backward(
            means2d,
            ray_transforms,
            opacities,
            J_flat,
            scales,
            freq_vec,
            freq_map,
            render_alphas,
            last_ids,
            tile_offsets,
            flatten_ids,
        )
        ctx.image_width = image_width
        ctx.image_height = image_height
        ctx.freq_downsample = freq_downsample
        ctx.tile_size = tile_size
        ctx.C, ctx.N = C, N

        return orient_loss.reshape(C, N)

    @staticmethod
    def backward(ctx, v_orient_loss):
        (
            means2d,
            ray_transforms,
            opacities,
            J_flat,
            scales,
            freq_vec,
            freq_map,
            render_alphas,
            last_ids,
            tile_offsets,
            flatten_ids,
        ) = ctx.saved_tensors

        v_freq_vec, v_J_flat = _get_C().freq_orient_bwd(
            means2d.contiguous(),
            ray_transforms.reshape(-1, 9).contiguous(),
            opacities.reshape(-1).contiguous(),
            J_flat.contiguous(),
            scales.contiguous(),
            freq_vec.contiguous(),
            freq_map.contiguous(),
            ctx.image_width,
            ctx.image_height,
            ctx.freq_downsample,
            ctx.tile_size,
            tile_offsets.contiguous(),
            flatten_ids.contiguous(),
            render_alphas,
            last_ids,
            v_orient_loss.reshape(-1).contiguous(),
        )  # v_freq_vec [N,2], v_J_flat [C*N,4]

        # inputs: means2d, ray_transforms, opacities, J_flat, scales, freq_vec,
        #         freq_map, image_width, image_height, freq_downsample,
        #         tile_size, tile_offsets, flatten_ids
        return (
            None,
            None,
            None,
            v_J_flat,
            None,
            v_freq_vec,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def compute_freq_orient_loss(
    freq_vec: Tensor,  # [N, 2]   requires_grad — UV wavelength vector
    scales: Tensor,  # [N, 3]   detached (supervised by scale loss)
    quats: Tensor,  # [N, 4]   requires_grad
    means: Tensor,  # [N, 3]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    means2d: Tensor,  # [C, N, 2]
    ray_transforms: Tensor,  # [C, N, 3, 3]
    opacities: Tensor,  # [C, N]
    tile_offsets: Tensor,
    flatten_ids: Tensor,
    freq_covariance: Tensor,  # [N_imgs, H, W, 2, 2]
    camera_indices: Tensor,  # [C]
    image_width: int,
    image_height: int,
    tile_size: int,
    freq_downsample: int = 1,
) -> Tensor:
    """
    Per-Gaussian orientation loss via UV-space wavelength vector alignment.

    For each pixel-Gaussian intersection the CUDA kernel finds the dominant
    screen-space frequency direction (principal eigenvector of sigma_ref) and
    penalises the mismatch between it and the Gaussian's projected wavelength:

        loss = lambda2 * ||J_unit @ diag(s) @ freq_vec  -  e2/sqrt(lambda2)||²

    Gradients flow to freq_vec (UV wavelength direction/magnitude) and to
    quats via J_unit (rotation alignment). Scales are detached.
    """
    cov_C = freq_covariance[camera_indices]  # [C, H, W, 2, 2]
    freq_map = torch.stack(
        [cov_C[..., 0, 0], cov_C[..., 0, 1], cov_C[..., 1, 1]], dim=-1
    )  # [C, H, W, 3]

    # J_unit with quats NOT detached so rotation gradient can flow
    J = compute_j_unit(quats, means, viewmats, Ks, detach_quats=False)
    C, N = J.shape[:2]
    J_flat = J.reshape(C * N, 4)

    # Sanity-check tensor shapes before launching CUDA kernel.
    assert freq_map.shape[1] == image_height // freq_downsample, (
        f"orient freq_map height {freq_map.shape[1]} != image_height // freq_downsample "
        f"({image_height} // {freq_downsample} = {image_height // freq_downsample})"
    )
    assert freq_map.shape[2] == image_width // freq_downsample, (
        f"orient freq_map width {freq_map.shape[2]} != image_width // freq_downsample "
        f"({image_width} // {freq_downsample} = {image_width // freq_downsample})"
    )
    assert scales.shape[0] == N, (
        f"orient scales N={scales.shape[0]} != J N={N}"
    )
    assert freq_vec.shape[0] == N, (
        f"orient freq_vec N={freq_vec.shape[0]} != J N={N}"
    )
    if flatten_ids.numel() > 0:
        max_id = int(flatten_ids.max())
        assert max_id < C * N, (
            f"orient flatten_ids max={max_id} >= C*N={C*N}"
        )

    orient_loss = _FreqOrient.apply(
        means2d.detach(),
        ray_transforms.detach(),
        opacities.detach(),
        J_flat,  # grad flows → quats
        scales[:, :2].detach().contiguous(),
        freq_vec.contiguous(),  # grad flows → freq_vec
        freq_map.detach(),
        image_width,
        image_height,
        freq_downsample,
        tile_size,
        tile_offsets.detach(),
        flatten_ids.detach(),
    )  # [C, N]

    return orient_loss.sum() / image_width / image_height
