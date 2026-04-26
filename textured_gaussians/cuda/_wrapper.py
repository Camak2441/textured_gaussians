from typing import Callable, Optional, Tuple, Any
import warnings
from typing_extensions import Literal

import torch
from torch import Tensor

from textured_gaussians.utils import (
    Filtering,
    TextureGrads,
    TextureInputType,
    TEXTURE_INPUT_SIZES,
)


def _make_lazy_cuda_func(name: str) -> Callable:
    def call_cuda(*args, **kwargs):
        # pylint: disable=import-outside-toplevel
        from ._backend import _C

        return getattr(_C, name)(*args, **kwargs)

    return call_cuda


def selective_adam_update(
    param: Tensor,
    param_grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    tiles_touched: Tensor,
    lr: float,
    b1: float,
    b2: float,
    eps: float,
    N: int,
    M: int,
) -> None:
    _make_lazy_cuda_func("selective_adam_update")(
        param, param_grad, exp_avg, exp_avg_sq, tiles_touched, lr, b1, b2, eps, N, M
    )


def _make_lazy_cuda_obj(name: str) -> Any:
    # pylint: disable=import-outside-toplevel
    from ._backend import _C

    obj = _C
    for name_split in name.split("."):
        obj = getattr(_C, name_split)
    return obj


def spherical_harmonics(
    degrees_to_use: int,
    dirs: Tensor,  # [..., 3]
    coeffs: Tensor,  # [..., K, 3]
    masks: Optional[Tensor] = None,
) -> Tensor:
    """Computes spherical harmonics.

    Args:
        degrees_to_use: The degree to be used.
        dirs: Directions. [..., 3]
        coeffs: Coefficients. [..., K, 3]
        masks: Optional boolen masks to skip some computation. [...,] Default: None.

    Returns:
        Spherical harmonics. [..., 3]
    """
    assert (degrees_to_use + 1) ** 2 <= coeffs.shape[-2], coeffs.shape
    assert dirs.shape[:-1] == coeffs.shape[:-2], (dirs.shape, coeffs.shape)
    assert dirs.shape[-1] == 3, dirs.shape
    assert coeffs.shape[-1] == 3, coeffs.shape
    if masks is not None:
        assert masks.shape == dirs.shape[:-1], masks.shape
        masks = masks.contiguous()
    return _SphericalHarmonics.apply(
        degrees_to_use, dirs.contiguous(), coeffs.contiguous(), masks
    )


def quat_scale_to_covar_preci(
    quats: Tensor,  # [N, 4],
    scales: Tensor,  # [N, 3],
    compute_covar: bool = True,
    compute_preci: bool = True,
    triu: bool = False,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """Converts quaternions and scales to covariance and precision matrices.

    Args:
        quats: Quaternions (No need to be normalized). [N, 4]
        scales: Scales. [N, 3]
        compute_covar: Whether to compute covariance matrices. Default: True. If False,
            the returned covariance matrices will be None.
        compute_preci: Whether to compute precision matrices. Default: True. If False,
            the returned precision matrices will be None.
        triu: If True, the return matrices will be upper triangular. Default: False.

    Returns:
        A tuple:

        - **Covariance matrices**. If `triu` is True the returned shape is [N, 6], otherwise [N, 3, 3].
        - **Precision matrices**. If `triu` is True the returned shape is [N, 6], otherwise [N, 3, 3].
    """
    assert quats.dim() == 2 and quats.size(1) == 4, quats.size()
    assert scales.dim() == 2 and scales.size(1) == 3, scales.size()
    quats = quats.contiguous()
    scales = scales.contiguous()
    covars, precis = _QuatScaleToCovarPreci.apply(
        quats, scales, compute_covar, compute_preci, triu
    )
    return covars if compute_covar else None, precis if compute_preci else None


def persp_proj(
    means: Tensor,  # [C, N, 3]
    covars: Tensor,  # [C, N, 3, 3]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
) -> Tuple[Tensor, Tensor]:
    """Perspective projection on Gaussians.
    DEPRECATED: please use `proj` with `ortho=False` instead.

    Args:
        means: Gaussian means. [C, N, 3]
        covars: Gaussian covariances. [C, N, 3, 3]
        Ks: Camera intrinsics. [C, 3, 3]
        width: Image width.
        height: Image height.

    Returns:
        A tuple:

        - **Projected means**. [C, N, 2]
        - **Projected covariances**. [C, N, 2, 2]
    """
    warnings.warn(
        "persp_proj is deprecated and will be removed in a future release. "
        "Use proj with ortho=False instead.",
        DeprecationWarning,
    )
    return proj(means, covars, Ks, width, height, ortho=False)


def proj(
    means: Tensor,  # [C, N, 3]
    covars: Tensor,  # [C, N, 3, 3]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
) -> Tuple[Tensor, Tensor]:
    """Projection of Gaussians (perspective or orthographic).

    Args:
        means: Gaussian means. [C, N, 3]
        covars: Gaussian covariances. [C, N, 3, 3]
        Ks: Camera intrinsics. [C, 3, 3]
        width: Image width.
        height: Image height.

    Returns:
        A tuple:

        - **Projected means**. [C, N, 2]
        - **Projected covariances**. [C, N, 2, 2]
    """
    C, N, _ = means.shape
    assert means.shape == (C, N, 3), means.size()
    assert covars.shape == (C, N, 3, 3), covars.size()
    assert Ks.shape == (C, 3, 3), Ks.size()
    means = means.contiguous()
    covars = covars.contiguous()
    Ks = Ks.contiguous()
    return _Proj.apply(means, covars, Ks, width, height, camera_model)


def world_to_cam(
    means: Tensor,  # [N, 3]
    covars: Tensor,  # [N, 3, 3]
    viewmats: Tensor,  # [C, 4, 4]
) -> Tuple[Tensor, Tensor]:
    """Transforms Gaussians from world to camera coordinate system.

    Args:
        means: Gaussian means. [N, 3]
        covars: Gaussian covariances. [N, 3, 3]
        viewmats: World-to-camera transformation matrices. [C, 4, 4]

    Returns:
        A tuple:

        - **Gaussian means in camera coordinate system**. [C, N, 3]
        - **Gaussian covariances in camera coordinate system**. [C, N, 3, 3]
    """
    C = viewmats.size(0)
    N = means.size(0)
    assert means.size() == (N, 3), means.size()
    assert covars.size() == (N, 3, 3), covars.size()
    assert viewmats.size() == (C, 4, 4), viewmats.size()
    means = means.contiguous()
    covars = covars.contiguous()
    viewmats = viewmats.contiguous()
    return _WorldToCam.apply(means, covars, viewmats)


def fully_fused_projection(
    means: Tensor,  # [N, 3]
    covars: Optional[Tensor],  # [N, 6] or None
    quats: Optional[Tensor],  # [N, 4] or None
    scales: Optional[Tensor],  # [N, 3] or None
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    packed: bool = False,
    sparse_grad: bool = False,
    calc_compensations: bool = False,
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Projects Gaussians to 2D.

    This function fuse the process of computing covariances
    (:func:`quat_scale_to_covar_preci()`), transforming to camera space (:func:`world_to_cam()`),
    and projection (:func:`proj()`).

    .. note::

        During projection, we ignore the Gaussians that are outside of the camera frustum.
        So not all the elements in the output tensors are valid. The output `radii` could serve as
        an indicator, in which zero radii means the corresponding elements are invalid in
        the output tensors and will be ignored in the next rasterization process. If `packed=True`,
        the output tensors will be packed into a flattened tensor, in which all elements are valid.
        In this case, a `camera_ids` tensor and `gaussian_ids` tensor will be returned to indicate the
        row (camera) and column (Gaussian) indices of the packed flattened tensor, which is essentially
        following the COO sparse tensor format.

    .. note::

        This functions supports projecting Gaussians with either covariances or {quaternions, scales},
        which will be converted to covariances internally in a fused CUDA kernel. Either `covars` or
        {`quats`, `scales`} should be provided.

    Args:
        means: Gaussian means. [N, 3]
        covars: Gaussian covariances (flattened upper triangle). [N, 6] Optional.
        quats: Quaternions (No need to be normalized). [N, 4] Optional.
        scales: Scales. [N, 3] Optional.
        viewmats: Camera-to-world matrices. [C, 4, 4]
        Ks: Camera intrinsics. [C, 3, 3]
        width: Image width.
        height: Image height.
        eps2d: A epsilon added to the 2D covariance for numerical stability. Default: 0.3.
        near_plane: Near plane distance. Default: 0.01.
        far_plane: Far plane distance. Default: 1e10.
        radius_clip: Gaussians with projected radii smaller than this value will be ignored. Default: 0.0.
        packed: If True, the output tensors will be packed into a flattened tensor. Default: False.
        sparse_grad: This is only effective when `packed` is True. If True, during backward the gradients
          of {`means`, `covars`, `quats`, `scales`} will be a sparse Tensor in COO layout. Default: False.
        calc_compensations: If True, a view-dependent opacity compensation factor will be computed, which
          is useful for anti-aliasing. Default: False.

    Returns:
        A tuple:

        If `packed` is True:

        - **camera_ids**. The row indices of the projected Gaussians. Int32 tensor of shape [nnz].
        - **gaussian_ids**. The column indices of the projected Gaussians. Int32 tensor of shape [nnz].
        - **radii**. The maximum radius of the projected Gaussians in pixel unit. Int32 tensor of shape [nnz].
        - **means**. Projected Gaussian means in 2D. [nnz, 2]
        - **depths**. The z-depth of the projected Gaussians. [nnz]
        - **conics**. Inverse of the projected covariances. Return the flattend upper triangle with [nnz, 3]
        - **compensations**. The view-dependent opacity compensation factor. [nnz]

        If `packed` is False:

        - **radii**. The maximum radius of the projected Gaussians in pixel unit. Int32 tensor of shape [C, N].
        - **means**. Projected Gaussian means in 2D. [C, N, 2]
        - **depths**. The z-depth of the projected Gaussians. [C, N]
        - **conics**. Inverse of the projected covariances. Return the flattend upper triangle with [C, N, 3]
        - **compensations**. The view-dependent opacity compensation factor. [C, N]
    """
    C = viewmats.size(0)
    N = means.size(0)
    assert means.size() == (N, 3), means.size()
    assert viewmats.size() == (C, 4, 4), viewmats.size()
    assert Ks.size() == (C, 3, 3), Ks.size()
    means = means.contiguous()
    if covars is not None:
        assert covars.size() == (N, 6), covars.size()
        covars = covars.contiguous()
    else:
        assert quats is not None, "covars or quats is required"
        assert scales is not None, "covars or scales is required"
        assert quats.size() == (N, 4), quats.size()
        assert scales.size() == (N, 3), scales.size()
        quats = quats.contiguous()
        scales = scales.contiguous()
    if sparse_grad:
        assert packed, "sparse_grad is only supported when packed is True"

    viewmats = viewmats.contiguous()
    Ks = Ks.contiguous()
    if packed:
        return _FullyFusedProjectionPacked.apply(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            sparse_grad,
            calc_compensations,
            camera_model,
        )
    else:
        return _FullyFusedProjection.apply(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            calc_compensations,
            camera_model,
        )


@torch.no_grad()
def isect_tiles(
    means2d: Tensor,  # [C, N, 2] or [nnz, 2]
    radii: Tensor,  # [C, N] or [nnz]
    depths: Tensor,  # [C, N] or [nnz]
    tile_size: int,
    tile_width: int,
    tile_height: int,
    sort: bool = True,
    packed: bool = False,
    n_cameras: Optional[int] = None,
    camera_ids: Optional[Tensor] = None,
    gaussian_ids: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Maps projected Gaussians to intersecting tiles.

    Args:
        means2d: Projected Gaussian means. [C, N, 2] if packed is False, [nnz, 2] if packed is True.
        radii: Maximum radii of the projected Gaussians. [C, N] if packed is False, [nnz] if packed is True.
        depths: Z-depth of the projected Gaussians. [C, N] if packed is False, [nnz] if packed is True.
        tile_size: Tile size.
        tile_width: Tile width.
        tile_height: Tile height.
        sort: If True, the returned intersections will be sorted by the intersection ids. Default: True.
        packed: If True, the input tensors are packed. Default: False.
        n_cameras: Number of cameras. Required if packed is True.
        camera_ids: The row indices of the projected Gaussians. Required if packed is True.
        gaussian_ids: The column indices of the projected Gaussians. Required if packed is True.

    Returns:
        A tuple:

        - **Tiles per Gaussian**. The number of tiles intersected by each Gaussian.
          Int32 [C, N] if packed is False, Int32 [nnz] if packed is True.
        - **Intersection ids**. Each id is an 64-bit integer with the following
          information: camera_id (Xc bits) | tile_id (Xt bits) | depth (32 bits).
          Xc and Xt are the maximum number of bits required to represent the camera and
          tile ids, respectively. Int64 [n_isects]
        - **Flatten ids**. The global flatten indices in [C * N] or [nnz] (packed). [n_isects]
    """
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.size()
        assert radii.shape == (nnz,), radii.size()
        assert depths.shape == (nnz,), depths.size()
        assert camera_ids is not None, "camera_ids is required if packed is True"
        assert gaussian_ids is not None, "gaussian_ids is required if packed is True"
        assert n_cameras is not None, "n_cameras is required if packed is True"
        camera_ids = camera_ids.contiguous()
        gaussian_ids = gaussian_ids.contiguous()
        C = n_cameras

    else:
        C, N, _ = means2d.shape
        assert means2d.shape == (C, N, 2), means2d.size()
        assert radii.shape == (C, N), radii.size()
        assert depths.shape == (C, N), depths.size()

    tiles_per_gauss, isect_ids, flatten_ids = _make_lazy_cuda_func("isect_tiles")(
        means2d.contiguous(),
        radii.contiguous(),
        depths.contiguous(),
        camera_ids,
        gaussian_ids,
        C,
        tile_size,
        tile_width,
        tile_height,
        sort,
        True,  # DoubleBuffer: memory efficient radixsort
    )
    return tiles_per_gauss, isect_ids, flatten_ids


@torch.no_grad()
def isect_offset_encode(
    isect_ids: Tensor, n_cameras: int, tile_width: int, tile_height: int
) -> Tensor:
    """Encodes intersection ids to offsets.

    Args:
        isect_ids: Intersection ids. [n_isects]
        n_cameras: Number of cameras.
        tile_width: Tile width.
        tile_height: Tile height.

    Returns:
        Offsets. [C, tile_height, tile_width]
    """
    return _make_lazy_cuda_func("isect_offset_encode")(
        isect_ids.contiguous(), n_cameras, tile_width, tile_height
    )


def rasterize_to_pixels(
    means2d: Tensor,  # [C, N, 2] or [nnz, 2]
    conics: Tensor,  # [C, N, 3] or [nnz, 3]
    colors: Tensor,  # [C, N, channels] or [nnz, channels]
    opacities: Tensor,  # [C, N] or [nnz]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [C, tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    backgrounds: Optional[Tensor] = None,  # [C, channels]
    masks: Optional[Tensor] = None,  # [C, tile_height, tile_width]
    packed: bool = False,
    absgrad: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Rasterizes Gaussians to pixels.

    Args:
        means2d: Projected Gaussian means. [C, N, 2] if packed is False, [nnz, 2] if packed is True.
        conics: Inverse of the projected covariances with only upper triangle values. [C, N, 3] if packed is False, [nnz, 3] if packed is True.
        colors: Gaussian colors or ND features. [C, N, channels] if packed is False, [nnz, channels] if packed is True.
        opacities: Gaussian opacities that support per-view values. [C, N] if packed is False, [nnz] if packed is True.
        image_width: Image width.
        image_height: Image height.
        tile_size: Tile size.
        isect_offsets: Intersection offsets outputs from `isect_offset_encode()`. [C, tile_height, tile_width]
        flatten_ids: The global flatten indices in [C * N] or [nnz] from  `isect_tiles()`. [n_isects]
        backgrounds: Background colors. [C, channels]. Default: None.
        masks: Optional tile mask to skip rendering GS to masked tiles. [C, tile_height, tile_width]. Default: None.
        packed: If True, the input tensors are expected to be packed with shape [nnz, ...]. Default: False.
        absgrad: If True, the backward pass will compute a `.absgrad` attribute for `means2d`. Default: False.

    Returns:
        A tuple:

        - **Rendered colors**. [C, image_height, image_width, channels]
        - **Rendered alphas**. [C, image_height, image_width, 1]
    """

    C = isect_offsets.size(0)
    device = means2d.device
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.shape
        assert conics.shape == (nnz, 3), conics.shape
        assert colors.shape[0] == nnz, colors.shape
        assert opacities.shape == (nnz,), opacities.shape
    else:
        N = means2d.size(1)
        assert means2d.shape == (C, N, 2), means2d.shape
        assert conics.shape == (C, N, 3), conics.shape
        assert colors.shape[:2] == (C, N), colors.shape
        assert opacities.shape == (C, N), opacities.shape
    if backgrounds is not None:
        assert backgrounds.shape == (C, colors.shape[-1]), backgrounds.shape
        backgrounds = backgrounds.contiguous()
    if masks is not None:
        assert masks.shape == isect_offsets.shape, masks.shape
        masks = masks.contiguous()

    # Pad the channels to the nearest supported number if necessary
    channels = colors.shape[-1]
    if channels > 513 or channels == 0:
        # TODO: maybe worth to support zero channels?
        raise ValueError(f"Unsupported number of color channels: {channels}")
    if channels not in (
        1,
        2,
        3,
        4,
        5,
        8,
        9,
        16,
        17,
        32,
        33,
        64,
        65,
        128,
        129,
        256,
        257,
        512,
        513,
    ):
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        colors = torch.cat(
            [
                colors,
                torch.zeros(*colors.shape[:-1], padded_channels, device=device),
            ],
            dim=-1,
        )
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(
                        *backgrounds.shape[:-1], padded_channels, device=device
                    ),
                ],
                dim=-1,
            )
    else:
        padded_channels = 0

    tile_height, tile_width = isect_offsets.shape[1:3]
    assert (
        tile_height * tile_size >= image_height
    ), f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
    assert (
        tile_width * tile_size >= image_width
    ), f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"

    render_colors, render_alphas = _RasterizeToPixels.apply(
        means2d.contiguous(),
        conics.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        backgrounds,
        masks,
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
        absgrad,
    )

    if padded_channels > 0:
        render_colors = render_colors[..., :-padded_channels]
    return render_colors, render_alphas


@torch.no_grad()
def rasterize_to_indices_in_range(
    range_start: int,
    range_end: int,
    transmittances: Tensor,  # [C, image_height, image_width]
    means2d: Tensor,  # [C, N, 2]
    conics: Tensor,  # [C, N, 3]
    opacities: Tensor,  # [C, N]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [C, tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
) -> Tuple[Tensor, Tensor, Tensor]:
    """Rasterizes a batch of Gaussians to images but only returns the indices.

    .. note::

        This function supports iterative rasterization, in which each call of this function
        will rasterize a batch of Gaussians from near to far, defined by `[range_start, range_end)`.
        If a one-step full rasterization is desired, set `range_start` to 0 and `range_end` to a really
        large number, e.g, 1e10.

    Args:
        range_start: The start batch of Gaussians to be rasterized (inclusive).
        range_end: The end batch of Gaussians to be rasterized (exclusive).
        transmittances: Currently transmittances. [C, image_height, image_width]
        means2d: Projected Gaussian means. [C, N, 2]
        conics: Inverse of the projected covariances with only upper triangle values. [C, N, 3]
        opacities: Gaussian opacities that support per-view values. [C, N]
        image_width: Image width.
        image_height: Image height.
        tile_size: Tile size.
        isect_offsets: Intersection offsets outputs from `isect_offset_encode()`. [C, tile_height, tile_width]
        flatten_ids: The global flatten indices in [C * N] from  `isect_tiles()`. [n_isects]

    Returns:
        A tuple:

        - **Gaussian ids**. Gaussian ids for the pixel intersection. A flattened list of shape [M].
        - **Pixel ids**. pixel indices (row-major). A flattened list of shape [M].
        - **Camera ids**. Camera indices. A flattened list of shape [M].
    """

    C, N, _ = means2d.shape
    assert conics.shape == (C, N, 3), conics.shape
    assert opacities.shape == (C, N), opacities.shape
    assert isect_offsets.shape[0] == C, isect_offsets.shape

    tile_height, tile_width = isect_offsets.shape[1:3]
    assert (
        tile_height * tile_size >= image_height
    ), f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
    assert (
        tile_width * tile_size >= image_width
    ), f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"

    out_gauss_ids, out_indices = _make_lazy_cuda_func("rasterize_to_indices_in_range")(
        range_start,
        range_end,
        transmittances.contiguous(),
        means2d.contiguous(),
        conics.contiguous(),
        opacities.contiguous(),
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
    )
    out_pixel_ids = out_indices % (image_width * image_height)
    out_camera_ids = out_indices // (image_width * image_height)
    return out_gauss_ids, out_pixel_ids, out_camera_ids


class _QuatScaleToCovarPreci(torch.autograd.Function):
    """Converts quaternions and scales to covariance and precision matrices."""

    @staticmethod
    def forward(
        ctx,
        quats: Tensor,  # [N, 4],
        scales: Tensor,  # [N, 3],
        compute_covar: bool = True,
        compute_preci: bool = True,
        triu: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        covars, precis = _make_lazy_cuda_func("quat_scale_to_covar_preci_fwd")(
            quats, scales, compute_covar, compute_preci, triu
        )
        ctx.save_for_backward(quats, scales)
        ctx.compute_covar = compute_covar
        ctx.compute_preci = compute_preci
        ctx.triu = triu
        return covars, precis

    @staticmethod
    def backward(ctx, v_covars: Tensor, v_precis: Tensor):
        quats, scales = ctx.saved_tensors
        compute_covar = ctx.compute_covar
        compute_preci = ctx.compute_preci
        triu = ctx.triu
        if compute_covar and v_covars.is_sparse:
            v_covars = v_covars.to_dense()
        if compute_preci and v_precis.is_sparse:
            v_precis = v_precis.to_dense()
        v_quats, v_scales = _make_lazy_cuda_func("quat_scale_to_covar_preci_bwd")(
            quats,
            scales,
            v_covars.contiguous() if compute_covar else None,
            v_precis.contiguous() if compute_preci else None,
            triu,
        )
        return v_quats, v_scales, None, None, None


class _Proj(torch.autograd.Function):
    """Perspective fully_fused_projection on Gaussians."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [C, N, 3]
        covars: Tensor,  # [C, N, 3, 3]
        Ks: Tensor,  # [C, 3, 3]
        width: int,
        height: int,
        camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
    ) -> Tuple[Tensor, Tensor]:
        camera_model_type = _make_lazy_cuda_obj(
            f"CameraModelType.{camera_model.upper()}"
        )

        means2d, covars2d = _make_lazy_cuda_func("proj_fwd")(
            means,
            covars,
            Ks,
            width,
            height,
            camera_model_type,
        )
        ctx.save_for_backward(means, covars, Ks)
        ctx.width = width
        ctx.height = height
        ctx.camera_model_type = camera_model_type
        return means2d, covars2d

    @staticmethod
    def backward(ctx, v_means2d: Tensor, v_covars2d: Tensor):
        means, covars, Ks = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        camera_model_type = ctx.camera_model_type
        v_means, v_covars = _make_lazy_cuda_func("proj_bwd")(
            means,
            covars,
            Ks,
            width,
            height,
            camera_model_type,
            v_means2d.contiguous(),
            v_covars2d.contiguous(),
        )
        return v_means, v_covars, None, None, None, None


class _WorldToCam(torch.autograd.Function):
    """Transforms Gaussians from world to camera space."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [N, 3]
        covars: Tensor,  # [N, 3, 3]
        viewmats: Tensor,  # [C, 4, 4]
    ) -> Tuple[Tensor, Tensor]:
        means_c, covars_c = _make_lazy_cuda_func("world_to_cam_fwd")(
            means, covars, viewmats
        )
        ctx.save_for_backward(means, covars, viewmats)
        return means_c, covars_c

    @staticmethod
    def backward(ctx, v_means_c: Tensor, v_covars_c: Tensor):
        means, covars, viewmats = ctx.saved_tensors
        v_means, v_covars, v_viewmats = _make_lazy_cuda_func("world_to_cam_bwd")(
            means,
            covars,
            viewmats,
            v_means_c.contiguous(),
            v_covars_c.contiguous(),
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
            ctx.needs_input_grad[2],
        )
        if not ctx.needs_input_grad[0]:
            v_means = None
        if not ctx.needs_input_grad[1]:
            v_covars = None
        if not ctx.needs_input_grad[2]:
            v_viewmats = None
        return v_means, v_covars, v_viewmats


class _FullyFusedProjection(torch.autograd.Function):
    """Projects Gaussians to 2D."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [N, 3]
        covars: Tensor,  # [N, 6] or None
        quats: Tensor,  # [N, 4] or None
        scales: Tensor,  # [N, 3] or None
        viewmats: Tensor,  # [C, 4, 4]
        Ks: Tensor,  # [C, 3, 3]
        width: int,
        height: int,
        eps2d: float,
        near_plane: float,
        far_plane: float,
        radius_clip: float,
        calc_compensations: bool,
        camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        camera_model_type = _make_lazy_cuda_obj(
            f"CameraModelType.{camera_model.upper()}"
        )

        # "covars" and {"quats", "scales"} are mutually exclusive
        radii, means2d, depths, conics, compensations = _make_lazy_cuda_func(
            "fully_fused_projection_fwd"
        )(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            calc_compensations,
            camera_model_type,
        )
        if not calc_compensations:
            compensations = None
        ctx.save_for_backward(
            means, covars, quats, scales, viewmats, Ks, radii, conics, compensations
        )
        ctx.width = width
        ctx.height = height
        ctx.eps2d = eps2d
        ctx.camera_model_type = camera_model_type

        return radii, means2d, depths, conics, compensations

    @staticmethod
    def backward(ctx, v_radii, v_means2d, v_depths, v_conics, v_compensations):
        (
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            radii,
            conics,
            compensations,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        eps2d = ctx.eps2d
        camera_model_type = ctx.camera_model_type
        if v_compensations is not None:
            v_compensations = v_compensations.contiguous()
        v_means, v_covars, v_quats, v_scales, v_viewmats = _make_lazy_cuda_func(
            "fully_fused_projection_bwd"
        )(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            camera_model_type,
            radii,
            conics,
            compensations,
            v_means2d.contiguous(),
            v_depths.contiguous(),
            v_conics.contiguous(),
            v_compensations,
            ctx.needs_input_grad[4],  # viewmats_requires_grad
        )
        if not ctx.needs_input_grad[0]:
            v_means = None
        if not ctx.needs_input_grad[1]:
            v_covars = None
        if not ctx.needs_input_grad[2]:
            v_quats = None
        if not ctx.needs_input_grad[3]:
            v_scales = None
        if not ctx.needs_input_grad[4]:
            v_viewmats = None
        return (
            v_means,
            v_covars,
            v_quats,
            v_scales,
            v_viewmats,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _RasterizeToPixels(torch.autograd.Function):
    """Rasterize gaussians"""

    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,  # [C, N, 2]
        conics: Tensor,  # [C, N, 3]
        colors: Tensor,  # [C, N, D]
        opacities: Tensor,  # [C, N]
        backgrounds: Tensor,  # [C, D], Optional
        masks: Tensor,  # [C, tile_height, tile_width], Optional
        width: int,
        height: int,
        tile_size: int,
        isect_offsets: Tensor,  # [C, tile_height, tile_width]
        flatten_ids: Tensor,  # [n_isects]
        absgrad: bool,
    ) -> Tuple[Tensor, Tensor]:
        render_colors, render_alphas, last_ids = _make_lazy_cuda_func(
            "rasterize_to_pixels_fwd"
        )(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
        )

        ctx.save_for_backward(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
        )
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size
        ctx.absgrad = absgrad

        # double to float
        render_alphas = render_alphas.float()
        return render_colors, render_alphas

    @staticmethod
    def backward(
        ctx,
        v_render_colors: Tensor,  # [C, H, W, 3]
        v_render_alphas: Tensor,  # [C, H, W, 1]
    ):
        (
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        tile_size = ctx.tile_size
        absgrad = ctx.absgrad

        (
            v_means2d_abs,
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
        ) = _make_lazy_cuda_func("rasterize_to_pixels_bwd")(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
            absgrad,
        )

        if absgrad:
            means2d.absgrad = v_means2d_abs

        if ctx.needs_input_grad[4]:
            v_backgrounds = (v_render_colors * (1.0 - render_alphas).float()).sum(
                dim=(1, 2)
            )
        else:
            v_backgrounds = None

        return (
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
            v_backgrounds,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _FullyFusedProjectionPacked(torch.autograd.Function):
    """Projects Gaussians to 2D. Return packed tensors."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [N, 3]
        covars: Tensor,  # [N, 6] or None
        quats: Tensor,  # [N, 4] or None
        scales: Tensor,  # [N, 3] or None
        viewmats: Tensor,  # [C, 4, 4]
        Ks: Tensor,  # [C, 3, 3]
        width: int,
        height: int,
        eps2d: float,
        near_plane: float,
        far_plane: float,
        radius_clip: float,
        sparse_grad: bool,
        calc_compensations: bool,
        camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        camera_model_type = _make_lazy_cuda_obj(
            f"CameraModelType.{camera_model.upper()}"
        )

        (
            indptr,
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            conics,
            compensations,
        ) = _make_lazy_cuda_func("fully_fused_projection_packed_fwd")(
            means,
            covars,  # optional
            quats,  # optional
            scales,  # optional
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            calc_compensations,
            camera_model_type,
        )
        if not calc_compensations:
            compensations = None
        ctx.save_for_backward(
            camera_ids,
            gaussian_ids,
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            conics,
            compensations,
        )
        ctx.width = width
        ctx.height = height
        ctx.eps2d = eps2d
        ctx.sparse_grad = sparse_grad
        ctx.camera_model_type = camera_model_type

        return camera_ids, gaussian_ids, radii, means2d, depths, conics, compensations

    @staticmethod
    def backward(
        ctx,
        v_camera_ids,
        v_gaussian_ids,
        v_radii,
        v_means2d,
        v_depths,
        v_conics,
        v_compensations,
    ):
        (
            camera_ids,
            gaussian_ids,
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            conics,
            compensations,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        eps2d = ctx.eps2d
        sparse_grad = ctx.sparse_grad
        camera_model_type = ctx.camera_model_type

        if v_compensations is not None:
            v_compensations = v_compensations.contiguous()
        v_means, v_covars, v_quats, v_scales, v_viewmats = _make_lazy_cuda_func(
            "fully_fused_projection_packed_bwd"
        )(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            camera_model_type,
            camera_ids,
            gaussian_ids,
            conics,
            compensations,
            v_means2d.contiguous(),
            v_depths.contiguous(),
            v_conics.contiguous(),
            v_compensations,
            ctx.needs_input_grad[4],  # viewmats_requires_grad
            sparse_grad,
        )

        if not ctx.needs_input_grad[0]:
            v_means = None
        else:
            if sparse_grad:
                # TODO: gaussian_ids is duplicated so not ideal.
                # An idea is to directly set the attribute (e.g., .sparse_grad) of
                # the tensor but this requires the tensor to be leaf node only. And
                # a customized optimizer would be needed in this case.
                v_means = torch.sparse_coo_tensor(
                    indices=gaussian_ids[None],  # [1, nnz]
                    values=v_means,  # [nnz, 3]
                    size=means.size(),  # [N, 3]
                    is_coalesced=len(viewmats) == 1,
                )
        if not ctx.needs_input_grad[1]:
            v_covars = None
        else:
            if sparse_grad:
                v_covars = torch.sparse_coo_tensor(
                    indices=gaussian_ids[None],  # [1, nnz]
                    values=v_covars,  # [nnz, 6]
                    size=covars.size(),  # [N, 6]
                    is_coalesced=len(viewmats) == 1,
                )
        if not ctx.needs_input_grad[2]:
            v_quats = None
        else:
            if sparse_grad:
                v_quats = torch.sparse_coo_tensor(
                    indices=gaussian_ids[None],  # [1, nnz]
                    values=v_quats,  # [nnz, 4]
                    size=quats.size(),  # [N, 4]
                    is_coalesced=len(viewmats) == 1,
                )
        if not ctx.needs_input_grad[3]:
            v_scales = None
        else:
            if sparse_grad:
                v_scales = torch.sparse_coo_tensor(
                    indices=gaussian_ids[None],  # [1, nnz]
                    values=v_scales,  # [nnz, 3]
                    size=scales.size(),  # [N, 3]
                    is_coalesced=len(viewmats) == 1,
                )
        if not ctx.needs_input_grad[4]:
            v_viewmats = None

        return (
            v_means,
            v_covars,
            v_quats,
            v_scales,
            v_viewmats,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _SphericalHarmonics(torch.autograd.Function):
    """Spherical Harmonics"""

    @staticmethod
    def forward(
        ctx, sh_degree: int, dirs: Tensor, coeffs: Tensor, masks: Tensor
    ) -> Tensor:
        colors = _make_lazy_cuda_func("compute_sh_fwd")(sh_degree, dirs, coeffs, masks)
        ctx.save_for_backward(dirs, coeffs, masks)
        ctx.sh_degree = sh_degree
        ctx.num_bases = coeffs.shape[-2]
        return colors

    @staticmethod
    def backward(ctx, v_colors: Tensor):
        dirs, coeffs, masks = ctx.saved_tensors
        sh_degree = ctx.sh_degree
        num_bases = ctx.num_bases
        compute_v_dirs = ctx.needs_input_grad[1]
        v_coeffs, v_dirs = _make_lazy_cuda_func("compute_sh_bwd")(
            num_bases,
            sh_degree,
            dirs,
            coeffs,
            masks,
            v_colors.contiguous(),
            compute_v_dirs,
        )
        if not compute_v_dirs:
            v_dirs = None
        return None, v_dirs, v_coeffs, None


###### 2DGS ######
def fully_fused_projection_2dgs(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    packed: bool = False,
    sparse_grad: bool = False,
    norm_rot_grad: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Prepare Gaussians for rasterization

    This function prepares ray-splat intersection matrices, computes
    per splat bounding box and 2D means in image space.

    Args:
        means: Gaussian means. [N, 3]
        quats: Quaternions (No need to be normalized). [N, 4].
        scales: Scales. [N, 3].
        viewmats: Camera-to-world matrices. [C, 4, 4]
        Ks: Camera intrinsics. [C, 3, 3]
        width: Image width.
        height: Image height.
        near_plane: Near plane distance. Default: 0.01.
        far_plane: Far plane distance. Default: 200.
        radius_clip: Gaussians with projected radii smaller than this value will be ignored. Default: 0.0.
        packed: If True, the output tensors will be packed into a flattened tensor. Default: False.
        sparse_grad (Experimental): This is only effective when `packed` is True. If True, during backward the gradients
          of {`means`, `covars`, `quats`, `scales`} will be a sparse Tensor in COO layout. Default: False.

    Returns:
        A tuple:

        If `packed` is True:

        - **camera_ids**. The row indices of the projected Gaussians. Int32 tensor of shape [nnz].
        - **gaussian_ids**. The column indices of the projected Gaussians. Int32 tensor of shape [nnz].
        - **radii**. The maximum radius of the projected Gaussians in pixel unit. Int32 tensor of shape [nnz].
        - **means**. Projected Gaussian means in 2D. [nnz, 2]
        - **depths**. The z-depth of the projected Gaussians. [nnz]
        - **ray_transforms**. transformation matrices that transforms xy-planes in pixel spaces into splat coordinates (WH)^T in equation (9) in paper [nnz, 3, 3]
        - **normals**. The normals in camera spaces. [nnz, 3]

        If `packed` is False:

        - **radii**. The maximum radius of the projected Gaussians in pixel unit. Int32 tensor of shape [C, N].
        - **means**. Projected Gaussian means in 2D. [C, N, 2]
        - **depths**. The z-depth of the projected Gaussians. [C, N]
        - **ray_transforms**. transformation matrices that transforms xy-planes in pixel spaces into splat coordinates.
        - **normals**. The normals in camera spaces. [C, N, 3]

    """
    C = viewmats.size(0)
    N = means.size(0)
    assert means.size() == (N, 3), means.size()
    assert viewmats.size() == (C, 4, 4), viewmats.size()
    assert Ks.size() == (C, 3, 3), Ks.size()
    means = means.contiguous()
    assert quats is not None, "quats is required"
    assert scales is not None, "scales is required"
    assert quats.size() == (N, 4), quats.size()
    assert scales.size() == (N, 3), scales.size()
    quats = quats.contiguous()
    scales = scales.contiguous()
    if sparse_grad:
        assert packed, "sparse_grad is only supported when packed is True"

    viewmats = viewmats.contiguous()
    Ks = Ks.contiguous()
    if packed:
        return _FullyFusedProjectionPacked2DGS.apply(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            near_plane,
            far_plane,
            radius_clip,
            sparse_grad,
            norm_rot_grad,
        )
    else:
        return _FullyFusedProjection2DGS.apply(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            norm_rot_grad,
        )


class _FullyFusedProjection2DGS(torch.autograd.Function):
    """Projects Gaussians to 2D."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,
        quats: Tensor,
        scales: Tensor,
        viewmats: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        eps2d: float,
        near_plane: float,
        far_plane: float,
        radius_clip: float,
        norm_rot_grad: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        radii, means2d, depths, ray_transforms, normals = _make_lazy_cuda_func(
            "fully_fused_projection_fwd_2dgs"
        )(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
        )
        ctx.save_for_backward(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            radii,
            ray_transforms,
            normals,
        )
        ctx.width = width
        ctx.height = height
        ctx.eps2d = eps2d
        ctx.norm_rot_grad = norm_rot_grad

        return radii, means2d, depths, ray_transforms, normals

    @staticmethod
    def backward(ctx, v_radii, v_means2d, v_depths, v_ray_transforms, v_normals):
        (
            means,
            quats,
            scales,
            viewmats,
            Ks,
            radii,
            ray_transforms,
            normals,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        eps2d = ctx.eps2d
        norm_rot_grad = ctx.norm_rot_grad

        v_means, v_quats, v_scales, v_viewmats = _make_lazy_cuda_func(
            "fully_fused_projection_bwd2_2dgs"
            if norm_rot_grad
            else "fully_fused_projection_bwd_2dgs"
        )(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            radii,
            ray_transforms,
            v_means2d.contiguous(),
            v_depths.contiguous(),
            v_normals.contiguous(),
            v_ray_transforms.contiguous(),
            ctx.needs_input_grad[3],  # viewmats_requires_grad
        )
        if not ctx.needs_input_grad[0]:
            v_means = None
        if not ctx.needs_input_grad[1]:
            v_quats = None
        if not ctx.needs_input_grad[2]:
            v_scales = None
        if not ctx.needs_input_grad[3]:
            v_viewmats = None

        return (
            v_means,
            v_quats,
            v_scales,
            v_viewmats,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _FullyFusedProjectionPacked2DGS(torch.autograd.Function):
    """Projects Gaussians to 2D. Return packed tensors."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,  # [N, 3]
        quats: Tensor,  # [N, 4]
        scales: Tensor,  # [N, 3]
        viewmats: Tensor,  # [C, 4, 4]
        Ks: Tensor,  # [C, 3, 3]
        width: int,
        height: int,
        near_plane: float,
        far_plane: float,
        radius_clip: float,
        sparse_grad: bool,
        norm_rot_grad: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        (
            indptr,
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            ray_transforms,
            normals,
        ) = _make_lazy_cuda_func("fully_fused_projection_packed_fwd_2dgs")(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            near_plane,
            far_plane,
            radius_clip,
        )
        ctx.save_for_backward(
            camera_ids,
            gaussian_ids,
            means,
            quats,
            scales,
            viewmats,
            Ks,
            ray_transforms,
        )
        ctx.width = width
        ctx.height = height
        ctx.sparse_grad = sparse_grad
        ctx.norm_rot_grad = norm_rot_grad

        return camera_ids, gaussian_ids, radii, means2d, depths, ray_transforms, normals

    @staticmethod
    def backward(
        ctx,
        v_camera_ids,
        v_gaussian_ids,
        v_radii,
        v_means2d,
        v_depths,
        v_ray_transforms,
        v_normals,
    ):
        (
            camera_ids,
            gaussian_ids,
            means,
            quats,
            scales,
            viewmats,
            Ks,
            ray_transforms,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        sparse_grad = ctx.sparse_grad
        norm_rot_grad = ctx.norm_rot_grad

        v_means, v_quats, v_scales, v_viewmats = _make_lazy_cuda_func(
            "fully_fused_projection_packed_bwd2_2dgs"
            if norm_rot_grad
            else "fully_fused_projection_packed_bwd_2dgs"
        )(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            camera_ids,
            gaussian_ids,
            ray_transforms,
            v_means2d.contiguous(),
            v_depths.contiguous(),
            v_ray_transforms.contiguous(),
            v_normals.contiguous(),
            ctx.needs_input_grad[4],  # viewmats_requires_grad
            sparse_grad,
        )

        if not ctx.needs_input_grad[0]:
            v_means = None
        else:
            if sparse_grad:
                # TODO: gaussian_ids is duplicated so not ideal.
                # An idea is to directly set the attribute (e.g., .sparse_grad) of
                # the tensor but this requires the tensor to be leaf node only. And
                # a customized optimizer would be needed in this case.
                v_means = torch.sparse_coo_tensor(
                    indices=gaussian_ids[None],  # [1, nnz]
                    values=v_means,  # [nnz, 3]
                    size=means.size(),  # [N, 3]
                    is_coalesced=len(viewmats) == 1,
                )
        if not ctx.needs_input_grad[1]:
            v_quats = None
        else:
            if sparse_grad:
                v_quats = torch.sparse_coo_tensor(
                    indices=gaussian_ids[None],  # [1, nnz]
                    values=v_quats,  # [nnz, 4]
                    size=quats.size(),  # [N, 4]
                    is_coalesced=len(viewmats) == 1,
                )
        if not ctx.needs_input_grad[2]:
            v_scales = None
        else:
            if sparse_grad:
                v_scales = torch.sparse_coo_tensor(
                    indices=gaussian_ids[None],  # [1, nnz]
                    values=v_scales,  # [nnz, 3]
                    size=scales.size(),  # [N, 3]
                    is_coalesced=len(viewmats) == 1,
                )
        if not ctx.needs_input_grad[4]:
            v_viewmats = None

        return (
            v_means,
            v_quats,
            v_scales,
            v_viewmats,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def rasterize_to_pixels_2dgs(
    means2d: Tensor,
    ray_transforms: Tensor,
    colors: Tensor,
    opacities: Tensor,
    normals: Tensor,
    densify: Tensor,
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,
    flatten_ids: Tensor,
    backgrounds: Optional[Tensor] = None,
    masks: Optional[Tensor] = None,
    packed: bool = False,
    absgrad: bool = False,
    distloss: bool = False,
    gs_contrib_threshold: float = 0.0,  # added
) -> Tuple[Tensor, Tensor]:
    """Rasterize Gaussians to pixels.

    Args:
        means2d: Projected Gaussian means. [C, N, 2] if packed is False, [nnz, 2] if packed is True.
        ray_transforms: transformation matrices that transforms xy-planes in pixel spaces into splat coordinates. [C, N, 3, 3] if packed is False, [nnz, channels] if packed is True.
        colors: Gaussian colors or ND features. [C, N, channels] if packed is False, [nnz, channels] if packed is True.
        opacities: Gaussian opacities that support per-view values. [C, N] if packed is False, [nnz] if packed is True.
        normals: The normals in camera space. [C, N, 3] if packed is False, [nnz, 3] if packed is True.
        densify: Dummy variable to keep track of gradient for densification. [C, N, 2] if packed, [nnz, 3] if packed is True.
        tile_size: Tile size.
        isect_offsets: Intersection offsets outputs from `isect_offset_encode()`. [C, tile_height, tile_width]
        flatten_ids: The global flatten indices in [C * N] or [nnz] from  `isect_tiles()`. [n_isects]
        backgrounds: Background colors. [C, channels]. Default: None.
        masks: Optional tile mask to skip rendering GS to masked tiles. [C, tile_height, tile_width]. Default: None.
        packed: If True, the input tensors are expected to be packed with shape [nnz, ...]. Default: False.
        absgrad: If True, the backward pass will compute a `.absgrad` attribute for `means2d`. Default: False.

        gs_contrib_threshold: The threshold for Gaussian opacity contribution; intersection counted if above this threshold. Default: 0.0.

    Returns:
        A tuple:

        - **Rendered colors**.      [C, image_height, image_width, channels]
        - **Rendered alphas**.      [C, image_height, image_width, 1]
        - **Rendered normals**.     [C, image_height, image_width, 3]
        - **Rendered distortion**.  [C, image_height, image_width, 1]
        - **Rendered median depth**.[C, image_height, image_width, 1]
        - **Gaussian opacity contribution (sum)**. [N,]
        - **Gaussian opacity contribution (count)**. [N,]

    """
    C = isect_offsets.size(0)
    device = means2d.device
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.shape
        assert ray_transforms.shape == (nnz, 3, 3), ray_transforms.shape
        assert colors.shape[0] == nnz, colors.shape
        assert opacities.shape == (nnz,), opacities.shape
    else:
        N = means2d.size(1)
        assert means2d.shape == (C, N, 2), means2d.shape
        assert ray_transforms.shape == (C, N, 3, 3), ray_transforms.shape
        assert colors.shape[:2] == (C, N), colors.shape
        assert opacities.shape == (C, N), opacities.shape
    if backgrounds is not None:
        assert backgrounds.shape == (C, colors.shape[-1]), backgrounds.shape
        backgrounds = backgrounds.contiguous()

    # Pad the channels to the nearest supported number if necessary
    channels = colors.shape[-1]
    if channels > 512 or channels == 0:
        # TODO: maybe worth to support zero channels?
        raise ValueError(f"Unsupported number of color channels: {channels}")
    if channels not in (1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512):
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        colors = torch.cat(
            [colors, torch.empty(*colors.shape[:-1], padded_channels, device=device)],
            dim=-1,
        )
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.empty(
                        *backgrounds.shape[:-1], padded_channels, device=device
                    ),
                ],
                dim=-1,
            )
    else:
        padded_channels = 0
    tile_height, tile_width = isect_offsets.shape[1:3]
    assert (
        tile_height * tile_size >= image_height
    ), f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
    assert (
        tile_width * tile_size >= image_width
    ), f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"

    (
        render_colors,
        render_alphas,
        render_normals,
        render_distort,
        render_median,
        gs_contrib_sum,  # added
        gs_contrib_count,  # added
    ) = _RasterizeToPixels2DGS.apply(
        means2d.contiguous(),
        ray_transforms.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        normals.contiguous(),
        densify.contiguous(),
        backgrounds,
        masks,
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
        absgrad,
        distloss,
        gs_contrib_threshold,  # added
    )

    if padded_channels > 0:
        render_colors = render_colors[..., :-padded_channels]

    return (
        render_colors,
        render_alphas,
        render_normals,
        render_distort,
        render_median,
        gs_contrib_sum,
        gs_contrib_count,
    )


PRE = "rasterize_to_pixels_"
TGSPOST = "_textured_gaussians"
tgs_fns = {
    "bilinear": (f"{PRE}fwd{TGSPOST}", f"{PRE}bwd{TGSPOST}"),
    "bilinear_bwd2": (f"{PRE}fwd{TGSPOST}", f"{PRE}bwd2{TGSPOST}"),
    "bilinear2": (f"{PRE}fwd_bilinear2{TGSPOST}", f"{PRE}bwd_bilinear2{TGSPOST}"),
    "bilinear3": (f"{PRE}fwd_bilinear3{TGSPOST}", f"{PRE}bwd_bilinear3{TGSPOST}"),
    "bilinear3_bwd2": (f"{PRE}fwd_bilinear3{TGSPOST}", f"{PRE}bwd2_bilinear3{TGSPOST}"),
    "bilinear4": (f"{PRE}fwd_bilinear4{TGSPOST}", f"{PRE}bwd_bilinear4{TGSPOST}"),
    "bilinear4_bwd2": (f"{PRE}fwd_bilinear4{TGSPOST}", f"{PRE}bwd2_bilinear4{TGSPOST}"),
    "mipmapped": (f"{PRE}fwd_mip{TGSPOST}", f"{PRE}bwd_mip{TGSPOST}"),
    "anisotropic": (f"{PRE}fwd_aniso{TGSPOST}", f"{PRE}bwd_aniso{TGSPOST}"),
    "anisotropic_bilinear": (
        f"{PRE}fwd_aniso_bilinear{TGSPOST}",
        f"{PRE}bwd_aniso_bilinear{TGSPOST}",
    ),
}


def rasterize_to_pixels_textured_gaussians(
    means2d: Tensor,
    ray_transforms: Tensor,
    colors: Tensor,
    opacities: Tensor,
    textures: Tensor,
    texture_range_x: float,
    texture_range_y: float,
    normals: Tensor,
    densify: Tensor,
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,
    flatten_ids: Tensor,
    backgrounds: Optional[Tensor] = None,
    masks: Optional[Tensor] = None,
    packed: bool = False,
    absgrad: bool = False,
    distloss: bool = False,
    gs_contrib_threshold: float = 0.0,  # added
    filtering: Filtering = "bilinear",
    g_weight: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """Rasterize Textured Gaussians to pixels.

    Args:
        means2d: Projected Gaussian means. [C, N, 2] if packed is False, [nnz, 2] if packed is True.
        ray_transforms: transformation matrices that transforms xy-planes in pixel spaces into splat coordinates. [C, N, 3, 3] if packed is False, [nnz, channels] if packed is True.
        colors: Gaussian colors or ND features. [C, N, channels] if packed is False, [nnz, channels] if packed is True.
        opacities: Gaussian opacities that support per-view values. [C, N] if packed is False, [nnz] if packed is True.
        textures: Gaussian 2D textures in the shape of [N, th, tw, 4]. packed not suported.
        normals: The normals in camera space. [C, N, 3] if packed is False, [nnz, 3] if packed is True.
        densify: Dummy variable to keep track of gradient for densification. [C, N, 2] if packed, [nnz, 3] if packed is True.
        tile_size: Tile size.
        isect_offsets: Intersection offsets outputs from `isect_offset_encode()`. [C, tile_height, tile_width]
        flatten_ids: The global flatten indices in [C * N] or [nnz] from  `isect_tiles()`. [n_isects]
        backgrounds: Background colors. [C, channels]. Default: None.
        masks: Optional tile mask to skip rendering GS to masked tiles. [C, tile_height, tile_width]. Default: None.
        packed: If True, the input tensors are expected to be packed with shape [nnz, ...]. Default: False.
        absgrad: If True, the backward pass will compute a `.absgrad` attribute for `means2d`. Default: False.

        gs_contrib_threshold: The threshold for Gaussian opacity contribution; intersection counted if above this threshold. Default: 0.0.

    Returns:
        A tuple:

        - **Rendered colors**.      [C, image_height, image_width, channels]
        - **Rendered alphas**.      [C, image_height, image_width, 1]
        - **Rendered normals**.     [C, image_height, image_width, 3]
        - **Rendered distortion**.  [C, image_height, image_width, 1]
        - **Rendered median depth**.[C, image_height, image_width, 1]
        - **Gaussian opacity contribution (sum)**. [N,]
        - **Gaussian opacity contribution (count)**. [N,]

    """
    C = isect_offsets.size(0)
    device = means2d.device
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.shape
        assert ray_transforms.shape == (nnz, 3, 3), ray_transforms.shape
        assert colors.shape[0] == nnz, colors.shape
        assert opacities.shape == (nnz,), opacities.shape
    else:
        N = means2d.size(1)
        assert means2d.shape == (C, N, 2), means2d.shape
        assert ray_transforms.shape == (C, N, 3, 3), ray_transforms.shape
        assert colors.shape[:2] == (C, N), colors.shape
        assert opacities.shape == (C, N), opacities.shape
    if backgrounds is not None:
        assert backgrounds.shape == (C, colors.shape[-1]), backgrounds.shape
        backgrounds = backgrounds.contiguous()

    # Pad the channels to the nearest supported number if necessary
    channels = colors.shape[-1]
    if channels > 512 or channels == 0:
        # TODO: maybe worth to support zero channels?
        raise ValueError(f"Unsupported number of color channels: {channels}")
    if channels not in (1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512):
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        colors = torch.cat(
            [colors, torch.empty(*colors.shape[:-1], padded_channels, device=device)],
            dim=-1,
        )
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.empty(
                        *backgrounds.shape[:-1], padded_channels, device=device
                    ),
                ],
                dim=-1,
            )
    else:
        padded_channels = 0
    tile_height, tile_width = isect_offsets.shape[1:3]
    assert (
        tile_height * tile_size >= image_height
    ), f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
    assert (
        tile_width * tile_size >= image_width
    ), f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"

    if filtering in tgs_fns:
        (
            render_colors,
            render_alphas,
            render_normals,
            render_distort,
            render_median,
            gs_contrib_sum,  # added
            gs_contrib_count,  # added
        ) = _RasterizeToPixelsTexturedGaussians.apply(
            tgs_fns[filtering][0],
            tgs_fns[filtering][1],
            means2d.contiguous(),
            ray_transforms.contiguous(),
            colors.contiguous(),
            opacities.contiguous(),
            textures.contiguous(),
            texture_range_x,
            texture_range_y,
            normals.contiguous(),
            densify.contiguous(),
            backgrounds,
            masks,
            image_width,
            image_height,
            tile_size,
            isect_offsets.contiguous(),
            flatten_ids.contiguous(),
            absgrad,
            distloss,
            gs_contrib_threshold,  # added
            g_weight,
        )
    else:
        match filtering:
            case "mipmapped2":
                assert textures.size(1) == textures.size(2)
                log_texture_res = textures.size(1).bit_length() - 1
                assert textures.size(1) == 1 << log_texture_res

                log_reduce = 4

                mip_textures = generate_mipmap(
                    textures, log_texture_res, log_reduce, tile_size
                )

                (
                    render_colors,
                    render_alphas,
                    render_normals,
                    render_distort,
                    render_median,
                    gs_contrib_sum,
                    gs_contrib_count,
                ) = _RasterizeToPixelsMip2TexturedGaussians.apply(
                    means2d.contiguous(),
                    ray_transforms.contiguous(),
                    colors.contiguous(),
                    opacities.contiguous(),
                    mip_textures.contiguous(),
                    log_texture_res,
                    texture_range_x,
                    texture_range_y,
                    normals.contiguous(),
                    densify.contiguous(),
                    backgrounds,
                    masks,
                    image_width,
                    image_height,
                    tile_size,
                    isect_offsets.contiguous(),
                    flatten_ids.contiguous(),
                    absgrad,
                    distloss,
                    gs_contrib_threshold,
                    g_weight,
                )
            case _:
                raise Exception(f"Unsupported filter type {filtering}")

    if padded_channels > 0:
        render_colors = render_colors[..., :-padded_channels]

    return (
        render_colors,
        render_alphas,
        render_normals,
        render_distort,
        render_median,
        gs_contrib_sum,
        gs_contrib_count,
    )


TSSPOST = "_textured_sigmoids"
tss_fns = {
    "bilinear_bwd2": (f"{PRE}fwd{TSSPOST}", f"{PRE}bwd{TSSPOST}"),
    "anisotropic_bilinear": (
        f"{PRE}fwd_aniso_bilinear{TSSPOST}",
        f"{PRE}bwd_aniso_bilinear{TSSPOST}",
    ),
}


def rasterize_to_pixels_textured_sigmoids(
    means2d: Tensor,
    steepnesses: Tensor,
    ray_transforms: Tensor,
    colors: Tensor,
    opacities: Tensor,
    textures: Tensor,
    texture_range_x: float,
    texture_range_y: float,
    normals: Tensor,
    densify: Tensor,
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,
    flatten_ids: Tensor,
    backgrounds: Optional[Tensor] = None,
    masks: Optional[Tensor] = None,
    packed: bool = False,
    absgrad: bool = False,
    distloss: bool = False,
    gs_contrib_threshold: float = 0.0,
    filtering: Filtering = "bilinear",
    s_weight: float = 1.0,
) -> Tuple[Tensor, ...]:
    """Rasterize Textured Sigmoids (smooth-step 2DSS with bilinear texture) to pixels.

    Args:
        means2d: Projected Gaussian means. [C, N, 2] or [nnz, 2] if packed.
        steepnesses: Per-gaussian smooth-step steepness. [C, N] or [nnz] if packed.
        ray_transforms: Ray-splat transform matrices. [C, N, 3, 3] or [nnz, 3, 3] if packed.
        colors: Gaussian colors or ND features. [C, N, channels] or [nnz, channels] if packed.
        opacities: Gaussian opacities. [C, N] or [nnz] if packed.
        textures: Gaussian 2D textures. [N, th, tw, 4]. Packed not supported.
        normals: Normals in camera space. [C, N, 3] or [nnz, 3] if packed.
        densify: Dummy gradient buffer for densification. [C, N, 2] or [nnz, 2] if packed.
        image_width: Image width.
        image_height: Image height.
        tile_size: Tile size.
        isect_offsets: Intersection offsets from `isect_offset_encode()`. [C, tile_height, tile_width]
        flatten_ids: Flattened intersection indices from `isect_tiles()`. [n_isects]
        backgrounds: Background colors. [C, channels]. Default: None.
        masks: Optional tile mask. [C, tile_height, tile_width]. Default: None.
        packed: Whether inputs are packed. Default: False.
        absgrad: Compute absolute gradients for means2d. Default: False.
        distloss: Enable distortion loss. Default: False.
        gs_contrib_threshold: Minimum alpha to count as a contribution. Default: 0.0.

    Returns:
        - **render_colors**    [C, image_height, image_width, channels]
        - **render_alphas**    [C, image_height, image_width, 1]
        - **render_normals**   [C, image_height, image_width, 3]
        - **render_distort**   [C, image_height, image_width, 1]
        - **render_median**    [C, image_height, image_width, 1]
        - **gs_contrib_sum**   [N]
        - **gs_contrib_count** [N]
    """
    C = isect_offsets.size(0)
    device = means2d.device
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.shape
        assert steepnesses.shape == (nnz,), steepnesses.shape
        assert ray_transforms.shape == (nnz, 3, 3), ray_transforms.shape
        assert colors.shape[0] == nnz, colors.shape
        assert opacities.shape == (nnz,), opacities.shape
    else:
        N = means2d.size(1)
        assert means2d.shape == (C, N, 2), means2d.shape
        assert steepnesses.shape == (C, N), steepnesses.shape
        assert ray_transforms.shape == (C, N, 3, 3), ray_transforms.shape
        assert colors.shape[:2] == (C, N), colors.shape
        assert opacities.shape == (C, N), opacities.shape
    if backgrounds is not None:
        assert backgrounds.shape == (C, colors.shape[-1]), backgrounds.shape
        backgrounds = backgrounds.contiguous()

    channels = colors.shape[-1]
    if channels > 512 or channels == 0:
        raise ValueError(f"Unsupported number of color channels: {channels}")
    if channels not in (1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512):
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        colors = torch.cat(
            [colors, torch.empty(*colors.shape[:-1], padded_channels, device=device)],
            dim=-1,
        )
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.empty(
                        *backgrounds.shape[:-1], padded_channels, device=device
                    ),
                ],
                dim=-1,
            )
    else:
        padded_channels = 0
    tile_height, tile_width = isect_offsets.shape[1:3]
    assert (
        tile_height * tile_size >= image_height
    ), f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
    assert (
        tile_width * tile_size >= image_width
    ), f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"

    if filtering in tss_fns:
        (
            render_colors,
            render_alphas,
            render_normals,
            render_distort,
            render_median,
            gs_contrib_sum,
            gs_contrib_count,
        ) = _RasterizeToPixelsTexturedSigmoids.apply(
            tss_fns[filtering][0],
            tss_fns[filtering][1],
            means2d.contiguous(),
            steepnesses.contiguous(),
            ray_transforms.contiguous(),
            colors.contiguous(),
            opacities.contiguous(),
            textures.contiguous(),
            texture_range_x,
            texture_range_y,
            normals.contiguous(),
            densify.contiguous(),
            backgrounds,
            masks,
            image_width,
            image_height,
            tile_size,
            isect_offsets.contiguous(),
            flatten_ids.contiguous(),
            absgrad,
            distloss,
            gs_contrib_threshold,
            s_weight,
        )
    else:
        raise Exception(f"Unsupported filter type {filtering}")

    if padded_channels > 0:
        render_colors = render_colors[..., :-padded_channels]

    return (
        render_colors,
        render_alphas,
        render_normals,
        render_distort,
        render_median,
        gs_contrib_sum,
        gs_contrib_count,
    )


@torch.no_grad()
def rasterize_to_samples(
    means2d: Tensor,
    ray_transforms: Tensor,
    opacities: Tensor,
    masks: Tensor,
    width: int,
    height: int,
    tile_size: int,
    isect_offsets: Tensor,
    flatten_ids: Tensor,
    num_texture_samples: int,
    opac_threshold: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    (sample_counts, sample_gaussian_ids, texture_inputs) = _make_lazy_cuda_func(
        "rasterize_to_samples_fwd_textured_gaussians"
    )(
        means2d,
        ray_transforms,
        opacities,
        masks,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        num_texture_samples,
        opac_threshold,
    )
    return sample_counts, sample_gaussian_ids, texture_inputs


def rasterize_to_world_samples(
    means2d: Tensor,
    ray_transforms: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    opacities: Tensor,
    masks: Tensor,
    width: int,
    height: int,
    tile_size: int,
    isect_offsets: Tensor,
    flatten_ids: Tensor,
    num_texture_samples: int,
    opac_threshold: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    (sample_counts, sample_gaussian_ids, texture_inputs) = _make_lazy_cuda_func(
        "rasterize_to_samples_world_fwd_textured_gaussians"
    )(
        means2d,
        ray_transforms,
        viewmats,
        Ks,
        opacities,
        masks,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        num_texture_samples,
        opac_threshold,
    )
    return sample_counts, sample_gaussian_ids, texture_inputs


def rasterize_to_world_and_view_samples(
    means2d: Tensor,
    ray_transforms: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    opacities: Tensor,
    masks: Tensor,
    width: int,
    height: int,
    tile_size: int,
    isect_offsets: Tensor,
    flatten_ids: Tensor,
    num_texture_samples: int,
    opac_threshold: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    (sample_counts, sample_gaussian_ids, texture_inputs) = _make_lazy_cuda_func(
        "rasterize_to_samples_world_and_view_fwd_textured_gaussians"
    )(
        means2d,
        ray_transforms,
        viewmats,
        Ks,
        opacities,
        masks,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        num_texture_samples,
        opac_threshold,
    )
    return sample_counts, sample_gaussian_ids, texture_inputs


def rasterize_to_pixels_implicit_textured_gaussians(
    means2d: Tensor,
    ray_transforms: Tensor,
    colors: Tensor,
    opacities: Tensor,
    textures: torch.nn.Module,
    normals: Tensor,
    densify: Tensor,
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,
    flatten_ids: Tensor,
    backgrounds: Optional[Tensor] = None,
    masks: Optional[Tensor] = None,
    packed: bool = False,
    absgrad: bool = False,
    distloss: bool = False,
    gs_contrib_threshold: float = 0.0,
    num_texture_samples: int = 10,
    sample_alpha_threshold: float = 0.1,
    base_color_factor: float = 0.0,
    texture_batch_size: Optional[int] = None,
    texture_grad_method: TextureGrads = "checkpoint",
    texture_input_type: TextureInputType = "gaussian",
    viewmats: Optional[Tensor] = None,
    Ks: Optional[Tensor] = None,
    coord_center: Optional[Tensor] = None,
    coord_scale: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Rasterize Textured Gaussians to pixels.

    Args:
        means2d: Projected Gaussian means. [C, N, 2] if packed is False, [nnz, 2] if packed is True.
        ray_transforms: transformation matrices that transforms xy-planes in pixel spaces into splat coordinates. [C, N, 3, 3] if packed is False, [nnz, channels] if packed is True.
        colors: Gaussian colors or ND features. [C, N, channels] if packed is False, [nnz, channels] if packed is True.
        opacities: Gaussian opacities that support per-view values. [C, N] if packed is False, [nnz] if packed is True.
        textures: Gaussian 2D textures in the shape of [N, th, tw, 4]. packed not suported.
        normals: The normals in camera space. [C, N, 3] if packed is False, [nnz, 3] if packed is True.
        densify: Dummy variable to keep track of gradient for densification. [C, N, 2] if packed, [nnz, 3] if packed is True.
        tile_size: Tile size.
        isect_offsets: Intersection offsets outputs from `isect_offset_encode()`. [C, tile_height, tile_width]
        flatten_ids: The global flatten indices in [C * N] or [nnz] from  `isect_tiles()`. [n_isects]
        backgrounds: Background colors. [C, channels]. Default: None.
        masks: Optional tile mask to skip rendering GS to masked tiles. [C, tile_height, tile_width]. Default: None.
        packed: If True, the input tensors are expected to be packed with shape [nnz, ...]. Default: False.
        absgrad: If True, the backward pass will compute a `.absgrad` attribute for `means2d`. Default: False.

        gs_contrib_threshold: The threshold for Gaussian opacity contribution; intersection counted if above this threshold. Default: 0.0.

    Returns:
        A tuple:

        - **Rendered colors**.      [C, image_height, image_width, channels]
        - **Rendered alphas**.      [C, image_height, image_width, 1]
        - **Rendered normals**.     [C, image_height, image_width, 3]
        - **Rendered distortion**.  [C, image_height, image_width, 1]
        - **Rendered median depth**.[C, image_height, image_width, 1]
        - **Gaussian opacity contribution (sum)**. [N,]
        - **Gaussian opacity contribution (count)**. [N,]

    """
    C = isect_offsets.size(0)
    device = means2d.device
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.shape
        assert ray_transforms.shape == (nnz, 3, 3), ray_transforms.shape
        assert colors.shape[0] == nnz, colors.shape
        assert opacities.shape == (nnz,), opacities.shape
    else:
        N = means2d.size(1)
        assert means2d.shape == (C, N, 2), means2d.shape
        assert ray_transforms.shape == (C, N, 3, 3), ray_transforms.shape
        assert colors.shape[:2] == (C, N), colors.shape
        assert opacities.shape == (C, N), opacities.shape
    if backgrounds is not None:
        assert backgrounds.shape == (C, colors.shape[-1]), backgrounds.shape
        backgrounds = backgrounds.contiguous()

    # Pad the channels to the nearest supported number if necessary
    channels = colors.shape[-1]
    if channels > 512 or channels == 0:
        # TODO: maybe worth to support zero channels?
        raise ValueError(f"Unsupported number of color channels: {channels}")
    if channels not in (1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512):
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        colors = torch.cat(
            [colors, torch.empty(*colors.shape[:-1], padded_channels, device=device)],
            dim=-1,
        )
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.empty(
                        *backgrounds.shape[:-1], padded_channels, device=device
                    ),
                ],
                dim=-1,
            )
    else:
        padded_channels = 0
    tile_height, tile_width = isect_offsets.shape[1:3]
    assert (
        tile_height * tile_size >= image_height
    ), f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
    assert (
        tile_width * tile_size >= image_width
    ), f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"

    input_dim = TEXTURE_INPUT_SIZES[texture_input_type]
    match texture_input_type:
        case "gaussian":
            sample_counts, sample_gaussian_ids, texture_inputs = rasterize_to_samples(
                means2d.contiguous(),
                ray_transforms.contiguous(),
                opacities.contiguous(),
                masks,
                image_width,
                image_height,
                tile_size,
                isect_offsets.contiguous(),
                flatten_ids.contiguous(),
                num_texture_samples,
                sample_alpha_threshold,
            )
        case "world":
            sample_counts, sample_gaussian_ids, texture_inputs = (
                rasterize_to_world_samples(
                    means2d.contiguous(),
                    ray_transforms.contiguous(),
                    viewmats.contiguous(),
                    Ks.contiguous(),
                    opacities.contiguous(),
                    masks,
                    image_width,
                    image_height,
                    tile_size,
                    isect_offsets.contiguous(),
                    flatten_ids.contiguous(),
                    num_texture_samples,
                    sample_alpha_threshold,
                )
            )
        case "world_and_view":
            sample_counts, sample_gaussian_ids, texture_inputs = (
                rasterize_to_world_and_view_samples(
                    means2d.contiguous(),
                    ray_transforms.contiguous(),
                    viewmats.contiguous(),
                    Ks.contiguous(),
                    opacities.contiguous(),
                    masks,
                    image_width,
                    image_height,
                    tile_size,
                    isect_offsets.contiguous(),
                    flatten_ids.contiguous(),
                    num_texture_samples,
                    sample_alpha_threshold,
                )
            )

    # Apply coordinate normalization to world-space positions if requested.
    # For "world_and_view", only the XYZ channels (first 3) are normalized;
    # view directions (last 3) are unit vectors and are left unchanged.
    if (
        coord_center is not None
        and coord_scale is not None
        and texture_input_type in ("world", "world_and_view")
    ):
        texture_inputs = texture_inputs.clone()
        texture_inputs[..., :3] = (texture_inputs[..., :3] - coord_center) / coord_scale

    texture_inputs = torch.reshape(
        texture_inputs,
        (C * image_height * image_width * num_texture_samples, input_dim),
    )

    match texture_grad_method:
        case "dev":
            if texture_batch_size is None:
                texture_outputs = textures(texture_inputs)
            else:
                outputs = []
                for i in range(0, texture_inputs.size(0), texture_batch_size):
                    outputs.append(textures(texture_inputs[i : i + texture_batch_size]))
                texture_outputs = torch.cat(outputs, dim=0)
        case "cpu":
            with torch.autograd.graph.save_on_cpu():
                if texture_batch_size is None:
                    texture_outputs = textures(texture_inputs)
                else:
                    outputs = []
                    for i in range(0, texture_inputs.size(0), texture_batch_size):
                        outputs.append(
                            textures(texture_inputs[i : i + texture_batch_size])
                        )
                    texture_outputs = torch.cat(outputs, dim=0)
        case "checkpoint":
            if texture_batch_size is None:
                texture_outputs: Tensor = torch.utils.checkpoint.checkpoint(
                    lambda x: textures(x), texture_inputs, use_reentrant=False
                )
            else:
                outputs = []
                for i in range(0, texture_inputs.size(0), texture_batch_size):
                    input_chunk = texture_inputs[i : i + texture_batch_size]
                    outputs.append(
                        torch.utils.checkpoint.checkpoint(
                            lambda x: textures(x), input_chunk, use_reentrant=False
                        )
                    )
                texture_outputs = torch.cat(outputs, dim=0)

    C = isect_offsets.shape[0]
    COLOR_DIM = colors.shape[-1]
    texture_outputs = torch.reshape(
        texture_outputs, (C, image_height, image_width, num_texture_samples, COLOR_DIM)
    )

    (
        render_colors,
        render_alphas,
        render_normals,
        render_distort,
        render_median,
        gs_contrib_sum,  # added
        gs_contrib_count,  # added
    ) = _RasterizeToPixelsImplicitTexturedGaussians.apply(
        means2d.contiguous(),
        ray_transforms.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        normals.contiguous(),
        densify.contiguous(),
        backgrounds,
        masks,
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
        sample_counts.contiguous(),
        sample_gaussian_ids.contiguous(),
        texture_outputs.contiguous(),
        absgrad,
        distloss,
        gs_contrib_threshold,
        num_texture_samples,
        sample_alpha_threshold,
        base_color_factor,
    )

    if padded_channels > 0:
        render_colors = render_colors[..., :-padded_channels]

    return (
        render_colors,
        render_alphas,
        render_normals,
        render_distort,
        render_median,
        gs_contrib_sum,
        gs_contrib_count,
    )


def rasterize_to_pixels_dct_textured_gaussians(
    means2d: Tensor,
    ray_transforms: Tensor,
    colors: Tensor,
    opacities: Tensor,
    textures: Tensor,
    texture_range_x: float,
    texture_range_y: float,
    normals: Tensor,
    densify: Tensor,
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,
    flatten_ids: Tensor,
    backgrounds: Optional[Tensor] = None,
    masks: Optional[Tensor] = None,
    packed: bool = False,
    absgrad: bool = False,
    distloss: bool = False,
    gs_contrib_threshold: float = 0.0,  # added
) -> Tuple[Tensor, Tensor]:
    """Rasterize DCT Textured Gaussians to pixels.

    Args:
        means2d: Projected Gaussian means. [C, N, 2] if packed is False, [nnz, 2] if packed is True.
        ray_transforms: transformation matrices that transforms xy-planes in pixel spaces into splat coordinates. [C, N, 3, 3] if packed is False, [nnz, channels] if packed is True.
        colors: Gaussian colors or ND features. [C, N, channels] if packed is False, [nnz, channels] if packed is True.
        opacities: Gaussian opacities that support per-view values. [C, N] if packed is False, [nnz] if packed is True.
        textures: Gaussian 2D textures in the shape of [N, th, tw, 4]. packed not suported.
        normals: The normals in camera space. [C, N, 3] if packed is False, [nnz, 3] if packed is True.
        densify: Dummy variable to keep track of gradient for densification. [C, N, 2] if packed, [nnz, 3] if packed is True.
        tile_size: Tile size.
        isect_offsets: Intersection offsets outputs from `isect_offset_encode()`. [C, tile_height, tile_width]
        flatten_ids: The global flatten indices in [C * N] or [nnz] from  `isect_tiles()`. [n_isects]
        backgrounds: Background colors. [C, channels]. Default: None.
        masks: Optional tile mask to skip rendering GS to masked tiles. [C, tile_height, tile_width]. Default: None.
        packed: If True, the input tensors are expected to be packed with shape [nnz, ...]. Default: False.
        absgrad: If True, the backward pass will compute a `.absgrad` attribute for `means2d`. Default: False.

        gs_contrib_threshold: The threshold for Gaussian opacity contribution; intersection counted if above this threshold. Default: 0.0.

    Returns:
        A tuple:

        - **Rendered colors**.      [C, image_height, image_width, channels]
        - **Rendered alphas**.      [C, image_height, image_width, 1]
        - **Rendered normals**.     [C, image_height, image_width, 3]
        - **Rendered distortion**.  [C, image_height, image_width, 1]
        - **Rendered median depth**.[C, image_height, image_width, 1]
        - **Gaussian opacity contribution (sum)**. [N,]
        - **Gaussian opacity contribution (count)**. [N,]

    """
    C = isect_offsets.size(0)
    device = means2d.device
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.shape
        assert ray_transforms.shape == (nnz, 3, 3), ray_transforms.shape
        assert colors.shape[0] == nnz, colors.shape
        assert opacities.shape == (nnz,), opacities.shape
    else:
        N = means2d.size(1)
        assert means2d.shape == (C, N, 2), means2d.shape
        assert ray_transforms.shape == (C, N, 3, 3), ray_transforms.shape
        assert colors.shape[:2] == (C, N), colors.shape
        assert opacities.shape == (C, N), opacities.shape
    if backgrounds is not None:
        assert backgrounds.shape == (C, colors.shape[-1]), backgrounds.shape
        backgrounds = backgrounds.contiguous()

    # Pad the channels to the nearest supported number if necessary
    channels = colors.shape[-1]
    if channels > 512 or channels == 0:
        # TODO: maybe worth to support zero channels?
        raise ValueError(f"Unsupported number of color channels: {channels}")
    if channels not in (1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512):
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        colors = torch.cat(
            [colors, torch.empty(*colors.shape[:-1], padded_channels, device=device)],
            dim=-1,
        )
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.empty(
                        *backgrounds.shape[:-1], padded_channels, device=device
                    ),
                ],
                dim=-1,
            )
    else:
        padded_channels = 0
    tile_height, tile_width = isect_offsets.shape[1:3]
    assert (
        tile_height * tile_size >= image_height
    ), f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
    assert (
        tile_width * tile_size >= image_width
    ), f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"

    (
        render_colors,
        render_alphas,
        render_normals,
        render_distort,
        render_median,
        gs_contrib_sum,  # added
        gs_contrib_count,  # added
    ) = _RasterizeToPixelsDctTexturedGaussians.apply(
        means2d.contiguous(),
        ray_transforms.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        textures.contiguous(),
        texture_range_x,
        texture_range_y,
        normals.contiguous(),
        densify.contiguous(),
        backgrounds,
        masks,
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
        absgrad,
        distloss,
        gs_contrib_threshold,  # added
    )

    if padded_channels > 0:
        render_colors = render_colors[..., :-padded_channels]

    return (
        render_colors,
        render_alphas,
        render_normals,
        render_distort,
        render_median,
        gs_contrib_sum,
        gs_contrib_count,
    )


@torch.no_grad()
def rasterize_to_indices_in_range_2dgs(
    range_start: int,
    range_end: int,
    transmittances: Tensor,
    means2d: Tensor,
    ray_transforms: Tensor,
    opacities: Tensor,
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,
    flatten_ids: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Rasterizes a batch of Gaussians to images but only returns the indices.

    .. note::

        This function supports iterative rasterization, in which each call of this function
        will rasterize a batch of Gaussians from near to far, defined by `[range_start, range_end)`.
        If a one-step full rasterization is desired, set `range_start` to 0 and `range_end` to a really
        large number, e.g, 1e10.

    Args:
        range_start: The start batch of Gaussians to be rasterized (inclusive).
        range_end: The end batch of Gaussians to be rasterized (exclusive).
        transmittances: Currently transmittances. [C, image_height, image_width]
        means2d: Projected Gaussian means. [C, N, 2]
        ray_transforms: transformation matrices that transforms xy-planes in pixel spaces into splat coordinates. [C, N, 3, 3]
        opacities: Gaussian opacities that support per-view values. [C, N]
        image_width: Image width.
        image_height: Image height.
        tile_size: Tile size.
        isect_offsets: Intersection offsets outputs from `isect_offset_encode()`. [C, tile_height, tile_width]
        flatten_ids: The global flatten indices in [C * N] from  `isect_tiles()`. [n_isects]

    Returns:
        A tuple:

        - **Gaussian ids**. Gaussian ids for the pixel intersection. A flattened list of shape [M].
        - **Pixel ids**. pixel indices (row-major). A flattened list of shape [M].
        - **Camera ids**. Camera indices. A flattened list of shape [M].
    """

    C, N, _ = means2d.shape
    assert ray_transforms.shape == (C, N, 3, 3), ray_transforms.shape
    assert opacities.shape == (C, N), opacities.shape
    assert isect_offsets.shape[0] == C, isect_offsets.shape

    tile_height, tile_width = isect_offsets.shape[1:3]
    assert (
        tile_height * tile_size >= image_height
    ), f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
    assert (
        tile_width * tile_size >= image_width
    ), f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"

    out_gauss_ids, out_indices = _make_lazy_cuda_func(
        "rasterize_to_indices_in_range_2dgs"
    )(
        range_start,
        range_end,
        transmittances.contiguous(),
        means2d.contiguous(),
        ray_transforms.contiguous(),
        opacities.contiguous(),
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
    )
    out_pixel_ids = out_indices % (image_width * image_height)
    out_camera_ids = out_indices // (image_width * image_height)
    return out_gauss_ids, out_pixel_ids, out_camera_ids


class _RasterizeToPixels2DGS(torch.autograd.Function):
    """Rasterize gaussians 2DGS"""

    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,
        ray_transforms: Tensor,
        colors: Tensor,
        opacities: Tensor,
        normals: Tensor,
        densify: Tensor,
        backgrounds: Tensor,
        masks: Tensor,
        width: int,
        height: int,
        tile_size: int,
        isect_offsets: Tensor,
        flatten_ids: Tensor,
        absgrad: bool,
        distloss: bool,
        gs_contrib_threshold: float,  # added
    ) -> Tuple[Tensor, Tensor]:
        (
            render_colors,
            render_alphas,
            render_normals,
            render_distort,
            render_median,
            last_ids,
            median_ids,
            gs_contrib_sum,  # added
            gs_contrib_count,  # added
        ) = _make_lazy_cuda_func("rasterize_to_pixels_fwd_2dgs")(
            means2d,
            ray_transforms,
            colors,
            opacities,
            normals,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            gs_contrib_threshold,  # added
        )

        ctx.save_for_backward(
            means2d,
            ray_transforms,
            colors,
            opacities,
            normals,
            densify,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
        )
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size
        ctx.absgrad = absgrad
        ctx.distloss = distloss

        # double to float
        render_alphas = render_alphas.float()
        return (
            render_colors,
            render_alphas,
            render_normals,
            render_distort,
            render_median,
            gs_contrib_sum,
            gs_contrib_count,
        )

    @staticmethod
    def backward(
        ctx,
        v_render_colors: Tensor,
        v_render_alphas: Tensor,
        v_render_normals: Tensor,
        v_render_distort: Tensor,
        v_render_median: Tensor,
        v_gs_contrib_sum: Tensor,  # added
        v_gs_contrib_count: Tensor,  # added
    ):

        (
            means2d,
            ray_transforms,
            colors,
            opacities,
            normals,
            densify,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        tile_size = ctx.tile_size
        absgrad = ctx.absgrad

        (
            v_means2d_abs,
            v_means2d,
            v_ray_transforms,
            v_colors,
            v_opacities,
            v_normals,
            v_densify,
        ) = _make_lazy_cuda_func("rasterize_to_pixels_bwd_2dgs")(
            means2d,
            ray_transforms,
            colors,
            opacities,
            normals,
            densify,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
            v_render_normals.contiguous(),
            v_render_distort.contiguous(),
            v_render_median.contiguous(),
            absgrad,
        )
        torch.cuda.synchronize()
        if absgrad:
            means2d.absgrad = v_means2d_abs

        if ctx.needs_input_grad[6]:
            v_backgrounds = (v_render_colors * (1.0 - render_alphas).float()).sum(
                dim=(1, 2)
            )
        else:
            v_backgrounds = None

        return (
            v_means2d,
            v_ray_transforms,
            v_colors,
            v_opacities,
            v_normals,
            v_densify,
            v_backgrounds,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,  # added for gs_contrib_threshold
        )


class _RasterizeToPixelsTexturedGaussians(torch.autograd.Function):
    """Rasterize gaussians Textured Gaussians"""

    @staticmethod
    def forward(
        ctx,
        fwd_fn: str,
        bwd_fn: str,
        means2d: Tensor,
        ray_transforms: Tensor,
        colors: Tensor,
        opacities: Tensor,
        textures: Tensor,
        texture_range_x: float,
        texture_range_y: float,
        normals: Tensor,
        densify: Tensor,
        backgrounds: Tensor,
        masks: Tensor,
        width: int,
        height: int,
        tile_size: int,
        isect_offsets: Tensor,
        flatten_ids: Tensor,
        absgrad: bool,
        distloss: bool,
        gs_contrib_threshold: float,  # added
        g_weight: float,
    ) -> Tuple[Tensor, Tensor]:
        (
            render_colors,
            render_alphas,
            render_normals,
            render_distort,
            render_median,
            last_ids,
            median_ids,
            gs_contrib_sum,  # added
            gs_contrib_count,  # added
        ) = _make_lazy_cuda_func(fwd_fn)(
            means2d,
            ray_transforms,
            colors,
            opacities,
            textures,
            texture_range_x,
            texture_range_y,
            normals,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            gs_contrib_threshold,  # added
            g_weight,
        )

        ctx.save_for_backward(
            means2d,
            ray_transforms,
            colors,
            opacities,
            textures,
            normals,
            densify,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
        )
        ctx.bwd_fn = bwd_fn
        ctx.texture_range_x = texture_range_x
        ctx.texture_range_y = texture_range_y
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size
        ctx.absgrad = absgrad
        ctx.distloss = distloss
        ctx.g_weight = g_weight

        # double to float
        render_alphas = render_alphas.float()
        return (
            render_colors,
            render_alphas,
            render_normals,
            render_distort,
            render_median,
            gs_contrib_sum,
            gs_contrib_count,
        )

    @staticmethod
    def backward(
        ctx,
        v_render_colors: Tensor,
        v_render_alphas: Tensor,
        v_render_normals: Tensor,
        v_render_distort: Tensor,
        v_render_median: Tensor,
        v_gs_contrib_sum: Tensor,  # added
        v_gs_contrib_count: Tensor,  # added
    ):

        (
            means2d,
            ray_transforms,
            colors,
            opacities,
            textures,
            normals,
            densify,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
        ) = ctx.saved_tensors
        bwd_fn = ctx.bwd_fn
        texture_range_x = ctx.texture_range_x
        texture_range_y = ctx.texture_range_y
        width = ctx.width
        height = ctx.height
        tile_size = ctx.tile_size
        absgrad = ctx.absgrad
        g_weight = ctx.g_weight

        (
            v_means2d_abs,
            v_means2d,
            v_ray_transforms,
            v_colors,
            v_opacities,
            v_textures,
            v_normals,
            v_densify,
        ) = _make_lazy_cuda_func(bwd_fn)(
            means2d,
            ray_transforms,
            colors,
            opacities,
            textures,
            texture_range_x,
            texture_range_y,
            normals,
            densify,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            g_weight,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
            v_render_normals.contiguous(),
            v_render_distort.contiguous(),
            v_render_median.contiguous(),
            absgrad,
        )
        torch.cuda.synchronize()
        if absgrad:
            means2d.absgrad = v_means2d_abs

        if ctx.needs_input_grad[7]:
            v_backgrounds = (v_render_colors * (1.0 - render_alphas).float()).sum(
                dim=(1, 2)
            )
        else:
            v_backgrounds = None

        return (
            None,
            None,
            v_means2d,
            v_ray_transforms,
            v_colors,
            v_opacities,
            v_textures,
            None,
            None,
            v_normals,
            v_densify,
            v_backgrounds,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,  # added for gs_contrib_threshold
            None,
        )


class _RasterizeToPixelsTexturedSigmoids(torch.autograd.Function):
    """Rasterize Textured Sigmoids (smooth-step 2DSS with bilinear texture) with bwd2 backward."""

    @staticmethod
    def forward(
        ctx,
        fwd_fn: str,
        bwd_fn: str,
        means2d: Tensor,
        steepnesses: Tensor,
        ray_transforms: Tensor,
        colors: Tensor,
        opacities: Tensor,
        textures: Tensor,
        texture_range_x: float,
        texture_range_y: float,
        normals: Tensor,
        densify: Tensor,
        backgrounds: Tensor,
        masks: Tensor,
        width: int,
        height: int,
        tile_size: int,
        isect_offsets: Tensor,
        flatten_ids: Tensor,
        absgrad: bool,
        distloss: bool,
        gs_contrib_threshold: float,
        s_weight: float,
    ) -> Tuple[Tensor, Tensor]:

        (
            render_colors,
            render_alphas,
            render_normals,
            render_distort,
            render_median,
            last_ids,
            median_ids,
            gs_contrib_sum,
            gs_contrib_count,
        ) = _make_lazy_cuda_func(fwd_fn)(
            means2d,
            steepnesses,
            ray_transforms,
            colors,
            opacities,
            textures,
            texture_range_x,
            texture_range_y,
            normals,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            gs_contrib_threshold,
            s_weight,
        )

        ctx.save_for_backward(
            means2d,
            steepnesses,
            ray_transforms,
            colors,
            opacities,
            textures,
            normals,
            densify,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
        )
        ctx.bwd_fn = bwd_fn
        ctx.texture_range_x = texture_range_x
        ctx.texture_range_y = texture_range_y
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size
        ctx.absgrad = absgrad
        ctx.distloss = distloss
        ctx.s_weight = s_weight

        render_alphas = render_alphas.float()
        return (
            render_colors,
            render_alphas,
            render_normals,
            render_distort,
            render_median,
            gs_contrib_sum,
            gs_contrib_count,
        )

    @staticmethod
    def backward(
        ctx,
        v_render_colors: Tensor,
        v_render_alphas: Tensor,
        v_render_normals: Tensor,
        v_render_distort: Tensor,
        v_render_median: Tensor,
        v_gs_contrib_sum: Tensor,
        v_gs_contrib_count: Tensor,
    ):
        (
            means2d,
            steepnesses,
            ray_transforms,
            colors,
            opacities,
            textures,
            normals,
            densify,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
        ) = ctx.saved_tensors
        bwd_fn = ctx.bwd_fn
        texture_range_x = ctx.texture_range_x
        texture_range_y = ctx.texture_range_y
        width = ctx.width
        height = ctx.height
        tile_size = ctx.tile_size
        absgrad = ctx.absgrad
        s_weight = ctx.s_weight

        (
            v_means2d_abs,
            v_means2d,
            v_steepnesses,
            v_ray_transforms,
            v_colors,
            v_opacities,
            v_textures,
            v_normals,
            v_densify,
        ) = _make_lazy_cuda_func(bwd_fn)(
            means2d,
            steepnesses,
            ray_transforms,
            colors,
            opacities,
            textures,
            texture_range_x,
            texture_range_y,
            normals,
            densify,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            s_weight,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
            v_render_normals.contiguous(),
            v_render_distort.contiguous(),
            v_render_median.contiguous(),
            absgrad,
        )
        torch.cuda.synchronize()
        if absgrad:
            means2d.absgrad = v_means2d_abs

        if ctx.needs_input_grad[8]:
            v_backgrounds = (v_render_colors * (1.0 - render_alphas).float()).sum(
                dim=(1, 2)
            )
        else:
            v_backgrounds = None

        return (
            None,
            None,
            v_means2d,
            v_steepnesses,
            v_ray_transforms,
            v_colors,
            v_opacities,
            v_textures,
            None,
            None,
            v_normals,
            v_densify,
            v_backgrounds,
            None,  # masks
            None,  # width
            None,  # height
            None,  # tile_size
            None,  # isect_offsets
            None,  # flatten_ids
            None,  # absgrad
            None,  # distloss
            None,  # gs_contrib_threshold
            None,  # s_weight
        )


class _RasterizeToPixelsMip2TexturedGaussians(torch.autograd.Function):
    """Rasterize gaussians Textured Gaussians"""

    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,
        ray_transforms: Tensor,
        colors: Tensor,
        opacities: Tensor,
        textures: Tensor,
        log_texture_res: int,
        texture_range_x: float,
        texture_range_y: float,
        normals: Tensor,
        densify: Tensor,
        backgrounds: Tensor,
        masks: Tensor,
        width: int,
        height: int,
        tile_size: int,
        isect_offsets: Tensor,
        flatten_ids: Tensor,
        absgrad: bool,
        distloss: bool,
        gs_contrib_threshold: float,
        g_weight: float,
    ) -> Tuple[Tensor, Tensor]:
        (
            render_colors,
            render_alphas,
            render_normals,
            render_distort,
            render_median,
            last_ids,
            median_ids,
            gs_contrib_sum,
            gs_contrib_count,
        ) = _make_lazy_cuda_func("rasterize_to_pixels_fwd_mip2_textured_gaussians")(
            means2d,
            ray_transforms,
            colors,
            opacities,
            textures,
            log_texture_res,
            texture_range_x,
            texture_range_y,
            normals,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            gs_contrib_threshold,
            g_weight,
        )

        ctx.save_for_backward(
            means2d,
            ray_transforms,
            colors,
            opacities,
            textures,
            normals,
            densify,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
        )
        ctx.log_texture_res = log_texture_res
        ctx.texture_range_x = texture_range_x
        ctx.texture_range_y = texture_range_y
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size
        ctx.absgrad = absgrad
        ctx.distloss = distloss
        ctx.g_weight = g_weight

        # double to float
        render_alphas = render_alphas.float()
        return (
            render_colors,
            render_alphas,
            render_normals,
            render_distort,
            render_median,
            gs_contrib_sum,
            gs_contrib_count,
        )

    @staticmethod
    def backward(
        ctx,
        v_render_colors: Tensor,
        v_render_alphas: Tensor,
        v_render_normals: Tensor,
        v_render_distort: Tensor,
        v_render_median: Tensor,
        v_gs_contrib_sum: Tensor,
        v_gs_contrib_count: Tensor,
    ):

        (
            means2d,
            ray_transforms,
            colors,
            opacities,
            textures,
            normals,
            densify,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
        ) = ctx.saved_tensors
        log_texture_res = ctx.log_texture_res
        texture_range_x = ctx.texture_range_x
        texture_range_y = ctx.texture_range_y
        width = ctx.width
        height = ctx.height
        tile_size = ctx.tile_size
        absgrad = ctx.absgrad
        g_weight = ctx.g_weight

        (
            v_means2d_abs,
            v_means2d,
            v_ray_transforms,
            v_colors,
            v_opacities,
            v_textures,
            v_normals,
            v_densify,
        ) = _make_lazy_cuda_func("rasterize_to_pixels_bwd_mip2_textured_gaussians")(
            means2d,
            ray_transforms,
            colors,
            opacities,
            textures,
            log_texture_res,
            texture_range_x,
            texture_range_y,
            normals,
            densify,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            g_weight,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
            v_render_normals.contiguous(),
            v_render_distort.contiguous(),
            v_render_median.contiguous(),
            absgrad,
        )
        torch.cuda.synchronize()
        if absgrad:
            means2d.absgrad = v_means2d_abs

        if ctx.needs_input_grad[8]:
            v_backgrounds = (v_render_colors * (1.0 - render_alphas).float()).sum(
                dim=(1, 2)
            )
        else:
            v_backgrounds = None

        return (
            v_means2d,  # means2d
            v_ray_transforms,  # ray_transforms
            v_colors,  # colors
            v_opacities,  # opacities
            v_textures,  # textures
            None,  # log_texture_res
            None,
            None,
            v_normals,  # normals
            v_densify,  # densify
            v_backgrounds,  # backgrounds
            None,  # masks
            None,  # width
            None,  # height
            None,  # tile_size
            None,  # isect_offsets
            None,  # flatten_ids
            None,  # absgrad
            None,  # distloss
            None,  # gs_contrib_threshold
            None,  # g_weight
        )


class _RasterizeToPixelsImplicitTexturedGaussians(torch.autograd.Function):
    """Rasterize gaussians implicit Textured Gaussians"""

    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,
        ray_transforms: Tensor,
        colors: Tensor,
        opacities: Tensor,
        normals: Tensor,
        densify: Tensor,
        backgrounds: Tensor,
        masks: Tensor,
        width: int,
        height: int,
        tile_size: int,
        isect_offsets: Tensor,
        flatten_ids: Tensor,
        sample_counts: Tensor,
        sample_gaussian_ids: Tensor,
        texture_outputs: Tensor,
        absgrad: bool,
        distloss: bool,
        gs_contrib_threshold: float,
        num_texture_samples: int,
        alpha_threshold: float,
        base_color_factor: float,
    ) -> Tuple[Tensor, Tensor]:
        (
            render_colors,
            render_alphas,
            render_normals,
            render_distort,
            render_median,
            last_ids,
            median_ids,
            gs_contrib_sum,
            gs_contrib_count,
        ) = _make_lazy_cuda_func("rasterize_to_pixels_fwd_implicit_textured_gaussians")(
            means2d,
            ray_transforms,
            colors,
            opacities,
            normals,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            texture_outputs,
            num_texture_samples,
            alpha_threshold,
            base_color_factor,
            gs_contrib_threshold,
        )

        ctx.save_for_backward(
            means2d,
            ray_transforms,
            colors,
            opacities,
            normals,
            densify,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            texture_outputs,
            sample_counts,
            sample_gaussian_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
        )
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size
        ctx.num_texture_samples = num_texture_samples
        ctx.alpha_threshold = alpha_threshold
        ctx.base_color_factor = base_color_factor
        ctx.absgrad = absgrad
        ctx.distloss = distloss

        # double to float
        render_alphas = render_alphas.float()
        return (
            render_colors,
            render_alphas,
            render_normals,
            render_distort,
            render_median,
            gs_contrib_sum,
            gs_contrib_count,
        )

    @staticmethod
    def backward(
        ctx,
        v_render_colors: Tensor,
        v_render_alphas: Tensor,
        v_render_normals: Tensor,
        v_render_distort: Tensor,
        v_render_median: Tensor,
        v_gs_contrib_sum: Tensor,  # added
        v_gs_contrib_count: Tensor,  # added
    ):

        (
            means2d,
            ray_transforms,
            colors,
            opacities,
            normals,
            densify,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            texture_outputs,
            sample_counts,
            sample_gaussian_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        tile_size = ctx.tile_size
        num_texture_samples = ctx.num_texture_samples
        alpha_threshold = ctx.alpha_threshold
        base_color_factor = ctx.base_color_factor
        absgrad = ctx.absgrad

        (
            v_means2d_abs,
            v_means2d,
            v_ray_transforms,
            v_colors,
            v_opacities,
            v_texture_outputs,
            v_normals,
            v_densify,
        ) = _make_lazy_cuda_func("rasterize_to_pixels_bwd_implicit_textured_gaussians")(
            means2d,
            ray_transforms,
            colors,
            opacities,
            normals,
            densify,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            texture_outputs,
            sample_counts.clone(),
            sample_gaussian_ids,
            num_texture_samples,
            alpha_threshold,
            base_color_factor,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
            v_render_normals.contiguous(),
            v_render_distort.contiguous(),
            v_render_median.contiguous(),
            absgrad,
        )
        torch.cuda.synchronize()
        if absgrad:
            means2d.absgrad = v_means2d_abs

        if ctx.needs_input_grad[6]:
            v_backgrounds = (v_render_colors * (1.0 - render_alphas).float()).sum(
                dim=(1, 2)
            )
        else:
            v_backgrounds = None

        return (
            v_means2d,
            v_ray_transforms,
            v_colors,
            v_opacities,
            v_normals,
            v_densify,
            v_backgrounds,
            None,  # masks
            None,  # width
            None,  # height
            None,  # tile size
            None,  # isect_offsets
            None,  # flatten_ids
            None,  # sample_counts
            None,  # sample_gaussian_ids
            v_texture_outputs,
            None,  # absgrad
            None,  # distloss
            None,  # gs_contrib_threshold
            None,  # num_texture_samples
            None,  # opac_threshold
            None,  # base_color_factor
        )


class _RasterizeToPixelsDctTexturedGaussians(torch.autograd.Function):
    """Rasterize gaussians DCT Textured Gaussians"""

    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,
        ray_transforms: Tensor,
        colors: Tensor,
        opacities: Tensor,
        textures: Tensor,
        texture_range_x: float,
        texture_range_y: float,
        normals: Tensor,
        densify: Tensor,
        backgrounds: Tensor,
        masks: Tensor,
        width: int,
        height: int,
        tile_size: int,
        isect_offsets: Tensor,
        flatten_ids: Tensor,
        absgrad: bool,
        distloss: bool,
        gs_contrib_threshold: float,  # added
    ) -> Tuple[Tensor, Tensor]:
        (
            render_colors,
            render_alphas,
            render_normals,
            render_distort,
            render_median,
            last_ids,
            median_ids,
            gs_contrib_sum,  # added
            gs_contrib_count,  # added
        ) = _make_lazy_cuda_func("rasterize_to_pixels_fwd_dct_textured_gaussians")(
            means2d,
            ray_transforms,
            colors,
            opacities,
            textures,
            texture_range_x,
            texture_range_y,
            normals,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            gs_contrib_threshold,  # added
        )

        ctx.save_for_backward(
            means2d,
            ray_transforms,
            colors,
            opacities,
            textures,
            normals,
            densify,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
        )
        ctx.texture_range_x = texture_range_x
        ctx.texture_range_y = texture_range_y
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size
        ctx.absgrad = absgrad
        ctx.distloss = distloss

        # double to float
        render_alphas = render_alphas.float()
        return (
            render_colors,
            render_alphas,
            render_normals,
            render_distort,
            render_median,
            gs_contrib_sum,
            gs_contrib_count,
        )

    @staticmethod
    def backward(
        ctx,
        v_render_colors: Tensor,
        v_render_alphas: Tensor,
        v_render_normals: Tensor,
        v_render_distort: Tensor,
        v_render_median: Tensor,
        v_gs_contrib_sum: Tensor,  # added
        v_gs_contrib_count: Tensor,  # added
    ):

        (
            means2d,
            ray_transforms,
            colors,
            opacities,
            textures,
            normals,
            densify,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
        ) = ctx.saved_tensors
        texture_range_x = ctx.texture_range_x
        texture_range_y = ctx.texture_range_y
        width = ctx.width
        height = ctx.height
        tile_size = ctx.tile_size
        absgrad = ctx.absgrad

        (
            v_means2d_abs,
            v_means2d,
            v_ray_transforms,
            v_colors,
            v_opacities,
            v_textures,
            v_normals,
            v_densify,
        ) = _make_lazy_cuda_func("rasterize_to_pixels_bwd_dct_textured_gaussians")(
            means2d,
            ray_transforms,
            colors,
            opacities,
            textures,
            texture_range_x,
            texture_range_y,
            normals,
            densify,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
            v_render_normals.contiguous(),
            v_render_distort.contiguous(),
            v_render_median.contiguous(),
            absgrad,
        )
        torch.cuda.synchronize()
        if absgrad:
            means2d.absgrad = v_means2d_abs

        if ctx.needs_input_grad[7]:
            v_backgrounds = (v_render_colors * (1.0 - render_alphas).float()).sum(
                dim=(1, 2)
            )
        else:
            v_backgrounds = None

        return (
            v_means2d,
            v_ray_transforms,
            v_colors,
            v_opacities,
            v_textures,
            None,
            v_normals,
            v_densify,
            v_backgrounds,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,  # added for gs_contrib_threshold
        )


@torch.no_grad()
def rasterize_dct_textures(
    textures: Tensor,
    width: int,
    height: int,
    tile_size: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    colors = _make_lazy_cuda_func("rasterize_dct_textures")(
        textures,
        width,
        height,
        tile_size,
    )
    return colors


class _GenerateMipmap(torch.autograd.Function):
    """Generate mipmaps from finest-level textures."""

    @staticmethod
    def forward(
        ctx,
        textures: Tensor,
        log_texture_res: int,
        log_reduce: int,
        tile_size: int,
    ) -> Tensor:
        mip_textures = _make_lazy_cuda_func("generate_mipmap_fwd")(
            textures,
            log_texture_res,
            log_reduce,
            tile_size,
        )
        ctx.N = textures.shape[0]
        ctx.channels = textures.shape[-1]
        ctx.log_texture_res = log_texture_res
        ctx.log_reduce = log_reduce
        ctx.tile_size = tile_size
        return mip_textures

    @staticmethod
    def backward(ctx, v_mip_textures: Tensor):
        v_textures = _make_lazy_cuda_func("generate_mipmap_bwd")(
            ctx.N,
            ctx.channels,
            ctx.log_texture_res,
            ctx.log_reduce,
            ctx.tile_size,
            v_mip_textures.contiguous(),
        )
        return v_textures, None, None, None


def generate_mipmap(
    textures: Tensor,
    log_texture_res: int,
    log_reduce: int,
    tile_size: int,
) -> Tensor:
    return _GenerateMipmap.apply(textures, log_texture_res, log_reduce, tile_size)


class _RasterizeToPixels2DSS(torch.autograd.Function):
    """Rasterize gaussians with smooth step (2DSS)"""

    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,
        steepnesses: Tensor,
        ray_transforms: Tensor,
        colors: Tensor,
        opacities: Tensor,
        normals: Tensor,
        densify: Tensor,
        backgrounds: Tensor,
        masks: Tensor,
        width: int,
        height: int,
        tile_size: int,
        isect_offsets: Tensor,
        flatten_ids: Tensor,
        absgrad: bool,
        distloss: bool,
        gs_contrib_threshold: float,
    ) -> Tuple[Tensor, ...]:
        (
            render_colors,
            render_alphas,
            render_normals,
            render_distort,
            render_median,
            last_ids,
            median_ids,
            gs_contrib_sum,
            gs_contrib_count,
        ) = _make_lazy_cuda_func("rasterize_to_pixels_fwd_2dss")(
            means2d,
            steepnesses,
            ray_transforms,
            colors,
            opacities,
            normals,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            gs_contrib_threshold,
        )

        ctx.save_for_backward(
            means2d,
            steepnesses,
            ray_transforms,
            colors,
            opacities,
            normals,
            densify,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
        )
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size
        ctx.absgrad = absgrad
        ctx.distloss = distloss

        render_alphas = render_alphas.float()
        return (
            render_colors,
            render_alphas,
            render_normals,
            render_distort,
            render_median,
            gs_contrib_sum,
            gs_contrib_count,
        )

    @staticmethod
    def backward(
        ctx,
        v_render_colors: Tensor,
        v_render_alphas: Tensor,
        v_render_normals: Tensor,
        v_render_distort: Tensor,
        v_render_median: Tensor,
        v_gs_contrib_sum: Tensor,
        v_gs_contrib_count: Tensor,
    ):
        (
            means2d,
            steepnesses,
            ray_transforms,
            colors,
            opacities,
            normals,
            densify,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        tile_size = ctx.tile_size
        absgrad = ctx.absgrad

        (
            v_means2d_abs,
            v_means2d,
            v_steepnesses,
            v_ray_transforms,
            v_colors,
            v_opacities,
            v_normals,
            v_densify,
        ) = _make_lazy_cuda_func("rasterize_to_pixels_bwd_2dss")(
            means2d,
            steepnesses,
            ray_transforms,
            colors,
            opacities,
            normals,
            densify,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
            v_render_normals.contiguous(),
            v_render_distort.contiguous(),
            v_render_median.contiguous(),
            absgrad,
        )
        torch.cuda.synchronize()
        if absgrad:
            means2d.absgrad = v_means2d_abs

        if ctx.needs_input_grad[7]:
            v_backgrounds = (v_render_colors * (1.0 - render_alphas).float()).sum(
                dim=(1, 2)
            )
        else:
            v_backgrounds = None

        return (
            v_means2d,
            v_steepnesses,
            v_ray_transforms,
            v_colors,
            v_opacities,
            v_normals,
            v_densify,
            v_backgrounds,
            None,  # masks
            None,  # width
            None,  # height
            None,  # tile_size
            None,  # isect_offsets
            None,  # flatten_ids
            None,  # absgrad
            None,  # distloss
            None,  # gs_contrib_threshold
        )


def rasterize_to_pixels_2dss(
    means2d: Tensor,
    steepnesses: Tensor,
    ray_transforms: Tensor,
    colors: Tensor,
    opacities: Tensor,
    normals: Tensor,
    densify: Tensor,
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,
    flatten_ids: Tensor,
    backgrounds: Optional[Tensor] = None,
    masks: Optional[Tensor] = None,
    packed: bool = False,
    absgrad: bool = False,
    distloss: bool = False,
    gs_contrib_threshold: float = 0.0,
) -> Tuple[Tensor, ...]:
    """Rasterize 2D smooth-step Gaussians to pixels.

    Args:
        means2d: Projected Gaussian means. [C, N, 2] or [nnz, 2] if packed.
        steepnesses: Per-gaussian step steepness. [C, N] or [nnz] if packed.
        ray_transforms: Ray-splat transform matrices. [C, N, 3, 3] or [nnz, 3, 3] if packed.
        colors: Gaussian colors or ND features. [C, N, channels] or [nnz, channels] if packed.
        opacities: Gaussian opacities. [C, N] or [nnz] if packed.
        normals: Normals in camera space. [C, N, 3] or [nnz, 3] if packed.
        densify: Dummy gradient buffer for densification. [C, N, 2] or [nnz, 2] if packed.
        image_width: Image width.
        image_height: Image height.
        tile_size: Tile size.
        isect_offsets: Intersection offsets from `isect_offset_encode()`. [C, tile_height, tile_width]
        flatten_ids: Flattened intersection indices from `isect_tiles()`. [n_isects]
        backgrounds: Background colors. [C, channels]. Default: None.
        masks: Optional tile mask. [C, tile_height, tile_width]. Default: None.
        packed: Whether inputs are packed. Default: False.
        absgrad: Compute absolute gradients for means2d. Default: False.
        distloss: Enable distortion loss. Default: False.
        gs_contrib_threshold: Minimum alpha to count as a contribution. Default: 0.0.

    Returns:
        - **render_colors**  [C, image_height, image_width, channels]
        - **render_alphas**  [C, image_height, image_width, 1]
        - **render_normals** [C, image_height, image_width, 3]
        - **render_distort** [C, image_height, image_width, 1]
        - **render_median**  [C, image_height, image_width, 1]
        - **gs_contrib_sum** [N]
        - **gs_contrib_count** [N]
    """
    C = isect_offsets.size(0)
    device = means2d.device
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.shape
        assert steepnesses.shape == (nnz,), steepnesses.shape
        assert ray_transforms.shape == (nnz, 3, 3), ray_transforms.shape
        assert colors.shape[0] == nnz, colors.shape
        assert opacities.shape == (nnz,), opacities.shape
    else:
        N = means2d.size(1)
        assert means2d.shape == (C, N, 2), means2d.shape
        assert steepnesses.shape == (C, N), steepnesses.shape
        assert ray_transforms.shape == (C, N, 3, 3), ray_transforms.shape
        assert colors.shape[:2] == (C, N), colors.shape
        assert opacities.shape == (C, N), opacities.shape
    if backgrounds is not None:
        assert backgrounds.shape == (C, colors.shape[-1]), backgrounds.shape
        backgrounds = backgrounds.contiguous()

    channels = colors.shape[-1]
    if channels > 512 or channels == 0:
        raise ValueError(f"Unsupported number of color channels: {channels}")
    if channels not in (1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512):
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        colors = torch.cat(
            [colors, torch.empty(*colors.shape[:-1], padded_channels, device=device)],
            dim=-1,
        )
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.empty(
                        *backgrounds.shape[:-1], padded_channels, device=device
                    ),
                ],
                dim=-1,
            )
    else:
        padded_channels = 0

    tile_height, tile_width = isect_offsets.shape[1:3]
    assert tile_height * tile_size >= image_height
    assert tile_width * tile_size >= image_width

    (
        render_colors,
        render_alphas,
        render_normals,
        render_distort,
        render_median,
        gs_contrib_sum,
        gs_contrib_count,
    ) = _RasterizeToPixels2DSS.apply(
        means2d.contiguous(),
        steepnesses.contiguous(),
        ray_transforms.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        normals.contiguous(),
        densify.contiguous(),
        backgrounds,
        masks,
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
        absgrad,
        distloss,
        gs_contrib_threshold,
    )

    if padded_channels > 0:
        render_colors = render_colors[..., :-padded_channels]

    return (
        render_colors,
        render_alphas,
        render_normals,
        render_distort,
        render_median,
        gs_contrib_sum,
        gs_contrib_count,
    )


class _RasterizeToPixels2DGSS(torch.autograd.Function):
    """Rasterize gaussians with mixed Gaussian+smooth-step kernel (2DGSS)"""

    @staticmethod
    def forward(
        ctx,
        means2d: Tensor,
        steepnesses: Tensor,
        ray_transforms: Tensor,
        colors: Tensor,
        opacities: Tensor,
        normals: Tensor,
        densify: Tensor,
        backgrounds: Tensor,
        masks: Tensor,
        width: int,
        height: int,
        tile_size: int,
        isect_offsets: Tensor,
        flatten_ids: Tensor,
        absgrad: bool,
        distloss: bool,
        gs_contrib_threshold: float,
        s_weight: float,
    ) -> Tuple[Tensor, ...]:
        (
            render_colors,
            render_alphas,
            render_normals,
            render_distort,
            render_median,
            last_ids,
            median_ids,
            gs_contrib_sum,
            gs_contrib_count,
        ) = _make_lazy_cuda_func("rasterize_to_pixels_fwd_2dgss")(
            means2d,
            steepnesses,
            ray_transforms,
            colors,
            opacities,
            normals,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            gs_contrib_threshold,
            s_weight,
        )

        ctx.save_for_backward(
            means2d,
            steepnesses,
            ray_transforms,
            colors,
            opacities,
            normals,
            densify,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
        )
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size
        ctx.absgrad = absgrad
        ctx.distloss = distloss
        ctx.s_weight = s_weight

        render_alphas = render_alphas.float()
        return (
            render_colors,
            render_alphas,
            render_normals,
            render_distort,
            render_median,
            gs_contrib_sum,
            gs_contrib_count,
        )

    @staticmethod
    def backward(
        ctx,
        v_render_colors: Tensor,
        v_render_alphas: Tensor,
        v_render_normals: Tensor,
        v_render_distort: Tensor,
        v_render_median: Tensor,
        v_gs_contrib_sum: Tensor,
        v_gs_contrib_count: Tensor,
    ):
        (
            means2d,
            steepnesses,
            ray_transforms,
            colors,
            opacities,
            normals,
            densify,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
        ) = ctx.saved_tensors
        width = ctx.width
        height = ctx.height
        tile_size = ctx.tile_size
        absgrad = ctx.absgrad
        s_weight = ctx.s_weight

        (
            v_means2d_abs,
            v_means2d,
            v_steepnesses,
            v_ray_transforms,
            v_colors,
            v_opacities,
            v_normals,
            v_densify,
        ) = _make_lazy_cuda_func("rasterize_to_pixels_bwd_2dgss")(
            means2d,
            steepnesses,
            ray_transforms,
            colors,
            opacities,
            normals,
            densify,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            s_weight,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
            v_render_normals.contiguous(),
            v_render_distort.contiguous(),
            v_render_median.contiguous(),
            absgrad,
        )
        torch.cuda.synchronize()
        if absgrad:
            means2d.absgrad = v_means2d_abs

        if ctx.needs_input_grad[7]:
            v_backgrounds = (v_render_colors * (1.0 - render_alphas).float()).sum(
                dim=(1, 2)
            )
        else:
            v_backgrounds = None

        return (
            v_means2d,
            v_steepnesses,
            v_ray_transforms,
            v_colors,
            v_opacities,
            v_normals,
            v_densify,
            v_backgrounds,
            None,  # masks
            None,  # width
            None,  # height
            None,  # tile_size
            None,  # isect_offsets
            None,  # flatten_ids
            None,  # absgrad
            None,  # distloss
            None,  # gs_contrib_threshold
            None,  # s_weight
        )


def rasterize_to_pixels_2dgss(
    means2d: Tensor,
    steepnesses: Tensor,
    ray_transforms: Tensor,
    colors: Tensor,
    opacities: Tensor,
    normals: Tensor,
    densify: Tensor,
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,
    flatten_ids: Tensor,
    backgrounds: Optional[Tensor] = None,
    masks: Optional[Tensor] = None,
    packed: bool = False,
    absgrad: bool = False,
    distloss: bool = False,
    gs_contrib_threshold: float = 0.0,
    s_weight: float = 0.5,
) -> Tuple[Tensor, ...]:
    """Rasterize 2D Gaussian-Smooth-Step (2DGSS) splats to pixels.

    The per-pixel visibility weight is a blend of a Gaussian and a smooth-step
    kernel: ``vis = gaussian*(1-s_weight) + sigmoid*s_weight``.

    Args:
        means2d: Projected Gaussian means. [C, N, 2] or [nnz, 2] if packed.
        steepnesses: Per-gaussian step steepness. [C, N] or [nnz] if packed.
        ray_transforms: Ray-splat transform matrices. [C, N, 3, 3] or [nnz, 3, 3] if packed.
        colors: Gaussian colors or ND features. [C, N, channels] or [nnz, channels] if packed.
        opacities: Gaussian opacities. [C, N] or [nnz] if packed.
        normals: Normals in camera space. [C, N, 3] or [nnz, 3] if packed.
        densify: Dummy gradient buffer for densification. [C, N, 2] or [nnz, 2] if packed.
        image_width: Image width.
        image_height: Image height.
        tile_size: Tile size.
        isect_offsets: Intersection offsets from `isect_offset_encode()`. [C, tile_height, tile_width]
        flatten_ids: Flattened intersection indices from `isect_tiles()`. [n_isects]
        backgrounds: Background colors. [C, channels]. Default: None.
        masks: Optional tile mask. [C, tile_height, tile_width]. Default: None.
        packed: Whether inputs are packed. Default: False.
        absgrad: Compute absolute gradients for means2d. Default: False.
        distloss: Enable distortion loss. Default: False.
        gs_contrib_threshold: Minimum alpha to count as a contribution. Default: 0.0.
        s_weight: Blend weight for sigmoid kernel (0=pure Gaussian, 1=pure sigmoid). Default: 0.5.

    Returns:
        - **render_colors**    [C, image_height, image_width, channels]
        - **render_alphas**    [C, image_height, image_width, 1]
        - **render_normals**   [C, image_height, image_width, 3]
        - **render_distort**   [C, image_height, image_width, 1]
        - **render_median**    [C, image_height, image_width, 1]
        - **gs_contrib_sum**   [N]
        - **gs_contrib_count** [N]
    """
    C = isect_offsets.size(0)
    device = means2d.device
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.shape
        assert steepnesses.shape == (nnz,), steepnesses.shape
        assert ray_transforms.shape == (nnz, 3, 3), ray_transforms.shape
        assert colors.shape[0] == nnz, colors.shape
        assert opacities.shape == (nnz,), opacities.shape
    else:
        N = means2d.size(1)
        assert means2d.shape == (C, N, 2), means2d.shape
        assert steepnesses.shape == (C, N), steepnesses.shape
        assert ray_transforms.shape == (C, N, 3, 3), ray_transforms.shape
        assert colors.shape[:2] == (C, N), colors.shape
        assert opacities.shape == (C, N), opacities.shape
    if backgrounds is not None:
        assert backgrounds.shape == (C, colors.shape[-1]), backgrounds.shape
        backgrounds = backgrounds.contiguous()

    channels = colors.shape[-1]
    if channels > 512 or channels == 0:
        raise ValueError(f"Unsupported number of color channels: {channels}")
    if channels not in (1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512):
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        colors = torch.cat(
            [colors, torch.empty(*colors.shape[:-1], padded_channels, device=device)],
            dim=-1,
        )
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.empty(
                        *backgrounds.shape[:-1], padded_channels, device=device
                    ),
                ],
                dim=-1,
            )
    else:
        padded_channels = 0

    tile_height, tile_width = isect_offsets.shape[1:3]
    assert tile_height * tile_size >= image_height
    assert tile_width * tile_size >= image_width

    (
        render_colors,
        render_alphas,
        render_normals,
        render_distort,
        render_median,
        gs_contrib_sum,
        gs_contrib_count,
    ) = _RasterizeToPixels2DGSS.apply(
        means2d.contiguous(),
        steepnesses.contiguous(),
        ray_transforms.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        normals.contiguous(),
        densify.contiguous(),
        backgrounds,
        masks,
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
        absgrad,
        distloss,
        gs_contrib_threshold,
        s_weight,
    )

    if padded_channels > 0:
        render_colors = render_colors[..., :-padded_channels]

    return (
        render_colors,
        render_alphas,
        render_normals,
        render_distort,
        render_median,
        gs_contrib_sum,
        gs_contrib_count,
    )


class _RasterizeToPixelsTexturedGaussSigs(torch.autograd.Function):
    """Rasterize Textured GaussSig (blended Gaussian+smooth-step+base with bilinear texture)."""

    @staticmethod
    def forward(
        ctx,
        fwd_fn: str,
        bwd_fn: str,
        means2d: Tensor,
        steepnesses: Tensor,
        ray_transforms: Tensor,
        colors: Tensor,
        opacities: Tensor,
        textures: Tensor,
        texture_range_x: float,
        texture_range_y: float,
        normals: Tensor,
        densify: Tensor,
        backgrounds: Tensor,
        masks: Tensor,
        width: int,
        height: int,
        tile_size: int,
        isect_offsets: Tensor,
        flatten_ids: Tensor,
        absgrad: bool,
        distloss: bool,
        gs_contrib_threshold: float,
        g_weight: float,
        s_weight: float,
    ) -> Tuple[Tensor, ...]:
        (
            render_colors,
            render_alphas,
            render_normals,
            render_distort,
            render_median,
            last_ids,
            median_ids,
            gs_contrib_sum,
            gs_contrib_count,
        ) = _make_lazy_cuda_func(fwd_fn)(
            means2d,
            steepnesses,
            ray_transforms,
            colors,
            opacities,
            textures,
            texture_range_x,
            texture_range_y,
            normals,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            gs_contrib_threshold,
            g_weight,
            s_weight,
        )

        ctx.save_for_backward(
            means2d,
            steepnesses,
            ray_transforms,
            colors,
            opacities,
            textures,
            normals,
            densify,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
        )
        ctx.bwd_fn = bwd_fn
        ctx.texture_range_x = texture_range_x
        ctx.texture_range_y = texture_range_y
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size
        ctx.absgrad = absgrad
        ctx.distloss = distloss
        ctx.g_weight = g_weight
        ctx.s_weight = s_weight

        render_alphas = render_alphas.float()
        return (
            render_colors,
            render_alphas,
            render_normals,
            render_distort,
            render_median,
            gs_contrib_sum,
            gs_contrib_count,
        )

    @staticmethod
    def backward(
        ctx,
        v_render_colors: Tensor,
        v_render_alphas: Tensor,
        v_render_normals: Tensor,
        v_render_distort: Tensor,
        v_render_median: Tensor,
        v_gs_contrib_sum: Tensor,
        v_gs_contrib_count: Tensor,
    ):
        (
            means2d,
            steepnesses,
            ray_transforms,
            colors,
            opacities,
            textures,
            normals,
            densify,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
        ) = ctx.saved_tensors
        bwd_fn = ctx.bwd_fn
        texture_range_x = ctx.texture_range_x
        texture_range_y = ctx.texture_range_y
        width = ctx.width
        height = ctx.height
        tile_size = ctx.tile_size
        absgrad = ctx.absgrad
        g_weight = ctx.g_weight
        s_weight = ctx.s_weight

        (
            v_means2d_abs,
            v_means2d,
            v_steepnesses,
            v_ray_transforms,
            v_colors,
            v_opacities,
            v_textures,
            v_normals,
            v_densify,
        ) = _make_lazy_cuda_func(bwd_fn)(
            means2d,
            steepnesses,
            ray_transforms,
            colors,
            opacities,
            textures,
            texture_range_x,
            texture_range_y,
            normals,
            densify,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            g_weight,
            s_weight,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
            v_render_colors.contiguous(),
            v_render_alphas.contiguous(),
            v_render_normals.contiguous(),
            v_render_distort.contiguous(),
            v_render_median.contiguous(),
            absgrad,
        )
        torch.cuda.synchronize()
        if absgrad:
            means2d.absgrad = v_means2d_abs

        if ctx.needs_input_grad[8]:
            v_backgrounds = (v_render_colors * (1.0 - render_alphas).float()).sum(
                dim=(1, 2)
            )
        else:
            v_backgrounds = None

        return (
            None,
            None,
            v_means2d,
            v_steepnesses,
            v_ray_transforms,
            v_colors,
            v_opacities,
            v_textures,
            None,
            None,
            v_normals,
            v_densify,
            v_backgrounds,
            None,  # masks
            None,  # width
            None,  # height
            None,  # tile_size
            None,  # isect_offsets
            None,  # flatten_ids
            None,  # absgrad
            None,  # distloss
            None,  # gs_contrib_threshold
            None,  # g_weight
            None,  # s_weight
        )


TGSSPOST = "_textured_gausssigs"
tgss_fns = {
    "bilinear_bwd2": (f"{PRE}fwd{TGSSPOST}", f"{PRE}bwd2{TGSSPOST}"),
    "bilinear4_bwd2": (
        f"{PRE}fwd_bilinear4{TGSSPOST}",
        f"{PRE}bwd2_bilinear4{TGSSPOST}",
    ),
    "anisotropic_bilinear": (
        f"{PRE}fwd_aniso_bilinear{TGSSPOST}",
        f"{PRE}bwd_aniso_bilinear{TGSSPOST}",
    ),
    "anisotropic_bilinear2": (
        f"{PRE}fwd_aniso_bilinear2{TGSSPOST}",
        f"{PRE}bwd_aniso_bilinear2{TGSSPOST}",
    ),
}


def rasterize_to_pixels_textured_gausssigs(
    means2d: Tensor,
    steepnesses: Tensor,
    ray_transforms: Tensor,
    colors: Tensor,
    opacities: Tensor,
    textures: Tensor,
    texture_range_x: float,
    texture_range_y: float,
    normals: Tensor,
    densify: Tensor,
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,
    flatten_ids: Tensor,
    backgrounds: Optional[Tensor] = None,
    masks: Optional[Tensor] = None,
    packed: bool = False,
    absgrad: bool = False,
    distloss: bool = False,
    gs_contrib_threshold: float = 0.0,
    filtering: Filtering = "bilinear",
    g_weight: float = 0.5,
    s_weight: float = 0.5,
) -> Tuple[Tensor, ...]:
    """Rasterize Textured GaussSig (blended Gaussian + smooth-step + constant base) to pixels.

    Args:
        means2d: Projected Gaussian means. [C, N, 2] or [nnz, 2] if packed.
        steepnesses: Per-gaussian smooth-step steepness. [C, N] or [nnz] if packed.
        ray_transforms: Ray-splat transform matrices. [C, N, 3, 3] or [nnz, 3, 3] if packed.
        colors: Gaussian colors or ND features. [C, N, channels] or [nnz, channels] if packed.
        opacities: Gaussian opacities. [C, N] or [nnz] if packed.
        textures: Gaussian 2D textures. [N, th, tw, 4].
        normals: Normals in camera space. [C, N, 3] or [nnz, 3] if packed.
        densify: Dummy gradient buffer for densification. [C, N, 2] or [nnz, 2] if packed.
        image_width: Image width.
        image_height: Image height.
        tile_size: Tile size.
        isect_offsets: Intersection offsets from `isect_offset_encode()`. [C, tile_height, tile_width]
        flatten_ids: Flattened intersection indices from `isect_tiles()`. [n_isects]
        backgrounds: Background colors. [C, channels]. Default: None.
        masks: Optional tile mask. [C, tile_height, tile_width]. Default: None.
        packed: Whether inputs are packed. Default: False.
        absgrad: Compute absolute gradients for means2d. Default: False.
        distloss: Enable distortion loss. Default: False.
        gs_contrib_threshold: Minimum alpha to count as a contribution. Default: 0.0.
        g_weight: Weight for the Gaussian kernel contribution. Default: 0.5.
        s_weight: Weight for the smooth-step sigmoid contribution. Default: 0.5.
            The constant base contribution is (1 - g_weight - s_weight).

    Returns:
        - **render_colors**    [C, image_height, image_width, channels]
        - **render_alphas**    [C, image_height, image_width, 1]
        - **render_normals**   [C, image_height, image_width, 3]
        - **render_distort**   [C, image_height, image_width, 1]
        - **render_median**    [C, image_height, image_width, 1]
        - **gs_contrib_sum**   [N]
        - **gs_contrib_count** [N]
    """
    C = isect_offsets.size(0)
    device = means2d.device
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.shape
        assert steepnesses.shape == (nnz,), steepnesses.shape
        assert ray_transforms.shape == (nnz, 3, 3), ray_transforms.shape
        assert colors.shape[0] == nnz, colors.shape
        assert opacities.shape == (nnz,), opacities.shape
    else:
        N = means2d.size(1)
        assert means2d.shape == (C, N, 2), means2d.shape
        assert steepnesses.shape == (C, N), steepnesses.shape
        assert ray_transforms.shape == (C, N, 3, 3), ray_transforms.shape
        assert colors.shape[:2] == (C, N), colors.shape
        assert opacities.shape == (C, N), opacities.shape
    if backgrounds is not None:
        assert backgrounds.shape == (C, colors.shape[-1]), backgrounds.shape
        backgrounds = backgrounds.contiguous()

    channels = colors.shape[-1]
    if channels > 512 or channels == 0:
        raise ValueError(f"Unsupported number of color channels: {channels}")
    if channels not in (1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512):
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        colors = torch.cat(
            [colors, torch.empty(*colors.shape[:-1], padded_channels, device=device)],
            dim=-1,
        )
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.empty(
                        *backgrounds.shape[:-1], padded_channels, device=device
                    ),
                ],
                dim=-1,
            )
    else:
        padded_channels = 0

    tile_height, tile_width = isect_offsets.shape[1:3]
    assert (
        tile_height * tile_size >= image_height
    ), f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
    assert (
        tile_width * tile_size >= image_width
    ), f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"

    if filtering in tgss_fns:
        (
            render_colors,
            render_alphas,
            render_normals,
            render_distort,
            render_median,
            gs_contrib_sum,
            gs_contrib_count,
        ) = _RasterizeToPixelsTexturedGaussSigs.apply(
            tgss_fns[filtering][0],
            tgss_fns[filtering][1],
            means2d.contiguous(),
            steepnesses.contiguous(),
            ray_transforms.contiguous(),
            colors.contiguous(),
            opacities.contiguous(),
            textures.contiguous(),
            texture_range_x,
            texture_range_y,
            normals.contiguous(),
            densify.contiguous(),
            backgrounds,
            masks,
            image_width,
            image_height,
            tile_size,
            isect_offsets.contiguous(),
            flatten_ids.contiguous(),
            absgrad,
            distloss,
            gs_contrib_threshold,
            g_weight,
            s_weight,
        )
    else:
        match filtering:
            case _:
                raise Exception(f"Unsupported filter type {filtering}")

    if padded_channels > 0:
        render_colors = render_colors[..., :-padded_channels]

    return (
        render_colors,
        render_alphas,
        render_normals,
        render_distort,
        render_median,
        gs_contrib_sum,
        gs_contrib_count,
    )
