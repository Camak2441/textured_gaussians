"""
World-space coordinate normalization utilities.

Design:
  - All normalization functions take explicit (center, scale) and implement:
        normalized = (points - center) / scale
        original   = normalized * scale + center

  - Separate "compute constants" functions return (center, scale) without
    modifying any data.

  - compute_unit_sphere_normalization is the single authoritative function for
    unit-sphere normalization; all sphere-based helpers call it.

  - Scene bounding boxes and sphere centers are derived from camera positions,
    following the conventions already used in this codebase.

Conventions:
  - Inputs use camtoworlds [C, 4, 4] (camera-to-world, OpenCV +Z forward).
    Camera positions: camtoworlds[:, :3, 3]
    Camera forward directions: camtoworlds[:, :3, 2]
  - Two focus-point strategies are provided:
      * compute_focus_point        — nerf-factory convention (datasets/normalize.py):
                                     median of feet-of-perpendiculars to world origin.
      * compute_focus_point_lstsq  — nerfstudio convention:
                                     iterative least-squares line–line intersection
                                     (https://en.wikipedia.org/wiki/Line-line_intersection
                                     #In_more_than_two_dimensions).
  - Scale conventions:
      * strict=True   →  max(camera distances from center)   (matches Parser.scene_scale)
      * strict=False  →  median(camera distances from center) (matches similarity_from_cameras
                         default strict_scaling=False)
  - multinerf (transform_poses_pca) uses mean centering + max-abs-position scale.
    This is available via center_method="mean" in compute_camera_unit_sphere_normalization.

References:
  - datasets/normalize.py::similarity_from_cameras  (nerf-factory, this codebase)
  - nerfstudio camera_utils.py::focus_of_attention
    https://github.com/nerfstudio-project/nerfstudio
  - google-research/multinerf camera_utils.py::transform_poses_pca
    https://github.com/google-research/multinerf
"""

from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Core workers
# ---------------------------------------------------------------------------

def normalize_points(points: Tensor, center: Tensor, scale: Tensor) -> Tensor:
    """Apply (center, scale) normalization:  out = (points - center) / scale.

    Args:
        points: Coordinates to normalize.  [..., 3]
        center: Translation to remove.     [3]
        scale:  Scalar divisor.

    Returns:
        Normalized coordinates.  [..., 3]
    """
    return (points - center) / scale


def unnormalize_points(points: Tensor, center: Tensor, scale: Tensor) -> Tensor:
    """Invert (center, scale) normalization:  out = points * scale + center.

    Args:
        points: Normalized coordinates.  [..., 3]
        center: Translation to restore.  [3]
        scale:  Scalar multiplier.

    Returns:
        Original-space coordinates.  [..., 3]
    """
    return points * scale + center


# ---------------------------------------------------------------------------
# Camera helpers  (camtoworlds convention)
# ---------------------------------------------------------------------------

def get_camera_positions(camtoworlds: Tensor) -> Tensor:
    """Extract camera positions in world space.

    For a c2w matrix [R | t; 0 1] the camera origin is the translation column.

    Args:
        camtoworlds: Camera-to-world transforms.  [C, 4, 4]

    Returns:
        Camera positions in world space.  [C, 3]
    """
    return camtoworlds[..., :3, 3]


def get_camera_directions(camtoworlds: Tensor) -> Tensor:
    """Extract unit optical-axis directions in world space (OpenCV +Z forward).

    The optical axis in camera space is [0, 0, 1].  In world space this is the
    third column of the rotation block: camtoworlds[:, :3, 2].

    Args:
        camtoworlds: Camera-to-world transforms.  [C, 4, 4]

    Returns:
        Unit viewing directions in world space.  [C, 3]
    """
    return F.normalize(camtoworlds[..., :3, 2], dim=-1)


# ---------------------------------------------------------------------------
# Focus-point computation  —  two strategies
# ---------------------------------------------------------------------------

def compute_focus_point(camtoworlds: Tensor) -> Tensor:
    """Compute the scene focus point using the nerf-factory strategy.

    For each camera ray (origin c_i, direction d_i), the closest point on that
    ray to the current world origin is:
        p_i = c_i + dot(-c_i, d_i) * d_i   (foot of perpendicular)

    The focus point is the median of {p_i}.

    Uses the same formula as datasets/normalize.py::similarity_from_cameras
    (center_method="focus"):
        nearest = t + (fwds * -t).sum(-1)[:, None] * fwds
        translate = -np.median(nearest, axis=0)

    Note: similarity_from_cameras applies an up-axis rotation (R_align) before
    computing the focus point.  This function operates in the frame it is called
    in.  When camtoworlds are already pre-normalised by the Parser (normalize=True),
    the two are equivalent.

    Args:
        camtoworlds: Camera-to-world transforms.  [C, 4, 4]

    Returns:
        Focus point in world space.  [3]
    """
    origins    = get_camera_positions(camtoworlds)   # [C, 3]
    directions = get_camera_directions(camtoworlds)  # [C, 3]

    # t_scalar = dot(-c_i, d_i) so that  nearest = c_i + t_scalar * d_i
    t_scalar = -(origins * directions).sum(dim=-1, keepdim=True)  # [C, 1]
    nearest  = origins + t_scalar * directions                     # [C, 3]

    return nearest.median(dim=0).values  # [3]


def compute_focus_point_lstsq(camtoworlds: Tensor) -> Tensor:
    """Compute the scene focus point using the nerfstudio least-squares strategy.

    Solves the line-line intersection problem in 3D:
        A = mean_i(I - d_i d_i^T)
        b = mean_i((I - d_i d_i^T) c_i)
        focus = A^{-1} b

    Reference: nerfstudio camera_utils.py::focus_of_attention
    https://en.wikipedia.org/wiki/Line-line_intersection#In_more_than_two_dimensions

    Unlike compute_focus_point this does not prune cameras iteratively; it is a
    single-pass mean-based solve.  Iterative pruning (as in nerfstudio) is
    omitted for simplicity; use compute_focus_point for robustness to outliers.

    Args:
        camtoworlds: Camera-to-world transforms.  [C, 4, 4]

    Returns:
        Focus point in world space.  [3]
    """
    origins    = get_camera_positions(camtoworlds)              # [C, 3]
    directions = get_camera_directions(camtoworlds)             # [C, 3]

    I   = torch.eye(3, device=camtoworlds.device, dtype=camtoworlds.dtype)
    ddt = torch.einsum("...i,...j->...ij", directions, directions)  # [C, 3, 3]
    # P_i = I - d_i d_i^T  is the projection matrix onto the plane perpendicular
    # to d_i.  It is symmetric and idempotent (P^T = P, P^2 = P when ||d||=1),
    # so the normal equations reduce to:
    #   A x = b   where  A = mean_i P_i,  b = mean_i P_i c_i
    P = I.unsqueeze(0) - ddt                                        # [C, 3, 3]
    A = P.mean(dim=0)                                               # [3, 3]
    b = (P @ origins.unsqueeze(-1)).mean(dim=0)[:, 0]              # [3]

    return torch.linalg.solve(A, b)  # [3]


# ---------------------------------------------------------------------------
# Base unit-sphere normalization constant computer
# ---------------------------------------------------------------------------

def compute_unit_sphere_normalization(
    points: Tensor,
    strict: bool = True,
    center: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Compute (center, scale) so that points fit within the unit sphere.

    This is the single authoritative function for unit-sphere normalization.
    All other sphere-based normalizers call this.

    Args:
        points: Point cloud.  [N, 3]  (or any leading batch dims, flattened internally)
        strict: True  → scale = max radius  (all points guaranteed inside sphere,
                         matches Parser.scene_scale convention).
                False → scale = median radius  (outlier-robust, matches
                         similarity_from_cameras strict_scaling=False).
        center: If provided, use this as the center instead of computing it from
                the points.  Useful when the center is pre-computed (e.g. the
                focus point) so that radii are measured from it directly without
                a second centering step.

    Returns:
        (center [3], scale scalar Tensor)
    """
    flat = points.reshape(-1, 3)
    if center is None:
        center = flat.median(dim=0).values
    radii = (flat - center).norm(dim=-1)
    scale = (radii.max() if strict else radii.median()).clamp(min=1e-8)
    return center, scale


# ---------------------------------------------------------------------------
# Camera-based unit-sphere normalization constants
# ---------------------------------------------------------------------------

def compute_camera_unit_sphere_normalization(
    camtoworlds: Tensor,
    strict: bool = True,
    center_method: Literal["focus", "mean"] = "focus",
) -> Tuple[Tensor, Tensor]:
    """Compute (center, scale) so that all camera positions lie within the unit sphere.

    Delegates to compute_unit_sphere_normalization for scale computation.
    The center is the focus point (nerf-factory / this codebase default) or the
    mean of camera positions (multinerf convention).

    center_method="focus":
        Center = median of feet-of-perpendiculars from world origin onto camera
        rays.  Matches datasets/normalize.py::similarity_from_cameras.
        strict=False matches the default (strict_scaling=False) scale.

    center_method="mean":
        Center = mean of camera positions.  Matches nerfstudio "poses" method
        and multinerf transform_poses_pca (with max-abs scale → use strict=True).

    Args:
        camtoworlds:   Camera-to-world transforms.  [C, 4, 4]
        strict:        True → max radius; False → median radius.
        center_method: "focus" (default, nerf-factory) or "mean" (multinerf/nerfstudio).

    Returns:
        (center [3], scale scalar Tensor)
    """
    cam_positions = get_camera_positions(camtoworlds)

    if center_method == "focus":
        focus = compute_focus_point(camtoworlds)
        # Pass focus explicitly so radii are measured from it, not from a second
        # median of the recentred cloud (which would introduce a double shift).
        _, scale = compute_unit_sphere_normalization(cam_positions, strict=strict, center=focus)
        return focus, scale
    elif center_method == "mean":
        center = cam_positions.mean(dim=0)
        _, scale = compute_unit_sphere_normalization(cam_positions, strict=strict, center=center)
        return center, scale
    else:
        raise ValueError(f"Unknown center_method: {center_method!r}")


# ---------------------------------------------------------------------------
# Scene bounding box derived from cameras
# ---------------------------------------------------------------------------

def compute_scene_bbox_from_cameras(
    camtoworlds: Tensor,
    padding: float = 0.1,
) -> Tuple[Tensor, Tensor]:
    """Compute a scene AABB from the camera positions.

    The bounding box of camera positions is grown by a relative padding on each
    side, following the common NeRF convention of inferring the scene extent
    from camera placement.

    Args:
        camtoworlds: Camera-to-world transforms.  [C, 4, 4]
        padding:     Fractional padding added on each side (0.1 → 10 %).

    Returns:
        (bbox_min [3], bbox_max [3])
    """
    cam_pos  = get_camera_positions(camtoworlds)
    bbox_min = cam_pos.min(dim=0).values
    bbox_max = cam_pos.max(dim=0).values
    extent   = bbox_max - bbox_min
    return bbox_min - padding * extent, bbox_max + padding * extent


def compute_bbox_normalization(
    bbox_min: Tensor,
    bbox_max: Tensor,
    output_range: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """Compute (center, scale) so that the bounding box maps to
    [-output_range, output_range]^3.

    The scale is driven by the longest axis to preserve aspect ratio, matching
    the convention used for scene initialisation in simple_trainer where
    Gaussians are spread over [-scene_scale, scene_scale]^3.

    Args:
        bbox_min:     Bounding box minimum.  [3]
        bbox_max:     Bounding box maximum.  [3]
        output_range: Half-width of the target cube (default 1 → [-1, 1]^3).

    Returns:
        (center [3], scale scalar Tensor)
    """
    center = 0.5 * (bbox_min + bbox_max)
    extent = (bbox_max - bbox_min).max()
    scale  = (extent / (2.0 * output_range)).clamp(min=1e-8)
    return center, scale
