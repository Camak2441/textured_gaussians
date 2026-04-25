#ifndef GSPLAT_CUDA_BINDINGS_H
#define GSPLAT_CUDA_BINDINGS_H

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <tuple>

#define GSPLAT_N_THREADS 256

#define GSPLAT_CHECK_CUDA(x) \
    TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define GSPLAT_CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define GSPLAT_CHECK_INPUT(x) \
    GSPLAT_CHECK_CUDA(x);     \
    GSPLAT_CHECK_CONTIGUOUS(x)
#define GSPLAT_DEVICE_GUARD(_ten) \
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_ten));

#define GSPLAT_PRAGMA_UNROLL _Pragma("unroll")

// https://github.com/pytorch/pytorch/blob/233305a852e1cd7f319b15b5137074c9eac455f6/aten/src/ATen/cuda/cub.cuh#L38-L46
#define GSPLAT_CUB_WRAPPER(func, ...)                                        \
    do                                                                       \
    {                                                                        \
        size_t temp_storage_bytes = 0;                                       \
        func(nullptr, temp_storage_bytes, __VA_ARGS__);                      \
        auto &caching_allocator = *::c10::cuda::CUDACachingAllocator::get(); \
        auto temp_storage = caching_allocator.allocate(temp_storage_bytes);  \
        func(temp_storage.get(), temp_storage_bytes, __VA_ARGS__);           \
    } while (false)

namespace gsplat
{

    enum CameraModelType
    {
        PINHOLE = 0,
        ORTHO = 1,
        FISHEYE = 2,
    };

    std::tuple<torch::Tensor, torch::Tensor> quat_scale_to_covar_preci_fwd_tensor(
        const torch::Tensor &quats,  // [N, 4]
        const torch::Tensor &scales, // [N, 3]
        const bool compute_covar,
        const bool compute_preci,
        const bool triu);

    std::tuple<torch::Tensor, torch::Tensor> quat_scale_to_covar_preci_bwd_tensor(
        const torch::Tensor &quats,                  // [N, 4]
        const torch::Tensor &scales,                 // [N, 3]
        const at::optional<torch::Tensor> &v_covars, // [N, 3, 3]
        const at::optional<torch::Tensor> &v_precis, // [N, 3, 3]
        const bool triu);

    std::tuple<torch::Tensor, torch::Tensor> proj_fwd_tensor(
        const torch::Tensor &means,  // [C, N, 3]
        const torch::Tensor &covars, // [C, N, 3, 3]
        const torch::Tensor &Ks,     // [C, 3, 3]
        const uint32_t width,
        const uint32_t height,
        const CameraModelType camera_model);

    std::tuple<torch::Tensor, torch::Tensor> proj_bwd_tensor(
        const torch::Tensor &means,  // [C, N, 3]
        const torch::Tensor &covars, // [C, N, 3, 3]
        const torch::Tensor &Ks,     // [C, 3, 3]
        const uint32_t width,
        const uint32_t height,
        const CameraModelType camera_model,
        const torch::Tensor &v_means2d, // [C, N, 2]
        const torch::Tensor &v_covars2d // [C, N, 2, 2]
    );

    std::tuple<torch::Tensor, torch::Tensor> world_to_cam_fwd_tensor(
        const torch::Tensor &means,   // [N, 3]
        const torch::Tensor &covars,  // [N, 3, 3]
        const torch::Tensor &viewmats // [C, 4, 4]
    );

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> world_to_cam_bwd_tensor(
        const torch::Tensor &means,                    // [N, 3]
        const torch::Tensor &covars,                   // [N, 3, 3]
        const torch::Tensor &viewmats,                 // [C, 4, 4]
        const at::optional<torch::Tensor> &v_means_c,  // [C, N, 3]
        const at::optional<torch::Tensor> &v_covars_c, // [C, N, 3, 3]
        const bool means_requires_grad,
        const bool covars_requires_grad,
        const bool viewmats_requires_grad);

    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    fully_fused_projection_fwd_tensor(
        const torch::Tensor &means,                // [N, 3]
        const at::optional<torch::Tensor> &covars, // [N, 6] optional
        const at::optional<torch::Tensor> &quats,  // [N, 4] optional
        const at::optional<torch::Tensor> &scales, // [N, 3] optional
        const torch::Tensor &viewmats,             // [C, 4, 4]
        const torch::Tensor &Ks,                   // [C, 3, 3]
        const uint32_t image_width,
        const uint32_t image_height,
        const float eps2d,
        const float near_plane,
        const float far_plane,
        const float radius_clip,
        const bool calc_compensations,
        const CameraModelType camera_model);

    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    fully_fused_projection_bwd_tensor(
        // fwd inputs
        const torch::Tensor &means,                // [N, 3]
        const at::optional<torch::Tensor> &covars, // [N, 6] optional
        const at::optional<torch::Tensor> &quats,  // [N, 4] optional
        const at::optional<torch::Tensor> &scales, // [N, 3] optional
        const torch::Tensor &viewmats,             // [C, 4, 4]
        const torch::Tensor &Ks,                   // [C, 3, 3]
        const uint32_t image_width,
        const uint32_t image_height,
        const float eps2d,
        const CameraModelType camera_model,
        // fwd outputs
        const torch::Tensor &radii,                       // [C, N]
        const torch::Tensor &conics,                      // [C, N, 3]
        const at::optional<torch::Tensor> &compensations, // [C, N] optional
        // grad outputs
        const torch::Tensor &v_means2d,                     // [C, N, 2]
        const torch::Tensor &v_depths,                      // [C, N]
        const torch::Tensor &v_conics,                      // [C, N, 3]
        const at::optional<torch::Tensor> &v_compensations, // [C, N] optional
        const bool viewmats_requires_grad);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> isect_tiles_tensor(
        const torch::Tensor &means2d,                    // [C, N, 2] or [nnz, 2]
        const torch::Tensor &radii,                      // [C, N] or [nnz]
        const torch::Tensor &depths,                     // [C, N] or [nnz]
        const at::optional<torch::Tensor> &camera_ids,   // [nnz]
        const at::optional<torch::Tensor> &gaussian_ids, // [nnz]
        const uint32_t C,
        const uint32_t tile_size,
        const uint32_t tile_width,
        const uint32_t tile_height,
        const bool sort,
        const bool double_buffer);

    torch::Tensor isect_offset_encode_tensor(
        const torch::Tensor &isect_ids, // [n_isects]
        const uint32_t C,
        const uint32_t tile_width,
        const uint32_t tile_height);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    rasterize_to_pixels_fwd_tensor(
        // Gaussian parameters
        const torch::Tensor &means2d,                   // [C, N, 2]
        const torch::Tensor &conics,                    // [C, N, 3]
        const torch::Tensor &colors,                    // [C, N, D]
        const torch::Tensor &opacities,                 // [N]
        const at::optional<torch::Tensor> &backgrounds, // [C, D]
        const at::optional<torch::Tensor> &mask,        // [C, tile_height, tile_width]
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        // intersections
        const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
        const torch::Tensor &flatten_ids   // [n_isects]
    );

    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    rasterize_to_pixels_bwd_tensor(
        // Gaussian parameters
        const torch::Tensor &means2d,                   // [C, N, 2]
        const torch::Tensor &conics,                    // [C, N, 3]
        const torch::Tensor &colors,                    // [C, N, 3]
        const torch::Tensor &opacities,                 // [N]
        const at::optional<torch::Tensor> &backgrounds, // [C, 3]
        const at::optional<torch::Tensor> &mask,        // [C, tile_height, tile_width]
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        // intersections
        const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
        const torch::Tensor &flatten_ids,  // [n_isects]
        // forward outputs
        const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
        const torch::Tensor &last_ids,      // [C, image_height, image_width]
        // gradients of outputs
        const torch::Tensor &v_render_colors, // [C, image_height, image_width, 3]
        const torch::Tensor &v_render_alphas, // [C, image_height, image_width, 1]
        // options
        bool absgrad);

    std::tuple<torch::Tensor, torch::Tensor> rasterize_to_indices_in_range_tensor(
        const uint32_t range_start,
        const uint32_t range_end,           // iteration steps
        const torch::Tensor transmittances, // [C, image_height, image_width]
        // Gaussian parameters
        const torch::Tensor &means2d,   // [C, N, 2]
        const torch::Tensor &conics,    // [C, N, 3]
        const torch::Tensor &opacities, // [N]
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        // intersections
        const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
        const torch::Tensor &flatten_ids   // [n_isects]
    );

    torch::Tensor compute_sh_fwd_tensor(
        const uint32_t degrees_to_use,
        const torch::Tensor &dirs,              // [..., 3]
        const torch::Tensor &coeffs,            // [..., K, 3]
        const at::optional<torch::Tensor> masks // [...]
    );
    std::tuple<torch::Tensor, torch::Tensor> compute_sh_bwd_tensor(
        const uint32_t K,
        const uint32_t degrees_to_use,
        const torch::Tensor &dirs,               // [..., 3]
        const torch::Tensor &coeffs,             // [..., K, 3]
        const at::optional<torch::Tensor> masks, // [...]
        const torch::Tensor &v_colors,           // [..., 3]
        bool compute_v_dirs);

    /****************************************************************************************
     * Packed Version
     ****************************************************************************************/
    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    fully_fused_projection_packed_fwd_tensor(
        const torch::Tensor &means,                // [N, 3]
        const at::optional<torch::Tensor> &covars, // [N, 6]
        const at::optional<torch::Tensor> &quats,  // [N, 3]
        const at::optional<torch::Tensor> &scales, // [N, 3]
        const torch::Tensor &viewmats,             // [C, 4, 4]
        const torch::Tensor &Ks,                   // [C, 3, 3]
        const uint32_t image_width,
        const uint32_t image_height,
        const float eps2d,
        const float near_plane,
        const float far_plane,
        const float radius_clip,
        const bool calc_compensations,
        const CameraModelType camera_model);

    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    fully_fused_projection_packed_bwd_tensor(
        // fwd inputs
        const torch::Tensor &means,                // [N, 3]
        const at::optional<torch::Tensor> &covars, // [N, 6]
        const at::optional<torch::Tensor> &quats,  // [N, 4]
        const at::optional<torch::Tensor> &scales, // [N, 3]
        const torch::Tensor &viewmats,             // [C, 4, 4]
        const torch::Tensor &Ks,                   // [C, 3, 3]
        const uint32_t image_width,
        const uint32_t image_height,
        const float eps2d,
        const CameraModelType camera_model,
        // fwd outputs
        const torch::Tensor &camera_ids,                  // [nnz]
        const torch::Tensor &gaussian_ids,                // [nnz]
        const torch::Tensor &conics,                      // [nnz, 3]
        const at::optional<torch::Tensor> &compensations, // [nnz] optional
        // grad outputs
        const torch::Tensor &v_means2d,                     // [nnz, 2]
        const torch::Tensor &v_depths,                      // [nnz]
        const torch::Tensor &v_conics,                      // [nnz, 3]
        const at::optional<torch::Tensor> &v_compensations, // [nnz] optional
        const bool viewmats_requires_grad,
        const bool sparse_grad);

    std::tuple<torch::Tensor, torch::Tensor> compute_relocation_tensor(
        torch::Tensor &opacities,
        torch::Tensor &scales,
        torch::Tensor &ratios,
        torch::Tensor &binoms,
        const int n_max);

    //====== 2DGS ======//
    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    fully_fused_projection_fwd_2dgs_tensor(
        const torch::Tensor &means,    // [N, 3]
        const torch::Tensor &quats,    // [N, 4]
        const torch::Tensor &scales,   // [N, 3]
        const torch::Tensor &viewmats, // [C, 4, 4]
        const torch::Tensor &Ks,       // [C, 3, 3]
        const uint32_t image_width,
        const uint32_t image_height,
        const float eps2d,
        const float near_plane,
        const float far_plane,
        const float radius_clip);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    fully_fused_projection_bwd_2dgs_tensor(
        // fwd inputs
        const torch::Tensor &means,    // [N, 3]
        const torch::Tensor &quats,    // [N, 4]
        const torch::Tensor &scales,   // [N, 3]
        const torch::Tensor &viewmats, // [C, 4, 4]
        const torch::Tensor &Ks,       // [C, 3, 3]
        const uint32_t image_width,
        const uint32_t image_height,
        // fwd outputs
        const torch::Tensor &radii,          // [C, N]
        const torch::Tensor &ray_transforms, // [C, N, 3, 3]
        // grad outputs
        const torch::Tensor &v_means2d,        // [C, N, 2]
        const torch::Tensor &v_depths,         // [C, N]
        const torch::Tensor &v_normals,        // [C, N, 3]
        const torch::Tensor &v_ray_transforms, // [C, N, 3, 3]
        const bool viewmats_requires_grad);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    fully_fused_projection_bwd2_2dgs_tensor(
        // fwd inputs
        const torch::Tensor &means,    // [N, 3]
        const torch::Tensor &quats,    // [N, 4]
        const torch::Tensor &scales,   // [N, 3]
        const torch::Tensor &viewmats, // [C, 4, 4]
        const torch::Tensor &Ks,       // [C, 3, 3]
        const uint32_t image_width,
        const uint32_t image_height,
        // fwd outputs
        const torch::Tensor &radii,          // [C, N]
        const torch::Tensor &ray_transforms, // [C, N, 3, 3]
        // grad outputs
        const torch::Tensor &v_means2d,        // [C, N, 2]
        const torch::Tensor &v_depths,         // [C, N]
        const torch::Tensor &v_normals,        // [C, N, 3]
        const torch::Tensor &v_ray_transforms, // [C, N, 3, 3]
        const bool viewmats_requires_grad);

    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    rasterize_to_pixels_fwd_2dgs_tensor(
        // Gaussian parameters
        const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
        const torch::Tensor &ray_transforms,            // [C, N, 3] or [nnz, 3]
        const torch::Tensor &colors,                    // [C, N, channels] or [nnz, channels]
        const torch::Tensor &opacities,                 // [C, N]  or [nnz]
        const torch::Tensor &normals,                   // [C, N, 3] or [nnz, 3]
        const at::optional<torch::Tensor> &backgrounds, // [C, channels]
        const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        // intersections
        const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
        const torch::Tensor &flatten_ids,  // [n_isects]
        const float gs_contrib_threshold);

    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    rasterize_to_pixels_bwd_2dgs_tensor(
        // Gaussian parameters
        const torch::Tensor &means2d,        // [C, N, 2] or [nnz, 2]
        const torch::Tensor &ray_transforms, // [C, N, 3, 3] or [nnz, 3, 3]
        const torch::Tensor &colors,         // [C, N, 3] or [nnz, 3]
        const torch::Tensor &opacities,      // [C, N] or [nnz]
        const torch::Tensor &normals,        // [C, N, 3] or [nnz, 3],
        const torch::Tensor &densify,
        const at::optional<torch::Tensor> &backgrounds, // [C, 3]
        const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        // ray_crossions
        const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
        const torch::Tensor &flatten_ids,  // [n_isects]
        // forward outputs
        const torch::Tensor
            &render_colors,                 // [C, image_height, image_width, COLOR_DIM]
        const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
        const torch::Tensor &last_ids,      // [C, image_height, image_width]
        const torch::Tensor &median_ids,    // [C, image_height, image_width]
        // gradients of outputs
        const torch::Tensor &v_render_colors,  // [C, image_height, image_width, 3]
        const torch::Tensor &v_render_alphas,  // [C, image_height, image_width, 1]
        const torch::Tensor &v_render_normals, // [C, image_height, image_width, 3]
        const torch::Tensor &v_render_distort, // [C, image_height, image_width, 1]
        const torch::Tensor &v_render_median,  // [C, image_height, image_width, 1]
        // options
        bool absgrad);

    std::tuple<torch::Tensor, torch::Tensor>
    rasterize_to_indices_in_range_2dgs_tensor(
        const uint32_t range_start,
        const uint32_t range_end,           // iteration steps
        const torch::Tensor transmittances, // [C, image_height, image_width]
        // Gaussian parameters
        const torch::Tensor &means2d,        // [C, N, 2]
        const torch::Tensor &ray_transforms, // [C, N, 3, 3]
        const torch::Tensor &opacities,      // [C, N]
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        // intersections
        const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
        const torch::Tensor &flatten_ids   // [n_isects]
    );

    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    fully_fused_projection_packed_fwd_2dgs_tensor(
        const torch::Tensor &means,    // [N, 3]
        const torch::Tensor &quats,    // [N, 3]
        const torch::Tensor &scales,   // [N, 3]
        const torch::Tensor &viewmats, // [C, 4, 4]
        const torch::Tensor &Ks,       // [C, 3, 3]
        const uint32_t image_width,
        const uint32_t image_height,
        const float near_plane,
        const float far_plane,
        const float radius_clip);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    fully_fused_projection_packed_bwd_2dgs_tensor(
        // fwd inputs
        const torch::Tensor &means,    // [N, 3]
        const torch::Tensor &quats,    // [N, 4]
        const torch::Tensor &scales,   // [N, 3]
        const torch::Tensor &viewmats, // [C, 4, 4]
        const torch::Tensor &Ks,       // [C, 3, 3]
        const uint32_t image_width,
        const uint32_t image_height,
        // fwd outputs
        const torch::Tensor &camera_ids,     // [nnz]
        const torch::Tensor &gaussian_ids,   // [nnz]
        const torch::Tensor &ray_transforms, // [nnz, 3, 3]
        // grad outputs
        const torch::Tensor &v_means2d,        // [nnz, 2]
        const torch::Tensor &v_depths,         // [nnz]
        const torch::Tensor &v_normals,        // [nnz, 3]
        const torch::Tensor &v_ray_transforms, // [nnz, 3, 3]
        const bool viewmats_requires_grad,
        const bool sparse_grad);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    fully_fused_projection_packed_bwd2_2dgs_tensor(
        // fwd inputs
        const torch::Tensor &means,    // [N, 3]
        const torch::Tensor &quats,    // [N, 4]
        const torch::Tensor &scales,   // [N, 3]
        const torch::Tensor &viewmats, // [C, 4, 4]
        const torch::Tensor &Ks,       // [C, 3, 3]
        const uint32_t image_width,
        const uint32_t image_height,
        // fwd outputs
        const torch::Tensor &camera_ids,     // [nnz]
        const torch::Tensor &gaussian_ids,   // [nnz]
        const torch::Tensor &ray_transforms, // [nnz, 3, 3]
        // grad outputs
        const torch::Tensor &v_means2d,        // [nnz, 2]
        const torch::Tensor &v_depths,         // [nnz]
        const torch::Tensor &v_normals,        // [nnz, 3]
        const torch::Tensor &v_ray_transforms, // [nnz, 3, 3]
        const bool viewmats_requires_grad,
        const bool sparse_grad);

    void selective_adam_update(
        torch::Tensor &param,
        torch::Tensor &param_grad,
        torch::Tensor &exp_avg,
        torch::Tensor &exp_avg_sq,
        torch::Tensor &tiles_touched,
        const float lr,
        const float b1,
        const float b2,
        const float eps,
        const uint32_t N,
        const uint32_t M);

    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    rasterize_to_pixels_fwd_textured_gaussians_tensor(
        // Gaussian parameters
        const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
        const torch::Tensor &ray_transforms,            // [C, N, 3] or [nnz, 3]
        const torch::Tensor &colors,                    // [C, N, channels] or [nnz, channels]
        const torch::Tensor &opacities,                 // [C, N]  or [nnz]
        const torch::Tensor &textures,                  //
        const float texture_range_x,                    //
        const float texture_range_y,                    //
        const torch::Tensor &normals,                   // [C, N, 3] or [nnz, 3]
        const at::optional<torch::Tensor> &backgrounds, // [C, channels]
        const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        // intersections
        const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
        const torch::Tensor &flatten_ids,  // [n_isects]
        const float gs_contrib_threshold,
        const float g_weight);

    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    rasterize_to_pixels_bwd_textured_gaussians_tensor(
        // Gaussian parameters
        const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
        const torch::Tensor &ray_transforms,            // [C, N, 3, 3] or [nnz, 3, 3]
        const torch::Tensor &colors,                    // [C, N, 3] or [nnz, 3]
        const torch::Tensor &opacities,                 // [C, N] or [nnz]
        const torch::Tensor &textures,                  //
        const float texture_range_x,                    //
        const float texture_range_y,                    //
        const torch::Tensor &normals,                   // [C, N, 3] or [nnz, 3],
        const torch::Tensor &densify,                   //
        const at::optional<torch::Tensor> &backgrounds, // [C, 3]
        const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        // ray_crossions
        const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
        const torch::Tensor &flatten_ids,  // [n_isects]
        const float g_weight,
        // forward outputs
        const torch::Tensor
            &render_colors,                 // [C, image_height, image_width, COLOR_DIM]
        const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
        const torch::Tensor &last_ids,      // [C, image_height, image_width]
        const torch::Tensor &median_ids,    // [C, image_height, image_width]
        // gradients of outputs
        const torch::Tensor &v_render_colors,  // [C, image_height, image_width, 3]
        const torch::Tensor &v_render_alphas,  // [C, image_height, image_width, 1]
        const torch::Tensor &v_render_normals, // [C, image_height, image_width, 3]
        const torch::Tensor &v_render_distort, // [C, image_height, image_width, 1]
        const torch::Tensor &v_render_median,  // [C, image_height, image_width, 1]
        // options
        bool absgrad);

    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    rasterize_to_pixels_bwd2_textured_gaussians_tensor(
        // Gaussian parameters
        const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
        const torch::Tensor &ray_transforms,            // [C, N, 3, 3] or [nnz, 3, 3]
        const torch::Tensor &colors,                    // [C, N, 3] or [nnz, 3]
        const torch::Tensor &opacities,                 // [C, N] or [nnz]
        const torch::Tensor &textures,                  //
        const float texture_range_x,                    //
        const float texture_range_y,                    //
        const torch::Tensor &normals,                   // [C, N, 3] or [nnz, 3]
        const torch::Tensor &densify,                   //
        const at::optional<torch::Tensor> &backgrounds, // [C, 3]
        const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        // intersections
        const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
        const torch::Tensor &flatten_ids,  // [n_isects]
        const float g_weight,
        // forward outputs
        const torch::Tensor &render_colors, // [C, image_height, image_width, COLOR_DIM]
        const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
        const torch::Tensor &last_ids,      // [C, image_height, image_width]
        const torch::Tensor &median_ids,    // [C, image_height, image_width]
        // gradients of outputs
        const torch::Tensor &v_render_colors,  // [C, image_height, image_width, 3]
        const torch::Tensor &v_render_alphas,  // [C, image_height, image_width, 1]
        const torch::Tensor &v_render_normals, // [C, image_height, image_width, 3]
        const torch::Tensor &v_render_distort, // [C, image_height, image_width, 1]
        const torch::Tensor &v_render_median,  // [C, image_height, image_width, 1]
        // options
        bool absgrad);

    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    rasterize_to_pixels_fwd_mip_textured_gaussians_tensor(
        // Gaussian parameters
        const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
        const torch::Tensor &ray_transforms,            // [C, N, 3] or [nnz, 3]
        const torch::Tensor &colors,                    // [C, N, channels] or [nnz, channels]
        const torch::Tensor &opacities,                 // [C, N]  or [nnz]
        const torch::Tensor &textures,                  //
        const float texture_range_x,                    //
        const float texture_range_y,                    //
        const torch::Tensor &normals,                   // [C, N, 3] or [nnz, 3]
        const at::optional<torch::Tensor> &backgrounds, // [C, channels]
        const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        // intersections
        const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
        const torch::Tensor &flatten_ids,  // [n_isects]
        const float gs_contrib_threshold,
        const float g_weight);

    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    rasterize_to_pixels_bwd_mip_textured_gaussians_tensor(
        // Gaussian parameters
        const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
        const torch::Tensor &ray_transforms,            // [C, N, 3, 3] or [nnz, 3, 3]
        const torch::Tensor &colors,                    // [C, N, 3] or [nnz, 3]
        const torch::Tensor &opacities,                 // [C, N] or [nnz]
        const torch::Tensor &textures,                  //
        const float texture_range_x,                    //
        const float texture_range_y,                    //
        const torch::Tensor &normals,                   // [C, N, 3] or [nnz, 3],
        const torch::Tensor &densify,                   //
        const at::optional<torch::Tensor> &backgrounds, // [C, 3]
        const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        // ray_crossions
        const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
        const torch::Tensor &flatten_ids,  // [n_isects]
        const float g_weight,
        // forward outputs
        const torch::Tensor
            &render_colors,                 // [C, image_height, image_width, COLOR_DIM]
        const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
        const torch::Tensor &last_ids,      // [C, image_height, image_width]
        const torch::Tensor &median_ids,    // [C, image_height, image_width]
        // gradients of outputs
        const torch::Tensor &v_render_colors,  // [C, image_height, image_width, 3]
        const torch::Tensor &v_render_alphas,  // [C, image_height, image_width, 1]
        const torch::Tensor &v_render_normals, // [C, image_height, image_width, 3]
        const torch::Tensor &v_render_distort, // [C, image_height, image_width, 1]
        const torch::Tensor &v_render_median,  // [C, image_height, image_width, 1]
        // options
        bool absgrad);

    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    rasterize_to_pixels_fwd_mip2_textured_gaussians_tensor(
        // Gaussian parameters
        const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
        const torch::Tensor &ray_transforms,            // [C, N, 3, 3] or [nnz, 3, 3]
        const torch::Tensor &colors,                    // [C, N, channels] or [nnz, channels]
        const torch::Tensor &opacities,                 // [C, N]  or [nnz]
        const torch::Tensor &textures,                  //
        const uint32_t log_texture_res,                 //
        const float texture_range_x,                    //
        const float texture_range_y,                    //
        const torch::Tensor &normals,                   // [C, N, 3] or [nnz, 3]
        const at::optional<torch::Tensor> &backgrounds, // [C, channels]
        const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        // intersections
        const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
        const torch::Tensor &flatten_ids,  // [n_isects]
        // additional parameters
        const float gs_contrib_threshold,
        const float g_weight);

    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    rasterize_to_pixels_bwd_mip2_textured_gaussians_tensor(
        // Gaussian parameters
        const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
        const torch::Tensor &ray_transforms,            // [C, N, 3, 3] or [nnz, 3, 3]
        const torch::Tensor &colors,                    // [C, N, 3] or [nnz, 3]
        const torch::Tensor &opacities,                 // [C, N] or [nnz]
        const torch::Tensor &textures,                  //
        const uint32_t log_texture_res,                 //
        const float texture_range_x,                    //
        const float texture_range_y,                    //
        const torch::Tensor &normals,                   // [C, N, 3] or [nnz, 3]
        const torch::Tensor &densify,                   //
        const at::optional<torch::Tensor> &backgrounds, // [C, 3]
        const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        // ray_crossions
        const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
        const torch::Tensor &flatten_ids,  // [n_isects]
        const float g_weight,
        // forward outputs
        const torch::Tensor
            &render_colors,                 // [C, image_height, image_width, COLOR_DIM]
        const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
        const torch::Tensor &last_ids,      // [C, image_height, image_width]
        const torch::Tensor &median_ids,    // [C, image_height, image_width]
        // gradients of outputs
        const torch::Tensor &v_render_colors,  // [C, image_height, image_width, 3]
        const torch::Tensor &v_render_alphas,  // [C, image_height, image_width, 1]
        const torch::Tensor &v_render_normals, // [C, image_height, image_width, 3]
        const torch::Tensor &v_render_distort, // [C, image_height, image_width, 1]
        const torch::Tensor &v_render_median,  // [C, image_height, image_width, 1]
        // options
        bool absgrad);

    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    rasterize_to_pixels_fwd_aniso_textured_gaussians_tensor(
        // Gaussian parameters
        const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
        const torch::Tensor &ray_transforms,            // [C, N, 3] or [nnz, 3]
        const torch::Tensor &colors,                    // [C, N, channels] or [nnz, channels]
        const torch::Tensor &opacities,                 // [C, N]  or [nnz]
        const torch::Tensor &textures,                  //
        const float texture_range_x,                    //
        const float texture_range_y,                    //
        const torch::Tensor &normals,                   // [C, N, 3] or [nnz, 3]
        const at::optional<torch::Tensor> &backgrounds, // [C, channels]
        const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        // intersections
        const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
        const torch::Tensor &flatten_ids,  // [n_isects]
        const float gs_contrib_threshold,
        const float g_weight);

    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    rasterize_to_pixels_bwd_aniso_textured_gaussians_tensor(
        // Gaussian parameters
        const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
        const torch::Tensor &ray_transforms,            // [C, N, 3, 3] or [nnz, 3, 3]
        const torch::Tensor &colors,                    // [C, N, 3] or [nnz, 3]
        const torch::Tensor &opacities,                 // [C, N] or [nnz]
        const torch::Tensor &textures,                  //
        const float texture_range_x,                    //
        const float texture_range_y,                    //
        const torch::Tensor &normals,                   // [C, N, 3] or [nnz, 3],
        const torch::Tensor &densify,                   //
        const at::optional<torch::Tensor> &backgrounds, // [C, 3]
        const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        // ray_crossions
        const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
        const torch::Tensor &flatten_ids,  // [n_isects]
        const float g_weight,
        // forward outputs
        const torch::Tensor
            &render_colors,                 // [C, image_height, image_width, COLOR_DIM]
        const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
        const torch::Tensor &last_ids,      // [C, image_height, image_width]
        const torch::Tensor &median_ids,    // [C, image_height, image_width]
        // gradients of outputs
        const torch::Tensor &v_render_colors,  // [C, image_height, image_width, 3]
        const torch::Tensor &v_render_alphas,  // [C, image_height, image_width, 1]
        const torch::Tensor &v_render_normals, // [C, image_height, image_width, 3]
        const torch::Tensor &v_render_distort, // [C, image_height, image_width, 1]
        const torch::Tensor &v_render_median,  // [C, image_height, image_width, 1]
        // options
        bool absgrad);

    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    rasterize_to_pixels_fwd_aniso_bilinear_textured_gaussians_tensor(
        const torch::Tensor &means2d,
        const torch::Tensor &ray_transforms,
        const torch::Tensor &colors,
        const torch::Tensor &opacities,
        const torch::Tensor &textures,
        const float texture_range_x,
        const float texture_range_y,
        const torch::Tensor &normals,
        const at::optional<torch::Tensor> &backgrounds,
        const at::optional<torch::Tensor> &masks,
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        const torch::Tensor &tile_offsets,
        const torch::Tensor &flatten_ids,
        const float gs_contrib_threshold,
        const float g_weight);

    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    rasterize_to_pixels_bwd_aniso_bilinear_textured_gaussians_tensor(
        const torch::Tensor &means2d,
        const torch::Tensor &ray_transforms,
        const torch::Tensor &colors,
        const torch::Tensor &opacities,
        const torch::Tensor &textures,
        const float texture_range_x,
        const float texture_range_y,
        const torch::Tensor &normals,
        const torch::Tensor &densify,
        const at::optional<torch::Tensor> &backgrounds,
        const at::optional<torch::Tensor> &masks,
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        const torch::Tensor &tile_offsets,
        const torch::Tensor &flatten_ids,
        const float g_weight,
        const torch::Tensor &render_colors,
        const torch::Tensor &render_alphas,
        const torch::Tensor &last_ids,
        const torch::Tensor &median_ids,
        const torch::Tensor &v_render_colors,
        const torch::Tensor &v_render_alphas,
        const torch::Tensor &v_render_normals,
        const torch::Tensor &v_render_distort,
        const torch::Tensor &v_render_median,
        bool absgrad);

    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    rasterize_to_pixels_fwd_bilinear2_textured_gaussians_tensor(
        // Gaussian parameters
        const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
        const torch::Tensor &ray_transforms,            // [C, N, 3, 3] or [nnz, 3, 3]
        const torch::Tensor &colors,                    // [C, N, channels] or [nnz, channels]
        const torch::Tensor &opacities,                 // [C, N]  or [nnz]
        const torch::Tensor &textures,                  //
        const float texture_range_x,                    //
        const float texture_range_y,                    //
        const torch::Tensor &normals,                   // [C, N, 3] or [nnz, 3]
        const at::optional<torch::Tensor> &backgrounds, // [C, channels]
        const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        // intersections
        const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
        const torch::Tensor &flatten_ids,  // [n_isects]
        const float gs_contrib_threshold,
        const float g_weight);

    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    rasterize_to_pixels_bwd_bilinear2_textured_gaussians_tensor(
        // Gaussian parameters
        const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
        const torch::Tensor &ray_transforms,            // [C, N, 3, 3] or [nnz, 3, 3]
        const torch::Tensor &colors,                    // [C, N, 3] or [nnz, 3]
        const torch::Tensor &opacities,                 // [C, N] or [nnz]
        const torch::Tensor &textures,                  //
        const float texture_range_x,                    //
        const float texture_range_y,                    //
        const torch::Tensor &normals,                   // [C, N, 3] or [nnz, 3]
        const torch::Tensor &densify,                   //
        const at::optional<torch::Tensor> &backgrounds, // [C, 3]
        const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        // intersections
        const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
        const torch::Tensor &flatten_ids,  // [n_isects]
        const float g_weight,
        // forward outputs
        const torch::Tensor &render_colors, // [C, image_height, image_width, COLOR_DIM]
        const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
        const torch::Tensor &last_ids,      // [C, image_height, image_width]
        const torch::Tensor &median_ids,    // [C, image_height, image_width]
        // gradients of outputs
        const torch::Tensor &v_render_colors,  // [C, image_height, image_width, 3]
        const torch::Tensor &v_render_alphas,  // [C, image_height, image_width, 1]
        const torch::Tensor &v_render_normals, // [C, image_height, image_width, 3]
        const torch::Tensor &v_render_distort, // [C, image_height, image_width, 1]
        const torch::Tensor &v_render_median,  // [C, image_height, image_width, 1]
        // options
        bool absgrad);

    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    rasterize_to_pixels_fwd_bilinear3_textured_gaussians_tensor(
        // Gaussian parameters
        const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
        const torch::Tensor &ray_transforms,            // [C, N, 3, 3] or [nnz, 3, 3]
        const torch::Tensor &colors,                    // [C, N, channels] or [nnz, channels]
        const torch::Tensor &opacities,                 // [C, N]  or [nnz]
        const torch::Tensor &textures,                  //
        const float texture_range_x,                    //
        const float texture_range_y,                    //
        const torch::Tensor &normals,                   // [C, N, 3] or [nnz, 3]
        const at::optional<torch::Tensor> &backgrounds, // [C, channels]
        const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        // intersections
        const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
        const torch::Tensor &flatten_ids,  // [n_isects]
        const float gs_contrib_threshold,
        const float g_weight);

    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    rasterize_to_pixels_bwd_bilinear3_textured_gaussians_tensor(
        // Gaussian parameters
        const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
        const torch::Tensor &ray_transforms,            // [C, N, 3, 3] or [nnz, 3, 3]
        const torch::Tensor &colors,                    // [C, N, 3] or [nnz, 3]
        const torch::Tensor &opacities,                 // [C, N] or [nnz]
        const torch::Tensor &textures,                  //
        const float texture_range_x,                    //
        const float texture_range_y,                    //
        const torch::Tensor &normals,                   // [C, N, 3] or [nnz, 3]
        const torch::Tensor &densify,                   //
        const at::optional<torch::Tensor> &backgrounds, // [C, 3]
        const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        // intersections
        const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
        const torch::Tensor &flatten_ids,  // [n_isects]
        const float g_weight,
        // forward outputs
        const torch::Tensor &render_colors, // [C, image_height, image_width, COLOR_DIM]
        const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
        const torch::Tensor &last_ids,      // [C, image_height, image_width]
        const torch::Tensor &median_ids,    // [C, image_height, image_width]
        // gradients of outputs
        const torch::Tensor &v_render_colors,  // [C, image_height, image_width, 3]
        const torch::Tensor &v_render_alphas,  // [C, image_height, image_width, 1]
        const torch::Tensor &v_render_normals, // [C, image_height, image_width, 3]
        const torch::Tensor &v_render_distort, // [C, image_height, image_width, 1]
        const torch::Tensor &v_render_median,  // [C, image_height, image_width, 1]
        // options
        bool absgrad);

    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    rasterize_to_pixels_bwd2_bilinear3_textured_gaussians_tensor(
        // Gaussian parameters
        const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
        const torch::Tensor &ray_transforms,            // [C, N, 3, 3] or [nnz, 3, 3]
        const torch::Tensor &colors,                    // [C, N, 3] or [nnz, 3]
        const torch::Tensor &opacities,                 // [C, N] or [nnz]
        const torch::Tensor &textures,                  //
        const float texture_range_x,                    //
        const float texture_range_y,                    //
        const torch::Tensor &normals,                   // [C, N, 3] or [nnz, 3]
        const torch::Tensor &densify,                   //
        const at::optional<torch::Tensor> &backgrounds, // [C, 3]
        const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        // intersections
        const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
        const torch::Tensor &flatten_ids,  // [n_isects]
        const float g_weight,
        // forward outputs
        const torch::Tensor &render_colors, // [C, image_height, image_width, COLOR_DIM]
        const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
        const torch::Tensor &last_ids,      // [C, image_height, image_width]
        const torch::Tensor &median_ids,    // [C, image_height, image_width]
        // gradients of outputs
        const torch::Tensor &v_render_colors,  // [C, image_height, image_width, 3]
        const torch::Tensor &v_render_alphas,  // [C, image_height, image_width, 1]
        const torch::Tensor &v_render_normals, // [C, image_height, image_width, 3]
        const torch::Tensor &v_render_distort, // [C, image_height, image_width, 1]
        const torch::Tensor &v_render_median,  // [C, image_height, image_width, 1]
        // options
        bool absgrad);

    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    rasterize_to_samples_fwd_textured_gaussians_tensor(
        // Gaussian parameters
        const torch::Tensor &means2d,             // [C, N, 2] or [nnz, 2]
        const torch::Tensor &ray_transforms,      // [C, N, 3, 3] or [nnz, 3, 3]
        const torch::Tensor &opacities,           // [C, N]  or [nnz]
        const at::optional<torch::Tensor> &masks, // [C, tile_height, tile_width]
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        // intersections
        const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
        const torch::Tensor &flatten_ids,  // [n_isects]
        const uint32_t num_texture_samples,
        const float sample_alpha_threshold);

    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    rasterize_to_pixels_fwd_implicit_textured_gaussians_tensor(
        // Gaussian parameters
        const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
        const torch::Tensor &ray_transforms,            // [C, N, 3, 3] or [nnz, 3, 3]
        const torch::Tensor &colors,                    // [C, N, channels] or [nnz, channels]
        const torch::Tensor &opacities,                 // [C, N]  or [nnz]
        const torch::Tensor &normals,                   // [C, N, 3] or [nnz, 3]
        const at::optional<torch::Tensor> &backgrounds, // [C, channels]
        const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        // intersections
        const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
        const torch::Tensor &flatten_ids,  // [n_isects]

        const torch::Tensor &texture_outputs, // [C, image_height, image_width, num_texture_samples, COLOR_DIM]
        const uint32_t num_texture_samples,
        const float sample_alpha_threshold,
        const float base_color_factor,
        // additional parameters
        const float gs_contrib_threshold);

    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    rasterize_to_pixels_bwd_implicit_textured_gaussians_tensor(
        // Gaussian parameters
        const torch::Tensor &means2d,        // [C, N, 2] or [nnz, 2]
        const torch::Tensor &ray_transforms, // [C, N, 3, 3] or [nnz, 3, 3]
        const torch::Tensor &colors,         // [C, N, 3] or [nnz, 3]
        const torch::Tensor &opacities,      // [C, N] or [nnz]
        const torch::Tensor &normals,        // [C, N, 3] or [nnz, 3]
        const torch::Tensor &densify,
        const at::optional<torch::Tensor> &backgrounds, // [C, 3]
        const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        // ray_crossions
        const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
        const torch::Tensor &flatten_ids,  // [n_isects]

        const torch::Tensor &texture_outputs,     //
        const torch::Tensor &sample_counts,       //
        const torch::Tensor &sample_gaussian_ids, //
        const uint32_t num_texture_samples,
        const float sample_alpha_threshold,
        const float base_color_factor,
        // forward outputs
        const torch::Tensor
            &render_colors,                 // [C, image_height, image_width, COLOR_DIM]
        const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
        const torch::Tensor &last_ids,      // [C, image_height, image_width]
        const torch::Tensor &median_ids,    // [C, image_height, image_width]
        // gradients of outputs
        const torch::Tensor &v_render_colors,  // [C, image_height, image_width, 3]
        const torch::Tensor &v_render_alphas,  // [C, image_height, image_width, 1]
        const torch::Tensor &v_render_normals, // [C, image_height, image_width, 3]
        const torch::Tensor &v_render_distort, // [C, image_height, image_width, 1]
        const torch::Tensor &v_render_median,  // [C, image_height, image_width, 1]
        // options
        bool absgrad);

    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    rasterize_to_pixels_fwd_dct_textured_gaussians_tensor(
        // Gaussian parameters
        const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
        const torch::Tensor &ray_transforms,            // [C, N, 3] or [nnz, 3]
        const torch::Tensor &colors,                    // [C, N, channels] or [nnz, channels]
        const torch::Tensor &opacities,                 // [C, N]  or [nnz]
        const torch::Tensor &textures,                  //
        const float texture_range_x,                    //
        const float texture_range_y,                    //
        const torch::Tensor &normals,                   // [C, N, 3] or [nnz, 3]
        const at::optional<torch::Tensor> &backgrounds, // [C, channels]
        const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        // intersections
        const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
        const torch::Tensor &flatten_ids,  // [n_isects]
        const float gs_contrib_threshold);

    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    rasterize_to_pixels_bwd_dct_textured_gaussians_tensor(
        // Gaussian parameters
        const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
        const torch::Tensor &ray_transforms,            // [C, N, 3, 3] or [nnz, 3, 3]
        const torch::Tensor &colors,                    // [C, N, 3] or [nnz, 3]
        const torch::Tensor &opacities,                 // [C, N] or [nnz]
        const torch::Tensor &textures,                  //
        const float texture_range_x,                    //
        const float texture_range_y,                    //
        const torch::Tensor &normals,                   // [C, N, 3] or [nnz, 3],
        const torch::Tensor &densify,                   //
        const at::optional<torch::Tensor> &backgrounds, // [C, 3]
        const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        // ray_crossions
        const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
        const torch::Tensor &flatten_ids,  // [n_isects]
        // forward outputs
        const torch::Tensor
            &render_colors,                 // [C, image_height, image_width, COLOR_DIM]
        const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
        const torch::Tensor &last_ids,      // [C, image_height, image_width]
        const torch::Tensor &median_ids,    // [C, image_height, image_width]
        // gradients of outputs
        const torch::Tensor &v_render_colors,  // [C, image_height, image_width, 3]
        const torch::Tensor &v_render_alphas,  // [C, image_height, image_width, 1]
        const torch::Tensor &v_render_normals, // [C, image_height, image_width, 3]
        const torch::Tensor &v_render_distort, // [C, image_height, image_width, 1]
        const torch::Tensor &v_render_median,  // [C, image_height, image_width, 1]
        // options
        bool absgrad);

    torch::Tensor
    rasterize_dct_textures_tensor(
        // Gaussian parameters
        const torch::Tensor &textures, //
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size);

    torch::Tensor
    generate_mipmap_fwd_tensor(
        const torch::Tensor &textures,
        const uint32_t log_texture_res,
        const uint32_t log_reduce,
        const uint32_t tile_size);

    torch::Tensor
    generate_mipmap_bwd_tensor(
        const uint32_t gaussians,
        const uint32_t channels,
        const uint32_t log_texture_res,
        const uint32_t log_reduce,
        const uint32_t tile_size,
        const torch::Tensor &v_mip_textures);
    // World-space sample rasterization (world coords only)
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    rasterize_to_samples_world_fwd_textured_gaussians_tensor(
        const torch::Tensor &means2d,
        const torch::Tensor &ray_transforms,
        const torch::Tensor &viewmats,
        const torch::Tensor &Ks,
        const torch::Tensor &opacities,
        const at::optional<torch::Tensor> &masks,
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        const torch::Tensor &tile_offsets,
        const torch::Tensor &flatten_ids,
        const uint32_t num_texture_samples,
        const float sample_alpha_threshold);

    // World-space + view-direction sample rasterization
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    rasterize_to_samples_world_and_view_fwd_textured_gaussians_tensor(
        const torch::Tensor &means2d,
        const torch::Tensor &ray_transforms,
        const torch::Tensor &viewmats,
        const torch::Tensor &Ks,
        const torch::Tensor &opacities,
        const at::optional<torch::Tensor> &masks,
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        const torch::Tensor &tile_offsets,
        const torch::Tensor &flatten_ids,
        const uint32_t num_texture_samples,
        const float sample_alpha_threshold);

    //====== 2DSS ======//
    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    rasterize_to_pixels_fwd_2dss_tensor(
        // Gaussian parameters
        const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
        const torch::Tensor &steepnesses,               // [C, N] or [nnz]
        const torch::Tensor &ray_transforms,            // [C, N, 3, 3] or [nnz, 3, 3]
        const torch::Tensor &colors,                    // [C, N, channels] or [nnz, channels]
        const torch::Tensor &opacities,                 // [C, N] or [nnz]
        const torch::Tensor &normals,                   // [C, N, 3] or [nnz, 3]
        const at::optional<torch::Tensor> &backgrounds, // [C, channels]
        const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        // intersections
        const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
        const torch::Tensor &flatten_ids,  // [n_isects]
        const float gs_contrib_threshold);

    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    rasterize_to_pixels_bwd_2dss_tensor(
        // Gaussian parameters
        const torch::Tensor &means2d,        // [C, N, 2] or [nnz, 2]
        const torch::Tensor &steepnesses,    // [C, N] or [nnz]
        const torch::Tensor &ray_transforms, // [C, N, 3, 3] or [nnz, 3, 3]
        const torch::Tensor &colors,         // [C, N, 3] or [nnz, 3]
        const torch::Tensor &opacities,      // [C, N] or [nnz]
        const torch::Tensor &normals,        // [C, N, 3] or [nnz, 3]
        const torch::Tensor &densify,
        const at::optional<torch::Tensor> &backgrounds, // [C, 3]
        const at::optional<torch::Tensor> &masks,       // [C, tile_height, tile_width]
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        // intersections
        const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
        const torch::Tensor &flatten_ids,  // [n_isects]
        // forward outputs
        const torch::Tensor &render_colors, // [C, image_height, image_width, COLOR_DIM]
        const torch::Tensor &render_alphas, // [C, image_height, image_width, 1]
        const torch::Tensor &last_ids,      // [C, image_height, image_width]
        const torch::Tensor &median_ids,    // [C, image_height, image_width]
        // gradients of outputs
        const torch::Tensor &v_render_colors,  // [C, image_height, image_width, 3]
        const torch::Tensor &v_render_alphas,  // [C, image_height, image_width, 1]
        const torch::Tensor &v_render_normals, // [C, image_height, image_width, 3]
        const torch::Tensor &v_render_distort, // [C, image_height, image_width, 1]
        const torch::Tensor &v_render_median,  // [C, image_height, image_width, 1]
        // options
        bool absgrad);

    // Textured Sigmoids 2DSS
    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    rasterize_to_pixels_fwd_textured_sigmoids_tensor(
        const torch::Tensor &means2d,
        const torch::Tensor &steepnesses,
        const torch::Tensor &ray_transforms,
        const torch::Tensor &colors,
        const torch::Tensor &opacities,
        const torch::Tensor &textures,
        const float texture_range_x,
        const float texture_range_y,
        const torch::Tensor &normals,
        const at::optional<torch::Tensor> &backgrounds,
        const at::optional<torch::Tensor> &masks,
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        const torch::Tensor &tile_offsets,
        const torch::Tensor &flatten_ids,
        const float gs_contrib_threshold,
        const float s_weight);

    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    rasterize_to_pixels_bwd2_textured_sigmoids_tensor(
        const torch::Tensor &means2d,
        const torch::Tensor &steepnesses,
        const torch::Tensor &ray_transforms,
        const torch::Tensor &colors,
        const torch::Tensor &opacities,
        const torch::Tensor &textures,
        const float texture_range_x,
        const float texture_range_y,
        const torch::Tensor &normals,
        const torch::Tensor &densify,
        const at::optional<torch::Tensor> &backgrounds,
        const at::optional<torch::Tensor> &masks,
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        const torch::Tensor &tile_offsets,
        const torch::Tensor &flatten_ids,
        float s_weight,
        const torch::Tensor &render_colors,
        const torch::Tensor &render_alphas,
        const torch::Tensor &last_ids,
        const torch::Tensor &median_ids,
        const torch::Tensor &v_render_colors,
        const torch::Tensor &v_render_alphas,
        const torch::Tensor &v_render_normals,
        const torch::Tensor &v_render_distort,
        const torch::Tensor &v_render_median,
        bool absgrad);

    // Textured GaussSig 2DGSS
    std::tuple<
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    rasterize_to_pixels_fwd_textured_gausssigs_tensor(
        const torch::Tensor &means2d,
        const torch::Tensor &steepnesses,
        const torch::Tensor &ray_transforms,
        const torch::Tensor &colors,
        const torch::Tensor &opacities,
        const torch::Tensor &textures,
        const float texture_range_x,
        const float texture_range_y,
        const torch::Tensor &normals,
        const at::optional<torch::Tensor> &backgrounds,
        const at::optional<torch::Tensor> &masks,
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        const torch::Tensor &tile_offsets,
        const torch::Tensor &flatten_ids,
        const float gs_contrib_threshold,
        const float g_weight,
        const float s_weight);

    std::tuple<
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    rasterize_to_pixels_bwd2_textured_gausssigs_tensor(
        const torch::Tensor &means2d,
        const torch::Tensor &steepnesses,
        const torch::Tensor &ray_transforms,
        const torch::Tensor &colors,
        const torch::Tensor &opacities,
        const torch::Tensor &textures,
        const float texture_range_x,
        const float texture_range_y,
        const torch::Tensor &normals,
        const torch::Tensor &densify,
        const at::optional<torch::Tensor> &backgrounds,
        const at::optional<torch::Tensor> &masks,
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        const torch::Tensor &tile_offsets,
        const torch::Tensor &flatten_ids,
        const torch::Tensor &render_colors,
        const torch::Tensor &render_alphas,
        const torch::Tensor &last_ids,
        const torch::Tensor &median_ids,
        const torch::Tensor &v_render_colors,
        const torch::Tensor &v_render_alphas,
        const torch::Tensor &v_render_normals,
        const torch::Tensor &v_render_distort,
        const torch::Tensor &v_render_median,
        bool absgrad,
        const float g_weight,
        const float s_weight);

    // 2DGSS
    std::tuple<
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    rasterize_to_pixels_fwd_2dgss_tensor(
        const torch::Tensor &means2d,
        const torch::Tensor &steepnesses,
        const torch::Tensor &ray_transforms,
        const torch::Tensor &colors,
        const torch::Tensor &opacities,
        const torch::Tensor &normals,
        const at::optional<torch::Tensor> &backgrounds,
        const at::optional<torch::Tensor> &masks,
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        const torch::Tensor &tile_offsets,
        const torch::Tensor &flatten_ids,
        const float gs_contrib_threshold,
        const float s_weight);

    std::tuple<
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    rasterize_to_pixels_bwd_2dgss_tensor(
        const torch::Tensor &means2d,
        const torch::Tensor &steepnesses,
        const torch::Tensor &ray_transforms,
        const torch::Tensor &colors,
        const torch::Tensor &opacities,
        const torch::Tensor &normals,
        const torch::Tensor &densify,
        const at::optional<torch::Tensor> &backgrounds,
        const at::optional<torch::Tensor> &masks,
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        const torch::Tensor &tile_offsets,
        const torch::Tensor &flatten_ids,
        const float s_weight,
        const torch::Tensor &render_colors,
        const torch::Tensor &render_alphas,
        const torch::Tensor &last_ids,
        const torch::Tensor &median_ids,
        const torch::Tensor &v_render_colors,
        const torch::Tensor &v_render_alphas,
        const torch::Tensor &v_render_normals,
        const torch::Tensor &v_render_distort,
        const torch::Tensor &v_render_median,
        bool absgrad);

    // Anisotropic Bilinear Textured Sigmoids
    std::tuple<
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    rasterize_to_pixels_fwd_aniso_bilinear_textured_sigmoids_tensor(
        const torch::Tensor &means2d,
        const torch::Tensor &steepnesses,
        const torch::Tensor &ray_transforms,
        const torch::Tensor &colors,
        const torch::Tensor &opacities,
        const torch::Tensor &textures,
        const float texture_range_x,
        const float texture_range_y,
        const torch::Tensor &normals,
        const at::optional<torch::Tensor> &backgrounds,
        const at::optional<torch::Tensor> &masks,
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        const torch::Tensor &tile_offsets,
        const torch::Tensor &flatten_ids,
        const float gs_contrib_threshold);

    std::tuple<
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    rasterize_to_pixels_bwd_aniso_bilinear_textured_sigmoids_tensor(
        const torch::Tensor &means2d,
        const torch::Tensor &steepnesses,
        const torch::Tensor &ray_transforms,
        const torch::Tensor &colors,
        const torch::Tensor &opacities,
        const torch::Tensor &textures,
        const float texture_range_x,
        const float texture_range_y,
        const torch::Tensor &normals,
        const torch::Tensor &densify,
        const at::optional<torch::Tensor> &backgrounds,
        const at::optional<torch::Tensor> &masks,
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        const torch::Tensor &tile_offsets,
        const torch::Tensor &flatten_ids,
        const torch::Tensor &render_colors,
        const torch::Tensor &render_alphas,
        const torch::Tensor &last_ids,
        const torch::Tensor &median_ids,
        const torch::Tensor &v_render_colors,
        const torch::Tensor &v_render_alphas,
        const torch::Tensor &v_render_normals,
        const torch::Tensor &v_render_distort,
        const torch::Tensor &v_render_median,
        bool absgrad);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    freq_accumulate_fwd_tensor(
        const torch::Tensor &means2d,
        const torch::Tensor &ray_transforms,
        const torch::Tensor &opacities,
        const torch::Tensor &J_unit,   // [C*N, 4] — (j00,j01,j10,j11) detached
        const torch::Tensor &scales,   // [N, 2]   — (s1, s2) linear scale
        const torch::Tensor &freq_map, // [C, H, W, 3] — (Suu, Suv, Svv) spectral cov
        const float block_T_sq,        // (block_size_screen * texture_size)²
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t freq_downsample,
        const uint32_t tile_size,
        const torch::Tensor &tile_offsets,
        const torch::Tensor &flatten_ids);

    torch::Tensor
    freq_accumulate_bwd_tensor(
        const torch::Tensor &means2d,
        const torch::Tensor &ray_transforms,
        const torch::Tensor &opacities,
        const torch::Tensor &J_unit,
        const torch::Tensor &scales,
        const torch::Tensor &freq_map,
        const float block_T_sq,
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t freq_downsample,
        const uint32_t tile_size,
        const torch::Tensor &tile_offsets,
        const torch::Tensor &flatten_ids,
        const torch::Tensor &render_alphas,
        const torch::Tensor &last_ids,
        const torch::Tensor &v_freq_loss);
    // Returns v_scales [N, 2]

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    freq_orient_fwd_tensor(
        const torch::Tensor &means2d,
        const torch::Tensor &ray_transforms,
        const torch::Tensor &opacities,
        const torch::Tensor &J_unit,   // [C*N, 4]
        const torch::Tensor &scales,   // [N, 2] detached
        const torch::Tensor &freq_vec, // [N, 2] learnable UV wavelength vector
        const torch::Tensor &freq_map, // [C, H, W, 3]
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t freq_downsample,
        const uint32_t tile_size,
        const torch::Tensor &tile_offsets,
        const torch::Tensor &flatten_ids);

    std::tuple<torch::Tensor, torch::Tensor>
    freq_orient_bwd_tensor(
        const torch::Tensor &means2d,
        const torch::Tensor &ray_transforms,
        const torch::Tensor &opacities,
        const torch::Tensor &J_unit,
        const torch::Tensor &scales,
        const torch::Tensor &freq_vec,
        const torch::Tensor &freq_map,
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t freq_downsample,
        const uint32_t tile_size,
        const torch::Tensor &tile_offsets,
        const torch::Tensor &flatten_ids,
        const torch::Tensor &render_alphas,
        const torch::Tensor &last_ids,
        const torch::Tensor &v_orient_loss);
    // Returns (v_freq_vec [N,2], v_J_unit [C*N,4])

} // namespace gsplat

#endif // GSPLAT_CUDA_BINDINGS_H
