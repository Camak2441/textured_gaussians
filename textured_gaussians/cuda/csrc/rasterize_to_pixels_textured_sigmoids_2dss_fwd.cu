#include "bindings.h"
#include "helpers.cuh"
#include "types.cuh"
#include "utils.cuh"
#include "filters/bilinear.cuh"
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/TensorAccessor.h>

#ifndef FILTER_INV_SQUARE
#define FILTER_INV_SQUARE 2.0f
#endif

namespace gsplat
{

    namespace cg = cooperative_groups;

    /****************************************************************************
     * Rasterization to Pixels Forward Pass Textured Sigmoids (2DSS kernel)
     ****************************************************************************/

    template <uint32_t COLOR_DIM, typename S>
    __global__ void rasterize_to_pixels_fwd_textured_sigmoids_kernel(
        const uint32_t C,                                                       // number of cameras
        const uint32_t N,                                                       // number of gaussians
        const uint32_t n_isects,                                                // number of ray-primitive intersections
        const bool packed,                                                      // whether the input tensors are packed
        const vec2<S> *__restrict__ means2d,                                    // [C, N, 2] or [nnz, 2]
        const S *__restrict__ steepnesses,                                      // [C, N] or [nnz]
        const S *__restrict__ ray_transforms,                                   // [C, N, 3, 3] or [nnz, 3, 3]
        const S *__restrict__ colors,                                           // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
        const S *__restrict__ opacities,                                        // [C, N] or [nnz]
        at::PackedTensorAccessor32<const S, 4, at::RestrictPtrTraits> textures, // [N, res, res, 4]
        const vec2<S> texture_range,                                            //
        const S *__restrict__ normals,                                          // [C, N, 3] or [nnz, 3]
        const S *__restrict__ backgrounds,                                      // [C, COLOR_DIM]
        const bool *__restrict__ masks,                                         // [C, tile_height, tile_width]
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        const uint32_t tile_width,
        const uint32_t tile_height,
        const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
        const int32_t *__restrict__ flatten_ids,  // [n_isects]
        const S gs_contrib_threshold,
        const S s_weight,

        // outputs
        S *__restrict__ render_colors,
        S *__restrict__ render_alphas,
        S *__restrict__ render_normals,
        S *__restrict__ render_distort,
        S *__restrict__ render_median,
        int32_t *__restrict__ last_ids,
        int32_t *__restrict__ median_ids,
        S *__restrict__ gs_contrib_sum,
        S *__restrict__ gs_contrib_count)
    {
        auto block = cg::this_thread_block();
        int32_t camera_id = block.group_index().x;
        int32_t tile_id = block.group_index().y * tile_width + block.group_index().z;
        uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
        uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

        uint32_t texture_res_y = textures.size(1);
        uint32_t texture_res_x = textures.size(2);

        tile_offsets += camera_id * tile_height * tile_width;
        render_colors += camera_id * image_height * image_width * COLOR_DIM;
        render_alphas += camera_id * image_height * image_width;
        last_ids += camera_id * image_height * image_width;
        render_normals += camera_id * image_height * image_width * 3;
        render_distort += camera_id * image_height * image_width;
        render_median += camera_id * image_height * image_width;
        median_ids += camera_id * image_height * image_width;

        if (backgrounds != nullptr)
            backgrounds += camera_id * COLOR_DIM;
        if (masks != nullptr)
            masks += camera_id * tile_height * tile_width;

        S px = (S)j + S(0.5);
        S py = (S)i + S(0.5);
        int32_t pix_id = i * image_width + j;

        bool inside = (i < image_height && j < image_width);
        bool done = !inside;

        if (masks != nullptr && inside && !masks[tile_id])
        {
            for (uint32_t k = 0; k < COLOR_DIM; ++k)
                render_colors[pix_id * COLOR_DIM + k] =
                    backgrounds == nullptr ? S(0) : backgrounds[k];
            return;
        }

        int32_t range_start = tile_offsets[tile_id];
        int32_t range_end =
            (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
                ? n_isects
                : tile_offsets[tile_id + 1];
        const uint32_t block_size = block.size();
        uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

        // Shared memory layout:
        // | id | xy_opacity | steepness | u_M | v_M | w_M |
        extern __shared__ int s[];
        int32_t *id_batch = (int32_t *)s;
        vec3<S> *xy_opacity_batch = reinterpret_cast<vec3<S> *>(&id_batch[block_size]);
        S *steepnesses_batch = (S *)(&xy_opacity_batch[block_size]);
        vec3<S> *u_Ms_batch = reinterpret_cast<vec3<S> *>(&steepnesses_batch[block_size]);
        vec3<S> *v_Ms_batch = reinterpret_cast<vec3<S> *>(&u_Ms_batch[block_size]);
        vec3<S> *w_Ms_batch = reinterpret_cast<vec3<S> *>(&v_Ms_batch[block_size]);

        S T = S(1);
        uint32_t cur_idx = 0;
        uint32_t tr = block.thread_rank();

        S distort = S(0);
        S accum_vis_depth = S(0);
        S median_depth = S(0);
        uint32_t median_idx = 0;

        S pix_out[COLOR_DIM] = {S(0)};
        S normal_out[3] = {S(0)};

        for (uint32_t b = 0; b < num_batches; ++b)
        {
            if (__syncthreads_count(done) >= block_size)
                break;

            uint32_t batch_start = range_start + block_size * b;
            uint32_t idx = batch_start + tr;

            if (idx < range_end)
            {
                int32_t g = flatten_ids[idx];
                id_batch[tr] = g;
                const vec2<S> xy = means2d[g];
                const S opac = opacities[g];
                xy_opacity_batch[tr] = {xy.x, xy.y, opac};
                steepnesses_batch[tr] = steepnesses[g];
                u_Ms_batch[tr] = {
                    ray_transforms[g * 9], ray_transforms[g * 9 + 1], ray_transforms[g * 9 + 2]};
                v_Ms_batch[tr] = {
                    ray_transforms[g * 9 + 3], ray_transforms[g * 9 + 4], ray_transforms[g * 9 + 5]};
                w_Ms_batch[tr] = {
                    ray_transforms[g * 9 + 6], ray_transforms[g * 9 + 7], ray_transforms[g * 9 + 8]};
            }

            block.sync();

            uint32_t batch_size = min(block_size, range_end - batch_start);
            for (uint32_t t = 0; (t < batch_size) && !done; ++t)
            {
                int32_t g = id_batch[t];

                const vec3<S> xy_opac = xy_opacity_batch[t];
                const S opac = xy_opac.z;
                const S steepness = steepnesses_batch[t];

                const vec3<S> u_M = u_Ms_batch[t];
                const vec3<S> v_M = v_Ms_batch[t];
                const vec3<S> w_M = w_Ms_batch[t];

                const vec3<S> h_u = px * w_M - u_M;
                const vec3<S> h_v = py * w_M - v_M;

                const vec3<S> ray_cross = glm::cross(h_u, h_v);
                if (fabsf(ray_cross.z) < S(1e-6f))
                    continue;

                const vec2<S> s = vec2<S>(ray_cross.x / ray_cross.z, ray_cross.y / ray_cross.z);

                // texture lookup for alpha scaling and color additive
                int32_t ucoords[4];
                int32_t vcoords[4];
                S bilerp_weights[4];
                int32_t valid_texture = bilinear::precompute(
                    s.x, s.y, texture_res_x, texture_res_y,
                    texture_range.x, texture_range.y,
                    ucoords, vcoords, bilerp_weights);

                S alpha_scaling_factor = S(0);
                if (valid_texture > 0)
                {
                    GSPLAT_PRAGMA_UNROLL
                    for (uint32_t i = 0; i < 4; ++i)
                        alpha_scaling_factor +=
                            bilerp_weights[i] * textures[g][vcoords[i]][ucoords[i]][3];
                }
                else
                {
                    alpha_scaling_factor = S(1);
                }

                // gaussian weight (minimum of 3D intersection and 2D projected)
                const S gauss_weight_3d = s.x * s.x + s.y * s.y;
                const vec2<S> d = {xy_opac.x - px, xy_opac.y - py};
                const S gauss_weight_2d = FILTER_INV_SQUARE * (d.x * d.x + d.y * d.y);
                const S gauss_weight = min(gauss_weight_3d, gauss_weight_2d);

                // smooth step visibility: vis = 1/(1+gw^k) = sigmoid(-k*log(gw))
                S step_vis;
                if (gauss_weight <= S(0))
                {
                    step_vis = S(1);
                }
                else
                {
                    step_vis = sigmoid(-(steepness * log(gauss_weight)));
                }

                const S alpha = min(S(0.999), opac * (S(0.998) - s_weight + s_weight * step_vis) * alpha_scaling_factor);

                if (gauss_weight < S(0) || alpha < S(1) / S(255))
                    continue;

                const S next_T = T * (S(1) - alpha);
                if (next_T <= 1e-4)
                {
                    done = true;
                    break;
                }

                // volumetric rendering weight for this gaussian at this pixel
                const S vis = alpha * T;
                const S *c_ptr = colors + g * COLOR_DIM;
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k)
                {
                    auto base_color = c_ptr[k];
                    S tex_color = S(0);
                    if (valid_texture > 0)
                    {
                        for (uint32_t i = 0; i < 4; ++i)
                            tex_color += bilerp_weights[i] * textures[g][vcoords[i]][ucoords[i]][k];
                    }
                    pix_out[k] += (base_color + tex_color) * vis;
                }

                const S *n_ptr = normals + g * 3;
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t k = 0; k < 3; ++k)
                    normal_out[k] += n_ptr[k] * vis;

                if (render_distort != nullptr)
                {
                    const S depth = c_ptr[COLOR_DIM - 1];
                    const S distort_bi_0 = vis * depth * (S(1) - T);
                    const S distort_bi_1 = vis * accum_vis_depth;
                    distort += S(2) * (distort_bi_0 - distort_bi_1);
                    accum_vis_depth += vis * depth;
                }

                if (T > 0.5)
                {
                    median_depth = c_ptr[COLOR_DIM - 1];
                    median_idx = batch_start + t;
                }

                cur_idx = batch_start + t;
                T = next_T;

                if (alpha > gs_contrib_threshold)
                {
                    atomicAdd(&gs_contrib_sum[g], alpha);
                    atomicAdd(&gs_contrib_count[g], S(1));
                }
            }
        }

        if (inside)
        {
            render_alphas[pix_id] = S(1) - T;
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t k = 0; k < COLOR_DIM; ++k)
                render_colors[pix_id * COLOR_DIM + k] =
                    backgrounds == nullptr ? pix_out[k] : (pix_out[k] + T * backgrounds[k]);
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t k = 0; k < 3; ++k)
                render_normals[pix_id * 3 + k] = normal_out[k];
            last_ids[pix_id] = static_cast<int32_t>(cur_idx);
            if (render_distort != nullptr)
                render_distort[pix_id] = distort;
            render_median[pix_id] = median_depth;
            median_ids[pix_id] = static_cast<int32_t>(median_idx);
        }
    }

    template <uint32_t CDIM>
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
    call_fwd_ts_kernel_with_dim(
        const torch::Tensor &means2d,
        const torch::Tensor &steepnesses,
        const torch::Tensor &ray_transforms,
        const torch::Tensor &colors,
        const torch::Tensor &opacities,
        const torch::Tensor &textures,
        const vec2<float> texture_range,
        const torch::Tensor &normals,
        const at::optional<torch::Tensor> &backgrounds,
        const at::optional<torch::Tensor> &masks,
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        const torch::Tensor &tile_offsets,
        const torch::Tensor &flatten_ids,
        const float gs_contrib_threshold,
        const float s_weight)
    {
        GSPLAT_DEVICE_GUARD(means2d);
        GSPLAT_CHECK_INPUT(means2d);
        GSPLAT_CHECK_INPUT(steepnesses);
        GSPLAT_CHECK_INPUT(ray_transforms);
        GSPLAT_CHECK_INPUT(colors);
        GSPLAT_CHECK_INPUT(opacities);
        GSPLAT_CHECK_INPUT(textures);
        GSPLAT_CHECK_INPUT(normals);
        GSPLAT_CHECK_INPUT(tile_offsets);
        GSPLAT_CHECK_INPUT(flatten_ids);
        if (backgrounds.has_value())
        {
            GSPLAT_CHECK_INPUT(backgrounds.value());
        }
        if (masks.has_value())
        {
            GSPLAT_CHECK_INPUT(masks.value());
        }

        bool packed = means2d.dim() == 2;
        uint32_t C = tile_offsets.size(0);
        uint32_t N = packed ? 0 : means2d.size(1);
        uint32_t channels = colors.size(-1);
        uint32_t tile_height = tile_offsets.size(1);
        uint32_t tile_width = tile_offsets.size(2);
        uint32_t n_isects = flatten_ids.size(0);

        dim3 threads = {tile_size, tile_size, 1};
        dim3 blocks = {C, tile_height, tile_width};

        torch::Tensor renders = torch::empty(
            {C, image_height, image_width, channels},
            means2d.options().dtype(torch::kFloat32));
        torch::Tensor alphas = torch::empty(
            {C, image_height, image_width, 1},
            means2d.options().dtype(torch::kFloat32));
        torch::Tensor last_ids = torch::empty(
            {C, image_height, image_width},
            means2d.options().dtype(torch::kInt32));
        torch::Tensor median_ids = torch::empty(
            {C, image_height, image_width},
            means2d.options().dtype(torch::kInt32));
        torch::Tensor render_normals = torch::empty(
            {C, image_height, image_width, 3},
            means2d.options().dtype(torch::kFloat32));
        torch::Tensor render_distort = torch::empty(
            {C, image_height, image_width, 1},
            means2d.options().dtype(torch::kFloat32));
        torch::Tensor render_median = torch::empty(
            {C, image_height, image_width, 1},
            means2d.options().dtype(torch::kFloat32));
        torch::Tensor gs_contrib_sum = torch::zeros(
            {N}, means2d.options().dtype(torch::kFloat32));
        torch::Tensor gs_contrib_count = torch::zeros(
            {N}, means2d.options().dtype(torch::kFloat32));

        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        const uint32_t shared_mem =
            tile_size * tile_size *
            (sizeof(int32_t) + sizeof(vec3<float>) + sizeof(float) +
             sizeof(vec3<float>) + sizeof(vec3<float>) + sizeof(vec3<float>));

        if (cudaFuncSetAttribute(
                rasterize_to_pixels_fwd_textured_sigmoids_kernel<CDIM, float>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                shared_mem) != cudaSuccess)
        {
            AT_ERROR(
                "Failed to set maximum shared memory size (requested ",
                shared_mem,
                " bytes), try lowering tile_size.");
        }
        rasterize_to_pixels_fwd_textured_sigmoids_kernel<CDIM, float>
            <<<blocks, threads, shared_mem, stream>>>(
                C,
                N,
                n_isects,
                packed,
                reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
                steepnesses.data_ptr<float>(),
                ray_transforms.data_ptr<float>(),
                colors.data_ptr<float>(),
                opacities.data_ptr<float>(),
                textures.packed_accessor32<const float, 4, at::RestrictPtrTraits>(),
                texture_range,
                normals.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
                masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
                image_width,
                image_height,
                tile_size,
                tile_width,
                tile_height,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                gs_contrib_threshold,
                s_weight,
                renders.data_ptr<float>(),
                alphas.data_ptr<float>(),
                render_normals.data_ptr<float>(),
                render_distort.data_ptr<float>(),
                render_median.data_ptr<float>(),
                last_ids.data_ptr<int32_t>(),
                median_ids.data_ptr<int32_t>(),
                gs_contrib_sum.data_ptr<float>(),
                gs_contrib_count.data_ptr<float>());

        return std::make_tuple(
            renders,
            alphas,
            render_normals,
            render_distort,
            render_median,
            last_ids,
            median_ids,
            gs_contrib_sum,
            gs_contrib_count);
    }

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
        const float s_weight)
    {
        GSPLAT_CHECK_INPUT(colors);
        uint32_t channels = colors.size(-1);

#define __GS__CALL_(N)                                     \
    case N:                                                \
        return call_fwd_ts_kernel_with_dim<N>(             \
            means2d,                                       \
            steepnesses,                                   \
            ray_transforms,                                \
            colors,                                        \
            opacities,                                     \
            textures,                                      \
            vec2<float>(texture_range_x, texture_range_y), \
            normals,                                       \
            backgrounds,                                   \
            masks,                                         \
            image_width,                                   \
            image_height,                                  \
            tile_size,                                     \
            tile_offsets,                                  \
            flatten_ids,                                   \
            gs_contrib_threshold,                          \
            s_weight);

        switch (channels)
        {
            __GS__CALL_(1)
            __GS__CALL_(2)
            __GS__CALL_(3)
            __GS__CALL_(4)
            __GS__CALL_(5)
        default:
            AT_ERROR("Unsupported number of channels: ", channels);
        }
    }

} // namespace gsplat
