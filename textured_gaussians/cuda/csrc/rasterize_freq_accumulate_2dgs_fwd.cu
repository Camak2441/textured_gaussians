#include "kernel_utils.h"
#include "helpers.cuh"
#include "types.cuh"
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/TensorAccessor.h>

#define FILTER_INV_SQUARE 2.0f

namespace gsplat
{
    namespace cg = cooperative_groups;

    // Shared memory layout per thread:
    //   int32_t  id             (4 B)
    //   vec3<S>  xy_opac        (12 B)
    //   vec3<S>  u_M, v_M, w_M (36 B)
    //   S[4]     J_unit         (16 B)  — (j00,j01,j10,j11)
    //   S[2]     scales         (8 B)   — (s1, s2)
    // Total: 76 B/thread

    template <typename S>
    __global__ void freq_accumulate_fwd_kernel(
        const uint32_t C, const uint32_t N, const uint32_t n_isects,
        const vec2<S> *__restrict__ means2d,                                    // [C*N, 2]
        const S *__restrict__ ray_transforms,                                   // [C*N, 9]
        const S *__restrict__ opacities,                                        // [C*N]
        const S *__restrict__ J_unit,                                           // [C*N, 4]
        const S *__restrict__ scales,                                           // [N, 2]
        at::PackedTensorAccessor32<const S, 4, at::RestrictPtrTraits> freq_map, // [C, H, W, 3]
        const S block_T_sq,  // (block_size_screen * texture_size)²
        const uint32_t image_width, const uint32_t image_height,
        const uint32_t freq_downsample,
        const uint32_t tile_size, const uint32_t tile_width, const uint32_t tile_height,
        const int32_t *__restrict__ tile_offsets,
        const int32_t *__restrict__ flatten_ids,
        S *__restrict__ freq_loss,     // [C*N]
        S *__restrict__ render_alphas, // [C*H*W]
        int32_t *__restrict__ last_ids // [C*H*W]
    )
    {
        auto block = cg::this_thread_block();
        const uint32_t camera_id = block.group_index().x;
        const int32_t tile_id = block.group_index().y * tile_width + block.group_index().z;
        const uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
        const uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

        tile_offsets += camera_id * tile_height * tile_width;
        render_alphas += camera_id * image_height * image_width;
        last_ids += camera_id * image_height * image_width;

        const S px = (S)j + S(0.5);
        const S py = (S)i + S(0.5);
        const int32_t pix_id = i * image_width + j;

        bool inside = (i < image_height && j < image_width);
        bool done = !inside;

        const int32_t range_start = tile_offsets[tile_id];
        const int32_t range_end = (camera_id == C - 1) && (tile_id == (int32_t)(tile_width * tile_height - 1))
                                      ? n_isects
                                      : tile_offsets[tile_id + 1];
        const uint32_t block_size = block.size();
        const uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

        // Shared memory
        extern __shared__ int s[];
        int32_t *id_batch = (int32_t *)s;
        vec3<S> *xy_opacity_batch = reinterpret_cast<vec3<S> *>(&id_batch[block_size]);
        vec3<S> *u_Ms_batch = reinterpret_cast<vec3<S> *>(&xy_opacity_batch[block_size]);
        vec3<S> *v_Ms_batch = reinterpret_cast<vec3<S> *>(&u_Ms_batch[block_size]);
        vec3<S> *w_Ms_batch = reinterpret_cast<vec3<S> *>(&v_Ms_batch[block_size]);
        S *J_batch     = (S *)&w_Ms_batch[block_size]; // [block_size * 4]
        S *scale_batch = (S *)&J_batch[block_size * 4]; // [block_size * 2]

        S T = S(1);
        uint32_t cur_idx = 0;

        // Spectral covariance at this pixel (same for all Gaussians)
        S ref_uu = S(0), ref_uv = S(0), ref_vv = S(0);
        if (inside)
        {
            const uint32_t fi = i / freq_downsample;
            const uint32_t fj = j / freq_downsample;
            ref_uu = freq_map[camera_id][fi][fj][0];
            ref_uv = freq_map[camera_id][fi][fj][1];
            ref_vv = freq_map[camera_id][fi][fj][2];
        }

        const uint32_t tr = block.thread_rank();

        for (uint32_t b = 0; b < num_batches; ++b)
        {
            if (__syncthreads_count(done) >= block_size)
                break;

            const uint32_t batch_start = range_start + block_size * b;
            const uint32_t idx = batch_start + tr;

            if (idx < (uint32_t)range_end)
            {
                const int32_t g = flatten_ids[idx];
                const uint32_t g_local = g % N;
                id_batch[tr] = g;
                const vec2<S> xy = means2d[g];
                xy_opacity_batch[tr] = {xy.x, xy.y, opacities[g]};
                u_Ms_batch[tr] = {ray_transforms[g * 9], ray_transforms[g * 9 + 1], ray_transforms[g * 9 + 2]};
                v_Ms_batch[tr] = {ray_transforms[g * 9 + 3], ray_transforms[g * 9 + 4], ray_transforms[g * 9 + 5]};
                w_Ms_batch[tr] = {ray_transforms[g * 9 + 6], ray_transforms[g * 9 + 7], ray_transforms[g * 9 + 8]};
                J_batch[tr * 4 + 0] = J_unit[g * 4 + 0]; // j00
                J_batch[tr * 4 + 1] = J_unit[g * 4 + 1]; // j01
                J_batch[tr * 4 + 2] = J_unit[g * 4 + 2]; // j10
                J_batch[tr * 4 + 3] = J_unit[g * 4 + 3]; // j11
                scale_batch[tr * 2 + 0] = scales[g_local * 2 + 0]; // s1
                scale_batch[tr * 2 + 1] = scales[g_local * 2 + 1]; // s2
            }
            block.sync();

            const uint32_t batch_size = min(block_size, (uint32_t)range_end - batch_start);
            for (uint32_t t = 0; (t < batch_size) && !done; ++t)
            {
                const int32_t g = id_batch[t];
                const vec3<S> xy_opac = xy_opacity_batch[t];
                const S opac = xy_opac.z;
                const vec3<S> u_M = u_Ms_batch[t];
                const vec3<S> v_M = v_Ms_batch[t];
                const vec3<S> w_M = w_Ms_batch[t];

                // Compute UV intersection (same as rasterizer forward)
                const vec3<S> h_u = px * w_M - u_M;
                const vec3<S> h_v = py * w_M - v_M;
                const vec3<S> ray_cross = glm::cross(h_u, h_v);
                if (ray_cross.z == 0.0)
                    continue;

                const vec2<S> sv = {ray_cross.x / ray_cross.z, ray_cross.y / ray_cross.z};
                const S gauss_weight_3d = sv.x * sv.x + sv.y * sv.y;
                const vec2<S> d = {xy_opac.x - px, xy_opac.y - py};
                const S gauss_weight_2d = FILTER_INV_SQUARE * (d.x * d.x + d.y * d.y);
                const S gauss_weight = min(gauss_weight_3d, gauss_weight_2d);
                const S sigma_val = S(0.5) * gauss_weight;
                const S alpha = min(S(0.999), opac * exp(-sigma_val));
                if (sigma_val < S(0) || alpha < S(1) / S(255))
                    continue;

                const S next_T = T * (S(1) - alpha);
                if (next_T <= 1e-4)
                {
                    done = true;
                    break;
                }
                const S fac = alpha * T;

                // Skip degenerate pixels (zero frequency content)
                if (ref_uu + ref_vv < S(1e-8))
                {
                    T = next_T;
                    cur_idx = batch_start + t;
                    continue;
                }

                // ── Eigendecompose sigma_ref = [[ruu, ruv],[ruv, rvv]] ──────────
                const S half_diff = S(0.5) * (ref_uu - ref_vv);
                const S disc = sqrtf(half_diff * half_diff + ref_uv * ref_uv);
                const S lambda1 = max(S(0.5) * (ref_uu + ref_vv) - disc, S(1e-8));
                const S lambda2 = max(S(0.5) * (ref_uu + ref_vv) + disc, S(1e-8));

                // Eigenvector for lambda2 (larger eigenvalue)
                S e2x, e2y;
                if (fabsf(ref_uv) > S(1e-7))
                {
                    const S norm_inv = rsqrtf(ref_uv * ref_uv + (disc - half_diff) * (disc - half_diff));
                    e2x = ref_uv * norm_inv;
                    e2y = (disc - half_diff) * norm_inv;
                }
                else
                {
                    e2x = (ref_uu >= ref_vv) ? S(1) : S(0);
                    e2y = (ref_uu >= ref_vv) ? S(0) : S(1);
                }
                const S e1x = -e2y, e1y = e2x; // perpendicular eigenvector

                // ── Load Jacobian and scales ─────────────────────────────────────
                const S j00 = J_batch[t * 4 + 0];
                const S j01 = J_batch[t * 4 + 1];
                const S j10 = J_batch[t * 4 + 2];
                const S j11 = J_batch[t * 4 + 3];
                const S s1  = scale_batch[t * 2 + 0];
                const S s2  = scale_batch[t * 2 + 1];

                const S det_J = j00 * j11 - j01 * j10;
                if (fabsf(det_J) < S(1e-8))
                {
                    T = next_T;
                    cur_idx = batch_start + t;
                    continue;
                }
                const S inv_det_J = S(1) / det_J;

                // ── Map eigenvectors to UV space via J_full^{-1} = diag(1/s) @ J_unit^{-1}
                // J_unit^{-1} @ e = [[j11, -j01],[-j10, j00]] / det_J @ [ex, ey]
                // then divide by (s1, s2) respectively
                const S w1_e1 = (j11 * e1x - j01 * e1y) * inv_det_J;
                const S w2_e1 = (-j10 * e1x + j00 * e1y) * inv_det_J;
                const S w1_e2 = (j11 * e2x - j01 * e2y) * inv_det_J;
                const S w2_e2 = (-j10 * e2x + j00 * e2y) * inv_det_J;

                const S inv_s1_sq = S(1) / (s1 * s1);
                const S inv_s2_sq = S(1) / (s2 * s2);
                const S euv_sq_1 = w1_e1 * w1_e1 * inv_s1_sq + w2_e1 * w2_e1 * inv_s2_sq;
                const S euv_sq_2 = w1_e2 * w1_e2 * inv_s1_sq + w2_e2 * w2_e2 * inv_s2_sq;

                // Guard: skip loss when scales are so small that euv_sq overflows
                // (logf(Inf) = Inf, which poisons the loss and backward NaN).
                if (isfinite(euv_sq_1) && isfinite(euv_sq_2))
                {
                    const S r1 = logf(max(euv_sq_1 * block_T_sq / lambda1, S(1e-8)));
                    const S r2 = logf(max(euv_sq_2 * block_T_sq / lambda2, S(1e-8)));
                    atomicAdd(&freq_loss[g], fac * (r1 * r1 + r2 * r2));
                }

                T = next_T;
                cur_idx = batch_start + t;
            }
        }

        if (inside)
        {
            render_alphas[pix_id] = S(1) - T;
            last_ids[pix_id] = static_cast<int32_t>(cur_idx);
        }
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    freq_accumulate_fwd_tensor(
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
        const torch::Tensor &flatten_ids)
    {
        GSPLAT_DEVICE_GUARD(means2d);
        GSPLAT_CHECK_INPUT(means2d);
        GSPLAT_CHECK_INPUT(ray_transforms);
        GSPLAT_CHECK_INPUT(opacities);
        GSPLAT_CHECK_INPUT(J_unit);
        GSPLAT_CHECK_INPUT(scales);
        GSPLAT_CHECK_INPUT(freq_map);
        GSPLAT_CHECK_INPUT(tile_offsets);
        GSPLAT_CHECK_INPUT(flatten_ids);

        TORCH_CHECK(means2d.dim() == 3, "freq_accumulate_fwd: packed mode not supported");

        uint32_t C = tile_offsets.size(0);
        uint32_t N = means2d.size(1);
        uint32_t tile_height = tile_offsets.size(1);
        uint32_t tile_width = tile_offsets.size(2);
        uint32_t n_isects = flatten_ids.size(0);

        dim3 threads = {tile_size, tile_size, 1};
        dim3 blocks = {C, tile_height, tile_width};

        auto f32 = means2d.options().dtype(torch::kFloat32);
        auto i32 = means2d.options().dtype(torch::kInt32);
        torch::Tensor freq_loss = torch::zeros({(int64_t)(C * N)}, f32);
        torch::Tensor render_alphas = torch::zeros({C, image_height, image_width}, f32);
        torch::Tensor last_ids_out = torch::zeros({C, image_height, image_width}, i32);

        // 76 B/thread: 4 (id) + 12*4 (4×vec3) + 16 (J[4]) + 8 (scales[2])
        const uint32_t shared_mem =
            tile_size * tile_size *
            (sizeof(int32_t) + sizeof(vec3<float>) * 4 + sizeof(float) * 4 + sizeof(float) * 2);

        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        if (cudaFuncSetAttribute(freq_accumulate_fwd_kernel<float>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess)
            AT_ERROR("freq_accumulate_fwd: failed to set shared mem (", shared_mem, " bytes)");

        freq_accumulate_fwd_kernel<float><<<blocks, threads, shared_mem, stream>>>(
            C, N, n_isects,
            reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
            ray_transforms.data_ptr<float>(),
            opacities.data_ptr<float>(),
            J_unit.data_ptr<float>(),
            scales.data_ptr<float>(),
            freq_map.packed_accessor32<const float, 4, at::RestrictPtrTraits>(),
            block_T_sq,
            image_width, image_height, freq_downsample, tile_size, tile_width, tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            freq_loss.data_ptr<float>(),
            render_alphas.data_ptr<float>(),
            last_ids_out.data_ptr<int32_t>());

        return std::make_tuple(freq_loss, render_alphas, last_ids_out);
    }

} // namespace gsplat