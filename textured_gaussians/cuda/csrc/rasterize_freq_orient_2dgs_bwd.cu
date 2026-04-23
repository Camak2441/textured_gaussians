#include "bindings.h"
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

    // Backward kernel for orientation loss.
    // Accumulates gradients for freq_vec [N,2] and J_unit [C*N,4].
    // Shared memory layout identical to forward: 84 B/thread.
    template <typename S>
    __global__ void freq_orient_bwd_kernel(
        const uint32_t C, const uint32_t N, const uint32_t n_isects,
        const vec2<S> *__restrict__ means2d,
        const S *__restrict__ ray_transforms,                                   // [C*N, 9]
        const S *__restrict__ opacities,                                        // [C*N]
        const S *__restrict__ J_unit,                                           // [C*N, 4]
        const S *__restrict__ scales,                                           // [N, 2]
        const S *__restrict__ freq_vec,                                         // [N, 2]
        at::PackedTensorAccessor32<const S, 4, at::RestrictPtrTraits> freq_map, // [C, H, W, 3]
        const uint32_t image_width, const uint32_t image_height,
        const uint32_t freq_downsample,
        const uint32_t tile_size, const uint32_t tile_width, const uint32_t tile_height,
        const int32_t *__restrict__ tile_offsets,
        const int32_t *__restrict__ flatten_ids,
        const S *__restrict__ render_alphas,   // [C, H, W]
        const int32_t *__restrict__ last_ids,  // [C, H, W]
        const S *__restrict__ v_orient_loss,   // [C*N]
        S *__restrict__ v_freq_vec,            // [N, 2]
        S *__restrict__ v_J_unit               // [C*N, 4]
    )
    {
        auto block = cg::this_thread_block();
        uint32_t camera_id = block.group_index().x;
        uint32_t tile_id   = block.group_index().y * tile_width + block.group_index().z;
        uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
        uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

        tile_offsets  += camera_id * tile_height * tile_width;
        render_alphas += camera_id * image_height * image_width;
        last_ids      += camera_id * image_height * image_width;

        const S px = (S)j + S(0.5);
        const S py = (S)i + S(0.5);
        const int32_t pix_id = min((int32_t)(i * image_width + j),
                                   (int32_t)(image_width * image_height - 1));

        bool inside = (i < image_height && j < image_width);

        const int32_t range_start = tile_offsets[tile_id];
        const int32_t range_end   =
            (camera_id == C - 1 && tile_id == tile_width * tile_height - 1)
                ? n_isects : tile_offsets[tile_id + 1];
        const uint32_t block_size  = block.size();
        const uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

        extern __shared__ int s[];
        int32_t *id_batch         = (int32_t *)s;
        vec3<S> *xy_opacity_batch = reinterpret_cast<vec3<S> *>(&id_batch[block_size]);
        vec3<S> *u_Ms_batch       = reinterpret_cast<vec3<S> *>(&xy_opacity_batch[block_size]);
        vec3<S> *v_Ms_batch       = reinterpret_cast<vec3<S> *>(&u_Ms_batch[block_size]);
        vec3<S> *w_Ms_batch       = reinterpret_cast<vec3<S> *>(&v_Ms_batch[block_size]);
        S *J_batch     = (S *)&w_Ms_batch[block_size];
        S *scale_batch = (S *)&J_batch[block_size * 4];
        S *fvec_batch  = (S *)&scale_batch[block_size * 2];

        S T_final = S(1) - render_alphas[pix_id];
        S T = T_final;
        const int32_t bin_final = inside ? last_ids[pix_id] : 0;

        const uint32_t tr = block.thread_rank();
        cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
        const int32_t warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());

        S ref_uu = S(0), ref_uv = S(0), ref_vv = S(0);
        if (inside)
        {
            const uint32_t fi = i / freq_downsample;
            const uint32_t fj = j / freq_downsample;
            ref_uu = freq_map[camera_id][fi][fj][0];
            ref_uv = freq_map[camera_id][fi][fj][1];
            ref_vv = freq_map[camera_id][fi][fj][2];
        }

        // Pre-compute per-pixel eigendecomposition (constant across Gaussians)
        S e2x = S(0), e2y = S(0), lambda2 = S(1e-8);
        bool valid_freq = inside && (ref_uu + ref_vv >= S(1e-8));
        if (valid_freq)
        {
            const S half_diff = S(0.5) * (ref_uu - ref_vv);
            const S disc = sqrtf(half_diff * half_diff + ref_uv * ref_uv);
            lambda2 = max(S(0.5) * (ref_uu + ref_vv) + disc, S(1e-8));
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
        }
        const S inv_sqrt_l2 = rsqrtf(lambda2);
        const S w_ref0 = e2x * inv_sqrt_l2;
        const S w_ref1 = e2y * inv_sqrt_l2;

        for (uint32_t b = 0; b < num_batches; ++b)
        {
            block.sync();
            const int32_t batch_end  = range_end - 1 - block_size * b;
            const int32_t batch_size = min((int32_t)block_size, batch_end + 1 - range_start);
            const int32_t idx        = batch_end - tr;

            if (idx >= range_start)
            {
                const int32_t g        = flatten_ids[idx];
                const uint32_t g_local = g % N;
                id_batch[tr] = g;
                const vec2<S> xy = means2d[g];
                xy_opacity_batch[tr] = {xy.x, xy.y, opacities[g]};
                u_Ms_batch[tr] = {ray_transforms[g*9+0], ray_transforms[g*9+1], ray_transforms[g*9+2]};
                v_Ms_batch[tr] = {ray_transforms[g*9+3], ray_transforms[g*9+4], ray_transforms[g*9+5]};
                w_Ms_batch[tr] = {ray_transforms[g*9+6], ray_transforms[g*9+7], ray_transforms[g*9+8]};
                J_batch[tr*4+0] = J_unit[g*4+0];
                J_batch[tr*4+1] = J_unit[g*4+1];
                J_batch[tr*4+2] = J_unit[g*4+2];
                J_batch[tr*4+3] = J_unit[g*4+3];
                scale_batch[tr*2+0] = scales[g_local*2+0];
                scale_batch[tr*2+1] = scales[g_local*2+1];
                fvec_batch[tr*2+0]  = freq_vec[g_local*2+0];
                fvec_batch[tr*2+1]  = freq_vec[g_local*2+1];
            }
            block.sync();

            for (uint32_t t = max(0, batch_end - warp_bin_final);
                 t < (uint32_t)batch_size; ++t)
            {
                const int32_t g        = id_batch[t];
                const uint32_t g_local = g % N;

                bool valid = inside && valid_freq && (batch_end - (int32_t)t <= bin_final);
                S alpha = S(0);

                if (valid)
                {
                    const vec3<S> xy_op = xy_opacity_batch[t];
                    const vec3<S> u_M   = u_Ms_batch[t];
                    const vec3<S> v_M   = v_Ms_batch[t];
                    const vec3<S> w_M   = w_Ms_batch[t];
                    const vec3<S> h_u   = px * w_M - u_M;
                    const vec3<S> h_v   = py * w_M - v_M;
                    const vec3<S> ray_cross = glm::cross(h_u, h_v);
                    if (ray_cross.z == 0.0)
                    {
                        valid = false;
                    }
                    else
                    {
                        const vec2<S> sv = {ray_cross.x / ray_cross.z, ray_cross.y / ray_cross.z};
                        const S gw3 = sv.x * sv.x + sv.y * sv.y;
                        const vec2<S> d = {xy_op.x - px, xy_op.y - py};
                        const S gw2  = FILTER_INV_SQUARE * (d.x * d.x + d.y * d.y);
                        const S sigma_g = S(0.5) * min(gw3, gw2);
                        alpha = min(S(0.999), xy_op.z * exp(-sigma_g));
                        if (sigma_g < S(0) || alpha < S(1) / S(255)) valid = false;
                    }
                }

                if (!warp.any(valid)) continue;

                // Accumulate: [v_fv0, v_fv1, v_j00, v_j01, v_j10, v_j11]
                S grads[6] = {S(0), S(0), S(0), S(0), S(0), S(0)};

                if (valid)
                {
                    const S ra    = S(1) / (S(1) - alpha);
                    T            *= ra;
                    const S fac   = alpha * T;
                    const S fac_v = fac * v_orient_loss[g];

                    const S j00 = J_batch[t*4+0], j01 = J_batch[t*4+1];
                    const S j10 = J_batch[t*4+2], j11 = J_batch[t*4+3];
                    const S s1  = scale_batch[t*2+0], s2 = scale_batch[t*2+1];
                    const S fv0 = fvec_batch[t*2+0],  fv1 = fvec_batch[t*2+1];

                    const S sfv0 = s1 * fv0, sfv1 = s2 * fv1;
                    const S w0   = j00 * sfv0 + j01 * sfv1;
                    const S w1   = j10 * sfv0 + j11 * sfv1;
                    const S res0 = w0 - w_ref0, res1 = w1 - w_ref1;

                    // Zero gradient when loss hits the clamp ceiling (matches fwd clamp).
                    if (lambda2 * (res0 * res0 + res1 * res1) <= S(10.0))
                    {
                        // Chain rule: d(lambda2 * (res0²+res1²)) through res, w, sfv, fv/J
                        const S two_fac_lam = S(2) * fac_v * lambda2;

                        // d/d(freq_vec)
                        grads[0] = two_fac_lam * (res0 * j00 * s1 + res1 * j10 * s1);
                        grads[1] = two_fac_lam * (res0 * j01 * s2 + res1 * j11 * s2);

                        // d/d(J_unit): outer product of [res0,res1] with [sfv0,sfv1]
                        grads[2] = two_fac_lam * res0 * sfv0; // d/dj00
                        grads[3] = two_fac_lam * res0 * sfv1; // d/dj01
                        grads[4] = two_fac_lam * res1 * sfv0; // d/dj10
                        grads[5] = two_fac_lam * res1 * sfv1; // d/dj11
                    }
                }

                warpSum<6, S>(grads, warp);

                if (warp.thread_rank() == 0)
                {
                    gpuAtomicAdd(&v_freq_vec[g_local*2+0], grads[0]);
                    gpuAtomicAdd(&v_freq_vec[g_local*2+1], grads[1]);
                    gpuAtomicAdd(&v_J_unit[g*4+0], grads[2]);
                    gpuAtomicAdd(&v_J_unit[g*4+1], grads[3]);
                    gpuAtomicAdd(&v_J_unit[g*4+2], grads[4]);
                    gpuAtomicAdd(&v_J_unit[g*4+3], grads[5]);
                }
            }
        }
    }

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
        const torch::Tensor &v_orient_loss)
    {
        GSPLAT_DEVICE_GUARD(means2d);
        GSPLAT_CHECK_INPUT(means2d);
        GSPLAT_CHECK_INPUT(ray_transforms);
        GSPLAT_CHECK_INPUT(opacities);
        GSPLAT_CHECK_INPUT(J_unit);
        GSPLAT_CHECK_INPUT(scales);
        GSPLAT_CHECK_INPUT(freq_vec);
        GSPLAT_CHECK_INPUT(freq_map);
        GSPLAT_CHECK_INPUT(tile_offsets);
        GSPLAT_CHECK_INPUT(flatten_ids);
        GSPLAT_CHECK_INPUT(render_alphas);
        GSPLAT_CHECK_INPUT(last_ids);
        GSPLAT_CHECK_INPUT(v_orient_loss);

        const uint32_t C = tile_offsets.size(0);
        const uint32_t N = means2d.size(1);
        const uint32_t tile_height = tile_offsets.size(1);
        const uint32_t tile_width  = tile_offsets.size(2);
        const uint32_t n_isects    = flatten_ids.size(0);

        dim3 threads = {tile_size, tile_size, 1};
        dim3 blocks  = {C, tile_height, tile_width};

        torch::Tensor v_freq_vec_out = torch::zeros_like(freq_vec);
        torch::Tensor v_J_unit_out   = torch::zeros_like(J_unit);

        if (n_isects)
        {
            const uint32_t shared_mem =
                tile_size * tile_size *
                (sizeof(int32_t) + sizeof(vec3<float>) * 4 + sizeof(float) * (4 + 2 + 2));

            at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
            if (cudaFuncSetAttribute(freq_orient_bwd_kernel<float>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_mem) != cudaSuccess)
                AT_ERROR("freq_orient_bwd: failed to set shared mem (", shared_mem, " bytes)");

            freq_orient_bwd_kernel<float><<<blocks, threads, shared_mem, stream>>>(
                C, N, n_isects,
                reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
                ray_transforms.data_ptr<float>(),
                opacities.data_ptr<float>(),
                J_unit.data_ptr<float>(),
                scales.data_ptr<float>(),
                freq_vec.data_ptr<float>(),
                freq_map.packed_accessor32<const float, 4, at::RestrictPtrTraits>(),
                image_width, image_height, freq_downsample, tile_size, tile_width, tile_height,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                render_alphas.data_ptr<float>(),
                last_ids.data_ptr<int32_t>(),
                v_orient_loss.data_ptr<float>(),
                v_freq_vec_out.data_ptr<float>(),
                v_J_unit_out.data_ptr<float>());
        }

        return std::make_tuple(v_freq_vec_out, v_J_unit_out);
    }

} // namespace gsplat
