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

    // Shared memory per thread: 84 B
    //   int32_t  id              (4 B)
    //   vec3<S>  xy_opac         (12 B)
    //   vec3<S>  u_M, v_M, w_M  (36 B)
    //   S[4]     J_unit          (16 B)
    //   S[2]     scales          (8 B)
    //   S[2]     freq_vec        (8 B)

    template <typename S>
    __global__ void freq_orient_fwd_kernel(
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
        S *__restrict__ orient_loss,   // [C*N]
        S *__restrict__ render_alphas, // [C*H*W]
        int32_t *__restrict__ last_ids // [C*H*W]
    )
    {
        auto block = cg::this_thread_block();
        const uint32_t camera_id = block.group_index().x;
        const int32_t tile_id    = block.group_index().y * tile_width + block.group_index().z;
        const uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
        const uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

        tile_offsets  += camera_id * tile_height * tile_width;
        render_alphas += camera_id * image_height * image_width;
        last_ids      += camera_id * image_height * image_width;

        const S px = (S)j + S(0.5);
        const S py = (S)i + S(0.5);
        const int32_t pix_id = i * image_width + j;

        bool inside = (i < image_height && j < image_width);
        bool done   = !inside;

        const int32_t range_start = tile_offsets[tile_id];
        const int32_t range_end   =
            (camera_id == C - 1 && tile_id == (int32_t)(tile_width * tile_height - 1))
                ? n_isects : tile_offsets[tile_id + 1];
        const uint32_t block_size  = block.size();
        const uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

        extern __shared__ int s[];
        int32_t *id_batch         = (int32_t *)s;
        vec3<S> *xy_opacity_batch = reinterpret_cast<vec3<S> *>(&id_batch[block_size]);
        vec3<S> *u_Ms_batch       = reinterpret_cast<vec3<S> *>(&xy_opacity_batch[block_size]);
        vec3<S> *v_Ms_batch       = reinterpret_cast<vec3<S> *>(&u_Ms_batch[block_size]);
        vec3<S> *w_Ms_batch       = reinterpret_cast<vec3<S> *>(&v_Ms_batch[block_size]);
        S *J_batch        = (S *)&w_Ms_batch[block_size];
        S *scale_batch    = (S *)&J_batch[block_size * 4];
        S *fvec_batch     = (S *)&scale_batch[block_size * 2];

        S T = S(1);
        uint32_t cur_idx = 0;

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
            if (__syncthreads_count(done) >= block_size) break;

            const uint32_t batch_start = range_start + block_size * b;
            const uint32_t idx         = batch_start + tr;

            if (idx < (uint32_t)range_end)
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

            const uint32_t batch_size = min(block_size, (uint32_t)range_end - batch_start);
            for (uint32_t t = 0; (t < batch_size) && !done; ++t)
            {
                const int32_t g     = id_batch[t];
                const vec3<S> xy_op = xy_opacity_batch[t];
                const S opac        = xy_op.z;
                const vec3<S> u_M   = u_Ms_batch[t];
                const vec3<S> v_M   = v_Ms_batch[t];
                const vec3<S> w_M   = w_Ms_batch[t];

                const vec3<S> h_u       = px * w_M - u_M;
                const vec3<S> h_v       = py * w_M - v_M;
                const vec3<S> ray_cross = glm::cross(h_u, h_v);
                if (ray_cross.z == 0.0) continue;

                const vec2<S> sv = {ray_cross.x / ray_cross.z, ray_cross.y / ray_cross.z};
                const S gw3 = sv.x * sv.x + sv.y * sv.y;
                const vec2<S> d = {xy_op.x - px, xy_op.y - py};
                const S gw2 = FILTER_INV_SQUARE * (d.x * d.x + d.y * d.y);
                const S sigma_val = S(0.5) * min(gw3, gw2);
                const S alpha = min(S(0.999), opac * exp(-sigma_val));
                if (sigma_val < S(0) || alpha < S(1) / S(255)) continue;

                const S next_T = T * (S(1) - alpha);
                if (next_T <= 1e-4) { done = true; break; }
                const S fac = alpha * T;

                if (ref_uu + ref_vv < S(1e-8))
                {
                    T = next_T; cur_idx = batch_start + t; continue;
                }

                // ── Eigendecompose sigma_ref: dominant eigenvector e2, eigenvalue lambda2 ──
                const S half_diff = S(0.5) * (ref_uu - ref_vv);
                const S disc      = sqrtf(half_diff * half_diff + ref_uv * ref_uv);
                const S lambda2   = max(S(0.5) * (ref_uu + ref_vv) + disc, S(1e-8));

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

                // ── w_screen = J_unit @ diag(s) @ freq_vec (covariant transform) ──
                const S j00 = J_batch[t*4+0], j01 = J_batch[t*4+1];
                const S j10 = J_batch[t*4+2], j11 = J_batch[t*4+3];
                const S s1  = scale_batch[t*2+0], s2 = scale_batch[t*2+1];
                const S fv0 = fvec_batch[t*2+0],  fv1 = fvec_batch[t*2+1];

                const S sfv0 = s1 * fv0, sfv1 = s2 * fv1;
                const S w0 = j00 * sfv0 + j01 * sfv1;
                const S w1 = j10 * sfv0 + j11 * sfv1;

                // ── Reference wavelength vector: e2 / sqrt(lambda2) ─────────
                const S inv_sqrt_l2 = rsqrtf(lambda2);
                const S w_ref0 = e2x * inv_sqrt_l2;
                const S w_ref1 = e2y * inv_sqrt_l2;

                // ── Loss: lambda2 * ||w_screen - w_ref||² (dimensionless) ───
                // Normalising by lambda2 (= 1/||w_ref||²) makes loss = 1 when
                // w_screen is off by one reference wavelength.
                // Clamped to 10.0 to prevent blow-up during early training.
                const S res0 = w0 - w_ref0, res1 = w1 - w_ref1;
                const S sq_loss = min(lambda2 * (res0 * res0 + res1 * res1), S(10.0));
                atomicAdd(&orient_loss[g], fac * sq_loss);

                T = next_T; cur_idx = batch_start + t;
            }
        }

        if (inside)
        {
            render_alphas[pix_id] = S(1) - T;
            last_ids[pix_id] = static_cast<int32_t>(cur_idx);
        }
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    freq_orient_fwd_tensor(
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
        const torch::Tensor &flatten_ids)
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
        TORCH_CHECK(means2d.dim() == 3, "freq_orient_fwd: packed mode not supported");

        const uint32_t C = tile_offsets.size(0);
        const uint32_t N = means2d.size(1);
        const uint32_t tile_height = tile_offsets.size(1);
        const uint32_t tile_width  = tile_offsets.size(2);
        const uint32_t n_isects    = flatten_ids.size(0);

        dim3 threads = {tile_size, tile_size, 1};
        dim3 blocks  = {C, tile_height, tile_width};

        auto f32 = means2d.options().dtype(torch::kFloat32);
        auto i32 = means2d.options().dtype(torch::kInt32);
        torch::Tensor orient_loss   = torch::zeros({(int64_t)(C * N)}, f32);
        torch::Tensor render_alphas = torch::zeros({C, image_height, image_width}, f32);
        torch::Tensor last_ids_out  = torch::zeros({C, image_height, image_width}, i32);

        // 84 B/thread: 4 (id) + 4*12 (vec3s) + 4*4 (J) + 2*4 (scales) + 2*4 (freq_vec)
        const uint32_t shared_mem =
            tile_size * tile_size *
            (sizeof(int32_t) + sizeof(vec3<float>) * 4 + sizeof(float) * (4 + 2 + 2));

        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        if (cudaFuncSetAttribute(freq_orient_fwd_kernel<float>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_mem) != cudaSuccess)
            AT_ERROR("freq_orient_fwd: failed to set shared mem (", shared_mem, " bytes)");

        freq_orient_fwd_kernel<float><<<blocks, threads, shared_mem, stream>>>(
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
            orient_loss.data_ptr<float>(),
            render_alphas.data_ptr<float>(),
            last_ids_out.data_ptr<int32_t>());

        return std::make_tuple(orient_loss, render_alphas, last_ids_out);
    }

} // namespace gsplat
