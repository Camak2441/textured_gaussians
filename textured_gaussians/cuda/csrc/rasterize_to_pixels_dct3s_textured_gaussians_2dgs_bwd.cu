#include "kernel_utils.h"
#include "helpers.cuh"
#include "types.cuh"
#include "utils.cuh"
#include "filters/dct3.cuh"
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/TensorAccessor.h>

namespace gsplat
{

    namespace cg = cooperative_groups;

    /****************************************************************************
     * Rasterization to Pixels Backward Pass Textured Gaussians (DCT3 + Sigmoid alpha)
     ****************************************************************************/
    template <uint32_t COLOR_DIM, typename S>
    __global__ void rasterize_to_pixels_bwd_dct3s_textured_gaussians_kernel(
        const uint32_t C,
        const uint32_t N,
        const uint32_t n_isects,
        const bool packed,
        const vec2<S> *__restrict__ means2d,
        const S *__restrict__ ray_transforms,
        const S *__restrict__ colors,
        const S *__restrict__ normals,
        const S *__restrict__ opacities,
        at::PackedTensorAccessor32<const S, 3, at::RestrictPtrTraits> textures,
        const vec2<S> texture_range,
        const bool texture_color,
        const bool texture_alpha,
        const bool texture_gradients,
        const S *__restrict__ backgrounds,
        const bool *__restrict__ masks,

        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        const uint32_t tile_width,
        const uint32_t tile_height,
        const int32_t *__restrict__ tile_offsets,
        const int32_t *__restrict__ flatten_ids,
        const S g_weight,

        const S *__restrict__ render_colors,
        const S *__restrict__ render_alphas,
        const int32_t *__restrict__ last_ids,
        const int32_t *__restrict__ median_ids,

        const S *__restrict__ v_render_colors,
        const S *__restrict__ v_render_alphas,
        const S *__restrict__ v_render_normals,
        const S *__restrict__ v_render_distort,
        const S *__restrict__ v_render_median,

        vec2<S> *__restrict__ v_means2d_abs,
        vec2<S> *__restrict__ v_means2d,
        S *__restrict__ v_ray_transforms,
        S *__restrict__ v_colors,
        S *__restrict__ v_opacities,
        at::PackedTensorAccessor32<S, 3, at::RestrictPtrTraits> v_textures,
        S *__restrict__ v_normals,
        S *__restrict__ v_densify)
    {
        auto block = cg::this_thread_block();
        uint32_t camera_id = block.group_index().x;
        uint32_t tile_id = block.group_index().y * tile_width + block.group_index().z;
        uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
        uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

        uint32_t texture_res = floor(sqrt(textures.size(1) * 2));

        tile_offsets += camera_id * tile_height * tile_width;
        render_colors += camera_id * image_height * image_width * COLOR_DIM;
        render_alphas += camera_id * image_height * image_width;
        last_ids += camera_id * image_height * image_width;
        median_ids += camera_id * image_height * image_width;
        v_render_colors += camera_id * image_height * image_width * COLOR_DIM;
        v_render_alphas += camera_id * image_height * image_width;
        v_render_normals += camera_id * image_height * image_width * 3;
        v_render_median += camera_id * image_height * image_width;

        if (backgrounds != nullptr)
            backgrounds += camera_id * COLOR_DIM;
        if (masks != nullptr)
            masks += camera_id * tile_height * tile_width;
        if (v_render_distort != nullptr)
            v_render_distort += camera_id * image_height * image_width;

        if (masks != nullptr && !masks[tile_id])
            return;

        const uint32_t alpha_channel = texture_color ? COLOR_DIM : 0;

        const S px = (S)j + S(0.5);
        const S py = (S)i + S(0.5);
        const int32_t pix_id = min(i * image_width + j, image_width * image_height - 1);
        bool inside = (i < image_height && j < image_width);

        int32_t range_start = tile_offsets[tile_id];
        int32_t range_end =
            (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
                ? n_isects
                : tile_offsets[tile_id + 1];
        const uint32_t block_size = block.size();
        const uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

        /**
         * Shared memory: ucos/vcos/ducos/dvcos stored column-major [coefficient][thread].
         */
        extern __shared__ int s[];
        int32_t *id_batch = (int32_t *)s;

        vec3<S> *xy_opacity_batch = reinterpret_cast<vec3<S> *>(&id_batch[block_size]);
        vec3<S> *u_Ms_batch = reinterpret_cast<vec3<S> *>(&xy_opacity_batch[block_size]);
        vec3<S> *v_Ms_batch = reinterpret_cast<vec3<S> *>(&u_Ms_batch[block_size]);
        vec3<S> *w_Ms_batch = reinterpret_cast<vec3<S> *>(&v_Ms_batch[block_size]);

        S *rgbs_batch = (S *)&w_Ms_batch[block_size];
        S *normals_batch = &rgbs_batch[block_size * COLOR_DIM];

        S *ucos = (S *)(&normals_batch[block_size * 3]);
        S *vcos = (S *)(&ucos[block_size * texture_res]);
        S *ducos = (S *)(&vcos[block_size * texture_res]);
        S *dvcos = (S *)(&ducos[block_size * texture_res]);

        S T_final = S(1) - render_alphas[pix_id];
        S T = T_final;
        S buffer[COLOR_DIM] = {S(0)};
        S buffer_normals[3] = {S(0)};

        const int32_t bin_final = inside ? last_ids[pix_id] : 0;
        const int32_t median_idx = inside ? median_ids[pix_id] : 0;

        S v_render_c[COLOR_DIM];
        GSPLAT_PRAGMA_UNROLL
        for (uint32_t k = 0; k < COLOR_DIM; ++k)
            v_render_c[k] = v_render_colors[pix_id * COLOR_DIM + k];

        const S v_render_a = v_render_alphas[pix_id];
        S v_render_n[3];
        GSPLAT_PRAGMA_UNROLL
        for (uint32_t k = 0; k < 3; ++k)
            v_render_n[k] = v_render_normals[pix_id * 3 + k];

        S v_distort = S(0);
        S accum_d, accum_w;
        S accum_d_buffer, accum_w_buffer, distort_buffer;
        if (v_render_distort != nullptr)
        {
            v_distort = v_render_distort[pix_id];
            accum_d_buffer = render_colors[pix_id * COLOR_DIM + COLOR_DIM - 1];
            accum_d = accum_d_buffer;
            accum_w_buffer = render_alphas[pix_id];
            accum_w = accum_w_buffer;
            distort_buffer = S(0);
        }

        S v_median = v_render_median[pix_id];

        const uint32_t tr = block.thread_rank();

        ucos += tr;
        vcos += tr;
        ducos += tr;
        dvcos += tr;

        cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
        const int32_t warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());

        for (uint32_t b = 0; b < num_batches; ++b)
        {
            block.sync();

            const int32_t batch_end = range_end - 1 - block_size * b;
            const int32_t batch_size = min(block_size, batch_end + 1 - range_start);
            const int32_t idx = batch_end - tr;

            if (idx >= range_start)
            {
                int32_t g = flatten_ids[idx];
                id_batch[tr] = g;
                const vec2<S> xy = means2d[g];
                const S opac = opacities[g];
                xy_opacity_batch[tr] = {xy.x, xy.y, opac};
                u_Ms_batch[tr] = {ray_transforms[g * 9], ray_transforms[g * 9 + 1], ray_transforms[g * 9 + 2]};
                v_Ms_batch[tr] = {ray_transforms[g * 9 + 3], ray_transforms[g * 9 + 4], ray_transforms[g * 9 + 5]};
                w_Ms_batch[tr] = {ray_transforms[g * 9 + 6], ray_transforms[g * 9 + 7], ray_transforms[g * 9 + 8]};
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k)
                    rgbs_batch[tr * COLOR_DIM + k] = colors[g * COLOR_DIM + k];
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t k = 0; k < 3; ++k)
                    normals_batch[tr * 3 + k] = normals[g * 3 + k];
            }
            block.sync();

            for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size; ++t)
            {
                int32_t g = id_batch[t];

                bool valid = inside;
                if (batch_end - t > bin_final)
                    valid = 0;

                S alpha, opac, vis, gaussian_kernel;
                S gauss_weight_3d, gauss_weight_2d, gauss_weight;
                vec2<S> s, d;
                vec3<S> h_u, h_v, ray_cross, w_M;
                S u, v;
                int32_t valid_texture = -1;
                S alpha_scaling_factor = S(0);

                if (valid)
                {
                    vec3<S> xy_opac = xy_opacity_batch[t];
                    opac = xy_opac.z;
                    const vec3<S> u_M = u_Ms_batch[t];
                    const vec3<S> v_M = v_Ms_batch[t];
                    w_M = w_Ms_batch[t];
                    h_u = px * w_M - u_M;
                    h_v = py * w_M - v_M;
                    ray_cross = glm::cross(h_u, h_v);
                    if (ray_cross.z == 0.0)
                        valid = false;
                    s = {ray_cross.x / ray_cross.z, ray_cross.y / ray_cross.z};

                    valid_texture = (s.x >= -texture_range.x && s.x <= texture_range.x &&
                                     s.y >= -texture_range.y && s.y <= texture_range.y)
                                        ? 1
                                        : -1;

                    u = (S)((s.x + texture_range.x) / (texture_range.x * 2) * (texture_res - 2) + S(1)) / texture_res;
                    v = (S)((s.y + texture_range.y) / (texture_range.y * 2) * (texture_res - 2) + S(1)) / texture_res;

                    if (valid_texture > 0)
                    {
                        dct3::grad_precompute(texture_res, u, v, ucos, vcos, ducos, dvcos, block_size);
                        if (texture_alpha)
                        {
                            // Sigmoid activation: alpha_scaling_factor = sigmoid(raw_alpha)
                            alpha_scaling_factor = sigmoid(dct3::sample(textures, texture_res, g, u, v, ucos, vcos, alpha_channel, block_size));
                        }
                        else
                        {
                            alpha_scaling_factor = S(1);
                        }
                    }
                    else
                    {
                        alpha_scaling_factor = S(1);
                    }

                    gauss_weight_3d = s.x * s.x + s.y * s.y;
                    d = {xy_opac.x - px, xy_opac.y - py};
                    gauss_weight_2d = FILTER_INV_SQUARE * (d.x * d.x + d.y * d.y);
                    gauss_weight = min(gauss_weight_3d, gauss_weight_2d);

                    const S sigma = S(0.5) * gauss_weight;
                    gaussian_kernel = exp(-sigma);
                    vis = S(0.998) - g_weight + g_weight * gaussian_kernel;
                    alpha = min(S(0.999), opac * vis * alpha_scaling_factor);

                    if (sigma < S(0) || alpha < S(1) / S(255))
                        valid = false;
                }

                if (!warp.any(valid))
                    continue;

                S v_rgb_local[COLOR_DIM] = {S(0)};
                S v_normal_local[3] = {S(0)};
                vec3<S> v_u_M_local = {S(0), S(0), S(0)};
                vec3<S> v_v_M_local = {S(0), S(0), S(0)};
                vec3<S> v_w_M_local = {S(0), S(0), S(0)};
                vec2<S> v_xy_local = {S(0), S(0)};
                vec2<S> v_xy_abs_local = {S(0), S(0)};
                S v_opacity_local = S(0);

                S tex_colors[COLOR_DIM] = {S(0)};
                vec2<S> v_s_tex = {S(0), S(0)};
                S deltas[COLOR_DIM] = {S(0)};
                S ra = S(0), fac = S(0);
                bool clip = false;
                S v_asf = S(0);

                /**
                 * Part A: transmittance update and per-channel deltas
                 */
                if (valid)
                {
                    if (batch_end - t == median_idx)
                        v_rgb_local[COLOR_DIM - 1] += v_median;

                    ra = S(1) / (S(1) - alpha);
                    T *= ra;
                    fac = alpha * T;
                    clip = (opac * vis * alpha_scaling_factor <= S(0.999));

                    GSPLAT_PRAGMA_UNROLL
                    for (uint32_t k = 0; k < COLOR_DIM; ++k)
                    {
                        deltas[k] = fac * v_render_c[k];
                        v_rgb_local[k] += deltas[k];
                    }
                }

                /**
                 * PASS 1 (warp-level): color texture
                 */
                if (texture_color)
                {
                    const bool do_tex = valid && (valid_texture > 0);
                    int index = 0;
                    for (uint32_t tj = 0; tj < texture_res; ++tj)
                    {
                        S vj = S(0), dvj = S(0);
                        if (do_tex)
                        {
                            vj = vcos[tj * block_size];
                            if (texture_gradients)
                                dvj = dvcos[tj * block_size];
                        }
                        for (uint32_t ti = 0; ti < texture_res - tj; ++ti)
                        {
                            S ui = S(0), dui = S(0);
                            if (do_tex)
                            {
                                ui = ucos[ti * block_size];
                                if (texture_gradients)
                                    dui = ducos[ti * block_size];
                            }
                            const S uivj = ui * vj;
                            const S duivj = dui * vj;
                            const S uidvj = ui * dvj;
                            GSPLAT_PRAGMA_UNROLL
                            for (uint32_t tk = 0; tk < COLOR_DIM; ++tk)
                            {
                                const S c = textures[g][index][tk];
                                if (do_tex)
                                    tex_colors[tk] += c * uivj;

                                const S per_pix = do_tex ? deltas[tk] * uivj : S(0);
                                const S wsum = cg::reduce(warp, per_pix, cg::plus<S>());
                                if (warp.thread_rank() == 0)
                                    gpuAtomicAdd(&v_textures[g][index][tk], wsum);

                                if (do_tex && texture_gradients)
                                {
                                    v_s_tex.x += duivj * c * deltas[tk];
                                    v_s_tex.y += uidvj * c * deltas[tk];
                                }
                            }
                            index++;
                        }
                    }
                }

                /**
                 * Part B: v_alpha, normals, geometry gradients, buffer updates
                 */
                if (valid)
                {
                    S v_alpha = S(0);
                    for (uint32_t k = 0; k < COLOR_DIM; ++k)
                    {
                        const S full_color = tex_colors[k] + rgbs_batch[t * COLOR_DIM + k];
                        v_alpha += (full_color * T - buffer[k] * ra) * v_render_c[k];
                    }

                    GSPLAT_PRAGMA_UNROLL
                    for (uint32_t k = 0; k < 3; ++k)
                        v_normal_local[k] = fac * v_render_n[k];

                    for (uint32_t k = 0; k < 3; ++k)
                        v_alpha += (normals_batch[t * 3 + k] * T - buffer_normals[k] * ra) * v_render_n[k];

                    v_alpha += T_final * ra * v_render_a;

                    if (backgrounds != nullptr)
                    {
                        S accum = S(0);
                        GSPLAT_PRAGMA_UNROLL
                        for (uint32_t k = 0; k < COLOR_DIM; ++k)
                            accum += backgrounds[k] * v_render_c[k];
                        v_alpha += -T_final * ra * accum;
                    }

                    if (v_render_distort != nullptr)
                    {
                        S depth = rgbs_batch[t * COLOR_DIM + COLOR_DIM - 1];
                        S dl_dw = S(2) * (S(2) * (depth * accum_w_buffer - accum_d_buffer) + (accum_d - depth * accum_w));
                        v_alpha += (dl_dw * T - distort_buffer * ra) * v_distort;
                        accum_d_buffer -= fac * depth;
                        accum_w_buffer -= fac;
                        distort_buffer += dl_dw * fac;
                        v_rgb_local[COLOR_DIM - 1] += S(2) * fac * (S(2) - S(2) * T - accum_w + fac) * v_distort;
                    }

                    if (clip)
                    {
                        S v_depth = S(0);
                        const S v_G = opac * v_alpha * alpha_scaling_factor;

                        if (gauss_weight_3d <= gauss_weight_2d)
                        {
                            const vec2<S> v_s = {
                                v_G * -gaussian_kernel * s.x + v_depth * w_M.x,
                                v_G * -gaussian_kernel * s.y + v_depth * w_M.y};
                            const vec3<S> v_z_w_M = {s.x, s.y, S(1)};
                            const S v_sx_pz = v_s.x / ray_cross.z;
                            const S v_sy_pz = v_s.y / ray_cross.z;
                            const vec3<S> v_ray_cross = {v_sx_pz, v_sy_pz, -(v_sx_pz * s.x + v_sy_pz * s.y)};
                            const vec3<S> v_h_u = glm::cross(h_v, v_ray_cross);
                            const vec3<S> v_h_v = glm::cross(v_ray_cross, h_u);
                            v_u_M_local = {-v_h_u.x, -v_h_u.y, -v_h_u.z};
                            v_v_M_local = {-v_h_v.x, -v_h_v.y, -v_h_v.z};
                            v_w_M_local = {
                                px * v_h_u.x + py * v_h_v.x + v_depth * v_z_w_M.x,
                                px * v_h_u.y + py * v_h_v.y + v_depth * v_z_w_M.y,
                                px * v_h_u.z + py * v_h_v.z + v_depth * v_z_w_M.z};
                        }
                        else
                        {
                            const S v_G_ddelx = -gaussian_kernel * FILTER_INV_SQUARE * d.x;
                            const S v_G_ddely = -gaussian_kernel * FILTER_INV_SQUARE * d.y;
                            v_xy_local = {v_G * v_G_ddelx, v_G * v_G_ddely};
                            if (v_means2d_abs != nullptr)
                                v_xy_abs_local = {abs(v_xy_local.x), abs(v_xy_local.y)};
                        }
                        v_opacity_local = vis * v_alpha * alpha_scaling_factor;

                        // Store for PASS 2
                        if (texture_alpha && valid_texture > 0)
                            v_asf = vis * opac * v_alpha;
                    }

                    GSPLAT_PRAGMA_UNROLL
                    for (uint32_t k = 0; k < COLOR_DIM; ++k)
                        buffer[k] += (tex_colors[k] + rgbs_batch[t * COLOR_DIM + k]) * fac;

                    GSPLAT_PRAGMA_UNROLL
                    for (uint32_t k = 0; k < 3; ++k)
                        buffer_normals[k] += normals_batch[t * 3 + k] * fac;
                }

                /**
                 * PASS 2 (warp-level): alpha texture
                 * Extra sigmoid guard: only propagate UV gradients when sigmoid is not saturated.
                 */
                if (texture_alpha)
                {
                    // sigmoid saturation guard: gradient vanishes at 0 and 1
                    const bool do_alpha = valid && clip && (valid_texture > 0) &&
                                          (alpha_scaling_factor > S(0)) && (alpha_scaling_factor < S(1));
                    int index = 0;
                    for (uint32_t tj = 0; tj < texture_res; ++tj)
                    {
                        S vj = S(0), dvj = S(0);
                        if (do_alpha)
                        {
                            vj = vcos[tj * block_size];
                            if (texture_gradients)
                                dvj = dvcos[tj * block_size];
                        }
                        for (uint32_t ti = 0; ti < texture_res - tj; ++ti)
                        {
                            S ui = S(0), dui = S(0);
                            if (do_alpha)
                            {
                                ui = ucos[ti * block_size];
                                if (texture_gradients)
                                    dui = ducos[ti * block_size];
                            }
                            const S uivj = ui * vj;

                            // For warp-reduction we also need the non-saturated threads' contributions.
                            // The alpha texture coeff gradient uses valid&&clip&&valid_texture (broader).
                            const bool do_alpha_coeff = valid && clip && (valid_texture > 0);
                            const S per_pix = do_alpha_coeff ? v_asf * (ui * vj) : S(0);
                            const S wsum = cg::reduce(warp, per_pix, cg::plus<S>());
                            if (warp.thread_rank() == 0)
                                gpuAtomicAdd(&v_textures[g][index][alpha_channel], wsum);

                            if (do_alpha && texture_gradients)
                            {
                                const S c_alpha = textures[g][index][alpha_channel];
                                const S duivj = dui * vj;
                                const S uidvj = ui * dvj;
                                v_s_tex.x += duivj * c_alpha * v_asf;
                                v_s_tex.y += uidvj * c_alpha * v_asf;
                            }
                            index++;
                        }
                    }
                }

                /**
                 * Projective backprop: v_s_tex → v_ray_transforms
                 */
                if (valid && texture_gradients && valid_texture > 0)
                {
                    const S du_dsx = (texture_res - 2) / (texture_range.x * 2) / texture_res;
                    const S dv_dsy = (texture_res - 2) / (texture_range.y * 2) / texture_res;
                    v_s_tex.x *= du_dsx;
                    v_s_tex.y *= dv_dsy;

                    const S v_sx_pz = v_s_tex.x / ray_cross.z;
                    const S v_sy_pz = v_s_tex.y / ray_cross.z;
                    const vec3<S> v_ray_cross_tex = {v_sx_pz, v_sy_pz, -(v_sx_pz * s.x + v_sy_pz * s.y)};
                    const vec3<S> v_h_u_tex = glm::cross(h_v, v_ray_cross_tex);
                    const vec3<S> v_h_v_tex = glm::cross(v_ray_cross_tex, h_u);

                    v_u_M_local.x += -v_h_u_tex.x;
                    v_u_M_local.y += -v_h_u_tex.y;
                    v_u_M_local.z += -v_h_u_tex.z;
                    v_v_M_local.x += -v_h_v_tex.x;
                    v_v_M_local.y += -v_h_v_tex.y;
                    v_v_M_local.z += -v_h_v_tex.z;
                    v_w_M_local.x += px * v_h_u_tex.x + py * v_h_v_tex.x;
                    v_w_M_local.y += px * v_h_u_tex.y + py * v_h_v_tex.y;
                    v_w_M_local.z += px * v_h_u_tex.z + py * v_h_v_tex.z;
                }

                warpSum<COLOR_DIM, S>(v_rgb_local, warp);
                warpSum<3, S>(v_normal_local, warp);
                warpSum<decltype(warp), S>(v_xy_local, warp);
                warpSum<decltype(warp), S>(v_u_M_local, warp);
                warpSum<decltype(warp), S>(v_v_M_local, warp);
                warpSum<decltype(warp), S>(v_w_M_local, warp);
                if (v_means2d_abs != nullptr)
                    warpSum<decltype(warp), S>(v_xy_abs_local, warp);
                warpSum<decltype(warp), S>(v_opacity_local, warp);

                if (warp.thread_rank() == 0)
                {
                    S *v_rgb_ptr = (S *)(v_colors) + COLOR_DIM * g;
                    GSPLAT_PRAGMA_UNROLL
                    for (uint32_t k = 0; k < COLOR_DIM; ++k)
                        gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);

                    S *v_normal_ptr = (S *)(v_normals) + 3 * g;
                    GSPLAT_PRAGMA_UNROLL
                    for (uint32_t k = 0; k < 3; ++k)
                        gpuAtomicAdd(v_normal_ptr + k, v_normal_local[k]);

                    S *v_ray_transforms_ptr = (S *)(v_ray_transforms) + 9 * g;
                    gpuAtomicAdd(v_ray_transforms_ptr, v_u_M_local.x);
                    gpuAtomicAdd(v_ray_transforms_ptr + 1, v_u_M_local.y);
                    gpuAtomicAdd(v_ray_transforms_ptr + 2, v_u_M_local.z);
                    gpuAtomicAdd(v_ray_transforms_ptr + 3, v_v_M_local.x);
                    gpuAtomicAdd(v_ray_transforms_ptr + 4, v_v_M_local.y);
                    gpuAtomicAdd(v_ray_transforms_ptr + 5, v_v_M_local.z);
                    gpuAtomicAdd(v_ray_transforms_ptr + 6, v_w_M_local.x);
                    gpuAtomicAdd(v_ray_transforms_ptr + 7, v_w_M_local.y);
                    gpuAtomicAdd(v_ray_transforms_ptr + 8, v_w_M_local.z);

                    S *v_xy_ptr = (S *)(v_means2d) + 2 * g;
                    gpuAtomicAdd(v_xy_ptr, v_xy_local.x);
                    gpuAtomicAdd(v_xy_ptr + 1, v_xy_local.y);

                    if (v_means2d_abs != nullptr)
                    {
                        S *v_xy_abs_ptr = (S *)(v_means2d_abs) + 2 * g;
                        gpuAtomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
                        gpuAtomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                    }

                    gpuAtomicAdd(v_opacities + g, v_opacity_local);
                }

                if (valid)
                {
                    S *v_densify_ptr = (S *)(v_densify) + 2 * g;
                    S *v_ray_transforms_ptr = (S *)(v_ray_transforms) + 9 * g;
                    S depth = w_M.z;
                    v_densify_ptr[0] = v_ray_transforms_ptr[2] * depth;
                    v_densify_ptr[1] = v_ray_transforms_ptr[5] * depth;
                }
            }
        }
    }

    template <uint32_t CDIM>
    std::tuple<
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    call_bwd_dct3s_g_kernel_with_dim(
        const torch::Tensor &means2d,
        const torch::Tensor &ray_transforms,
        const torch::Tensor &colors,
        const torch::Tensor &opacities,
        const torch::Tensor &textures,
        const vec2<float> texture_range,
        const bool texture_color,
        const bool texture_alpha,
        const bool texture_gradients,
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
        bool absgrad)
    {
        GSPLAT_DEVICE_GUARD(means2d);
        GSPLAT_CHECK_INPUT(means2d);
        GSPLAT_CHECK_INPUT(ray_transforms);
        GSPLAT_CHECK_INPUT(colors);
        GSPLAT_CHECK_INPUT(opacities);
        GSPLAT_CHECK_INPUT(textures);
        GSPLAT_CHECK_INPUT(normals);
        GSPLAT_CHECK_INPUT(densify);
        GSPLAT_CHECK_INPUT(tile_offsets);
        GSPLAT_CHECK_INPUT(flatten_ids);
        GSPLAT_CHECK_INPUT(render_colors);
        GSPLAT_CHECK_INPUT(render_alphas);
        GSPLAT_CHECK_INPUT(last_ids);
        GSPLAT_CHECK_INPUT(median_ids);
        GSPLAT_CHECK_INPUT(v_render_colors);
        GSPLAT_CHECK_INPUT(v_render_alphas);
        GSPLAT_CHECK_INPUT(v_render_normals);
        GSPLAT_CHECK_INPUT(v_render_distort);
        GSPLAT_CHECK_INPUT(v_render_median);
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
        uint32_t n_isects = flatten_ids.size(0);
        uint32_t COLOR_DIM = colors.size(-1);
        uint32_t tile_height = tile_offsets.size(1);
        uint32_t tile_width = tile_offsets.size(2);
        uint32_t texture_res = floor(sqrt(textures.size(1) * 2));

        dim3 threads = {tile_size, tile_size, 1};
        dim3 blocks = {C, tile_height, tile_width};

        torch::Tensor v_means2d = torch::zeros_like(means2d);
        torch::Tensor v_ray_transforms = torch::zeros_like(ray_transforms);
        torch::Tensor v_colors = torch::zeros_like(colors);
        torch::Tensor v_textures = torch::zeros_like(textures);
        torch::Tensor v_normals = torch::zeros_like(normals);
        torch::Tensor v_opacities = torch::zeros_like(opacities);
        torch::Tensor v_means2d_abs;
        if (absgrad)
            v_means2d_abs = torch::zeros_like(means2d);
        torch::Tensor v_densify = torch::zeros_like(densify);

        if (n_isects)
        {
            const uint32_t shared_mem =
                tile_size * tile_size *
                (sizeof(int32_t) + sizeof(vec3<float>) * 4 +
                 sizeof(float) * COLOR_DIM + sizeof(float) * 3 +
                 sizeof(float) * (texture_res * 4));
            at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

            if (cudaFuncSetAttribute(
                    rasterize_to_pixels_bwd_dct3s_textured_gaussians_kernel<CDIM, float>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    shared_mem) != cudaSuccess)
            {
                AT_ERROR("Failed to set maximum shared memory size (requested ", shared_mem, " bytes), try lowering tile_size.");
            }
            rasterize_to_pixels_bwd_dct3s_textured_gaussians_kernel<CDIM, float>
                <<<blocks, threads, shared_mem, stream>>>(
                    C, N, n_isects, packed,
                    reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
                    ray_transforms.data_ptr<float>(),
                    colors.data_ptr<float>(),
                    normals.data_ptr<float>(),
                    opacities.data_ptr<float>(),
                    textures.packed_accessor32<const float, 3, at::RestrictPtrTraits>(),
                    texture_range, texture_color, texture_alpha, texture_gradients,
                    backgrounds.has_value() ? backgrounds.value().data_ptr<float>() : nullptr,
                    masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
                    image_width, image_height, tile_size, tile_width, tile_height,
                    tile_offsets.data_ptr<int32_t>(),
                    flatten_ids.data_ptr<int32_t>(),
                    g_weight,
                    render_colors.data_ptr<float>(),
                    render_alphas.data_ptr<float>(),
                    last_ids.data_ptr<int32_t>(),
                    median_ids.data_ptr<int32_t>(),
                    v_render_colors.data_ptr<float>(),
                    v_render_alphas.data_ptr<float>(),
                    v_render_normals.data_ptr<float>(),
                    v_render_distort.data_ptr<float>(),
                    v_render_median.data_ptr<float>(),
                    absgrad ? reinterpret_cast<vec2<float> *>(v_means2d_abs.data_ptr<float>()) : nullptr,
                    reinterpret_cast<vec2<float> *>(v_means2d.data_ptr<float>()),
                    v_ray_transforms.data_ptr<float>(),
                    v_colors.data_ptr<float>(),
                    v_opacities.data_ptr<float>(),
                    v_textures.packed_accessor32<float, 3, at::RestrictPtrTraits>(),
                    v_normals.data_ptr<float>(),
                    v_densify.data_ptr<float>());
        }

        return std::make_tuple(v_means2d_abs, v_means2d, v_ray_transforms, v_colors,
                               v_opacities, v_textures, v_normals, v_densify);
    }

    std::tuple<
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    rasterize_to_pixels_bwd_dct3s_textured_gaussians_tensor(
        const torch::Tensor &means2d,
        const torch::Tensor &ray_transforms,
        const torch::Tensor &colors,
        const torch::Tensor &opacities,
        const torch::Tensor &textures,
        const float texture_range_x,
        const float texture_range_y,
        const bool texture_color,
        const bool texture_alpha,
        const bool texture_gradients,
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
        bool absgrad)
    {
        GSPLAT_CHECK_INPUT(colors);
        uint32_t COLOR_DIM = colors.size(-1);

#define __GS__CALL_(N)                                       \
    case N:                                                  \
        return call_bwd_dct3s_g_kernel_with_dim<N>(          \
            means2d, ray_transforms, colors, opacities,      \
            textures,                                        \
            vec2<float>(texture_range_x, texture_range_y),   \
            texture_color, texture_alpha, texture_gradients, \
            normals, densify, backgrounds, masks,            \
            image_width, image_height, tile_size,            \
            tile_offsets, flatten_ids, g_weight,             \
            render_colors, render_alphas, last_ids,          \
            median_ids, v_render_colors, v_render_alphas,    \
            v_render_normals, v_render_distort,              \
            v_render_median, absgrad);

        switch (COLOR_DIM)
        {
            __GS__CALL_(1)
            __GS__CALL_(2)
            __GS__CALL_(3)
            __GS__CALL_(4)
            __GS__CALL_(5)
        default:
            AT_ERROR("Unsupported number of channels: ", COLOR_DIM);
        }
    }

} // namespace gsplat