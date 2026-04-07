#include "bindings.h"
#include "helpers.cuh"
#include "types.cuh"
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/TensorAccessor.h>

#define FILTER_INV_SQUARE 2.0f

namespace gsplat
{

    namespace cg = cooperative_groups;

    /****************************************************************************
     * Rasterization to World-and-View Samples Forward Pass
     ****************************************************************************/

    /**
     * This function generates per-pixel world-space 3D intersection coordinates
     * AND the unit viewing direction from each intersection point toward the camera.
     *
     * For each pixel, it finds the ray-Gaussian intersections and records:
     *   q_cam    = K^{-1} * (KWH * [s_u, s_v, 1]^T)
     *   q_world  = R_c2w * q_cam + t_c2w
     *   view_dir = normalize(cam_pos - q_world)   (unit vector toward camera)
     *
     * Output layout per sample: [x, y, z, vx, vy, vz]
     *   [0..2] world-space XYZ of the intersection point
     *   [3..5] unit viewing direction (from surface toward camera)
     */
    template <typename S>
    __global__ void rasterize_to_samples_world_and_view_fwd_implicit_textured_gaussians_kernel(
        const uint32_t C,                     // number of cameras
        const uint32_t N,                     // number of gaussians
        const uint32_t n_isects,              // number of ray-primitive intersections.
        const bool packed,                    // whether the input tensors are packed
        const vec2<S> *__restrict__ means2d,  // Projected Gaussian means. [C, N, 2] if packed is False, [nnz, 2] if packed is True.
        const S *__restrict__ ray_transforms, // rows of KWH matrix per gaussian. [C, N, 3, 3] or [nnz, 3, 3]
                                              // This is (KWH)^{-1} in the paper (takes screen [x,y] and map to [u,v])
        const S *__restrict__ viewmats,       // world-to-camera transforms. [C, 4, 4] row-major [R t; 0 1]
        const S *__restrict__ Ks,             // camera intrinsics. [C, 3, 3] row-major [fx 0 cx; 0 fy cy; 0 0 1]
        const S *__restrict__ opacities,      // [C, N] or [nnz]
        const bool *__restrict__ masks,       // [C, tile_height, tile_width]
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        const uint32_t tile_width,
        const uint32_t tile_height,
        const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
        const int32_t *__restrict__ flatten_ids,  // [n_isects]
        const uint32_t num_texture_samples,
        const S sample_alpha_threshold,

        // outputs
        int32_t *__restrict__ sample_counts,       // [C, image_height, image_width]
        int32_t *__restrict__ sample_gaussian_ids, // [C, image_height, image_width]
        S *__restrict__ texture_inputs             // [C, image_height, image_width, num_texture_samples, 6]
                                                   // layout per sample: [x, y, z, vx, vy, vz]
    )
    {
        /**
         * ==============================
         * Thread and block setup
         * ==============================
         */
        auto block = cg::this_thread_block();
        int32_t camera_id = block.group_index().x;
        int32_t tile_id = block.group_index().y * tile_width + block.group_index().z;
        uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
        uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

        tile_offsets += camera_id * tile_height * tile_width;
        sample_counts += camera_id * image_height * image_width;
        sample_gaussian_ids += camera_id * image_height * image_width;
        texture_inputs += camera_id * image_height * image_width * num_texture_samples * 6;

        if (masks != nullptr)
        {
            masks += camera_id * tile_height * tile_width;
        }

        /**
         * ==============================
         * Load per-camera intrinsics and camera-to-world transform.
         *
         * viewmats is [C, 4, 4] row-major: [R t; 0 1] where R is world-to-camera rotation
         *   and t is world-to-camera translation, so:  p_cam = R * p_world + t
         * Camera-to-world: p_world = R^T * p_cam - R^T * t
         * Camera position in world: cam_pos = t_c2w = -R^T * t
         *
         * Ks is [C, 3, 3] row-major: [fx 0 cx; 0 fy cy; 0 0 1]
         * K^{-1} * v = [(v.x - cx*v.z)/fx, (v.y - cy*v.z)/fy, v.z]
         * ==============================
         */
        const S *cam_viewmat = viewmats + camera_id * 16;
        const S *cam_K = Ks + camera_id * 9;

        // World-to-camera rotation R (row-major input)
        // R[row][col] = cam_viewmat[row * 4 + col]
        const S r00 = cam_viewmat[0], r01 = cam_viewmat[1], r02 = cam_viewmat[2];
        const S r10 = cam_viewmat[4], r11 = cam_viewmat[5], r12 = cam_viewmat[6];
        const S r20 = cam_viewmat[8], r21 = cam_viewmat[9], r22 = cam_viewmat[10];
        // World-to-camera translation
        const S tx = cam_viewmat[3], ty = cam_viewmat[7], tz = cam_viewmat[11];

        // Camera position in world space: t_c2w = -R^T * t
        // Also serves as the origin for computing viewing directions.
        const S c2w_t_x = -(r00 * tx + r10 * ty + r20 * tz);
        const S c2w_t_y = -(r01 * tx + r11 * ty + r21 * tz);
        const S c2w_t_z = -(r02 * tx + r12 * ty + r22 * tz);

        // Camera intrinsics
        const S fx = cam_K[0], fy = cam_K[4], cx = cam_K[2], cy = cam_K[5];

        // find the center of the pixel
        S px = (S)j + 0.5f;
        S py = (S)i + 0.5f;
        int32_t pix_id = i * image_width + j;

        bool inside = (i < image_height && j < image_width);
        bool done = !inside;

        if (masks != nullptr && inside && !masks[tile_id])
        {
            return;
        }

        int32_t range_start = tile_offsets[tile_id];
        int32_t range_end =
            (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
                ? n_isects
                : tile_offsets[tile_id + 1];
        const uint32_t block_size = block.size();
        uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

        /**
         * ==============================
         * Shared memory layout:
         * | gaussian indices | x : y : alpha | u_M | v_M | w_M |
         * ==============================
         */
        extern __shared__ int s[];
        int32_t *id_batch = (int32_t *)s; // [block_size]

        vec3<S> *xy_opacity_batch =
            reinterpret_cast<vec3<float> *>(&id_batch[block_size]); // [block_size]

        vec3<S> *u_Ms_batch = reinterpret_cast<vec3<float> *>(&xy_opacity_batch[block_size]); // [block_size]
        vec3<S> *v_Ms_batch = reinterpret_cast<vec3<float> *>(&u_Ms_batch[block_size]);       // [block_size]
        vec3<S> *w_Ms_batch = reinterpret_cast<vec3<float> *>(&v_Ms_batch[block_size]);       // [block_size]

        uint32_t tr = block.thread_rank();

        for (uint32_t b = 0; b < num_batches; ++b)
        {
            if (__syncthreads_count(done) >= block_size)
            {
                break;
            }

            uint32_t batch_start = range_start + block_size * b;
            uint32_t idx = batch_start + tr;

            if (idx < range_end)
            {
                int32_t g = flatten_ids[idx];
                id_batch[tr] = g;
                const vec2<S> xy = means2d[g];
                const S opac = opacities[g];
                xy_opacity_batch[tr] = {xy.x, xy.y, opac};
                u_Ms_batch[tr] = {
                    ray_transforms[g * 9], ray_transforms[g * 9 + 1], ray_transforms[g * 9 + 2]};
                v_Ms_batch[tr] = {
                    ray_transforms[g * 9 + 3], ray_transforms[g * 9 + 4], ray_transforms[g * 9 + 5]};
                w_Ms_batch[tr] = {
                    ray_transforms[g * 9 + 6], ray_transforms[g * 9 + 7], ray_transforms[g * 9 + 8]};
            }

            block.sync();

            /**
             * ==================================================
             * Forward rasterization pass:
             * ==================================================
             *
             * 1. Compute homogeneous plane parameters:
             *    h_u = p_x * M_w - M_u
             *    h_v = p_y * M_w - M_v
             *
             * 2. Compute intersection:
             *    zeta = h_u × h_v
             *    s_uv = [zeta_1/zeta_3, zeta_2/zeta_3]
             *
             * 3. Convert s_uv to world space:
             *    kwh_s    = KWH * [s_u, s_v, 1]^T
             *    q_cam    = K^{-1} * kwh_s
             *    q_world  = R_c2w * q_cam + t_c2w
             *
             * 4. Compute unit viewing direction:
             *    delta    = cam_pos - q_world   (vector from surface to camera)
             *    view_dir = delta / ||delta||
             */
            uint32_t batch_size = min(block_size, range_end - batch_start);
            for (uint32_t t = 0; (t < batch_size) && !done; ++t)
            {
                int32_t g = id_batch[t];

                const vec3<S> xy_opac = xy_opacity_batch[t];
                const S opac = xy_opac.z;

                const vec3<S> u_M = u_Ms_batch[t];
                const vec3<S> v_M = v_Ms_batch[t];
                const vec3<S> w_M = w_Ms_batch[t];

                const vec3<S> h_u = px * w_M - u_M;
                const vec3<S> h_v = py * w_M - v_M;

                const vec3<S> ray_cross = glm::cross(h_u, h_v);
                if (ray_cross.z == 0.0)
                    continue;

                const vec2<S> s = vec2<S>(ray_cross.x / ray_cross.z, ray_cross.y / ray_cross.z);

                int32_t valid_texture = 0;

                const S gauss_weight_3d = s.x * s.x + s.y * s.y;
                if (gauss_weight_3d <= 9.0)
                    valid_texture = 1;

                const vec2<S> d = {xy_opac.x - px, xy_opac.y - py};
                const S gauss_weight_2d = FILTER_INV_SQUARE * (d.x * d.x + d.y * d.y);

                const S gauss_weight = min(gauss_weight_3d, gauss_weight_2d);

                const S sigma = 0.5f * gauss_weight;
                const S alpha_approx = min(0.999f, opac * __expf(-sigma));

                if (valid_texture > 0 && alpha_approx > sample_alpha_threshold)
                {
                    // KWH * [s.x, s.y, 1]^T: dot each row of KWH with [s.x, s.y, 1]
                    const S sv1_x = u_M.x * s.x + u_M.y * s.y + u_M.z;
                    const S sv1_y = v_M.x * s.x + v_M.y * s.y + v_M.z;
                    const S sv1_z = w_M.x * s.x + w_M.y * s.y + w_M.z;

                    // K^{-1} * kwh_s: unproject to camera space
                    const S qcam_x = (sv1_x - cx * sv1_z) / fx;
                    const S qcam_y = (sv1_y - cy * sv1_z) / fy;
                    const S qcam_z = sv1_z;

                    // R_c2w * q_cam + t_c2w: camera to world
                    // R_c2w = R^T, so R_c2w[row][col] = R[col][row]
                    const vec3<S> world_pos = {
                        r00 * qcam_x + r10 * qcam_y + r20 * qcam_z + c2w_t_x,
                        r01 * qcam_x + r11 * qcam_y + r21 * qcam_z + c2w_t_y,
                        r02 * qcam_x + r12 * qcam_y + r22 * qcam_z + c2w_t_z,
                    };

                    // Unit viewing direction: normalize(cam_pos - world_pos)
                    // cam_pos = (c2w_t_x, c2w_t_y, c2w_t_z)
                    const S vx = c2w_t_x - world_pos.x;
                    const S vy = c2w_t_y - world_pos.y;
                    const S vz = c2w_t_z - world_pos.z;
                    const S inv_len = rsqrtf(vx * vx + vy * vy + vz * vz + 1e-8f);
                    const vec3<S> view_dir = {vx * inv_len, vy * inv_len, vz * inv_len};

                    const int sample = sample_counts[pix_id] >> 1;

                    if (sample < num_texture_samples)
                    {
                        atomicAdd(sample_counts + pix_id, 2);
                        sample_gaussian_ids[pix_id] = g;
                        const int base = (pix_id * num_texture_samples + sample) * 6;
                        texture_inputs[base] = world_pos.x;
                        texture_inputs[base + 1] = world_pos.y;
                        texture_inputs[base + 2] = world_pos.z;
                        texture_inputs[base + 3] = view_dir.x;
                        texture_inputs[base + 4] = view_dir.y;
                        texture_inputs[base + 5] = view_dir.z;
                    }
                    else
                    {
                        done = true;
                        break;
                    }
                }
            }
        }
    }

    std::tuple<
        torch::Tensor,
        torch::Tensor,
        torch::Tensor>
    rasterize_to_samples_world_and_view_fwd_textured_gaussians_tensor(
        // Gaussian parameters
        const torch::Tensor &means2d,             // [C, N, 2] or [nnz, 2]
        const torch::Tensor &ray_transforms,      // [C, N, 3, 3] or [nnz, 3, 3]
        const torch::Tensor &viewmats,            // [C, 4, 4]
        const torch::Tensor &Ks,                  // [C, 3, 3]
        const torch::Tensor &opacities,           // [C, N] or [nnz]
        const at::optional<torch::Tensor> &masks, // [C, tile_height, tile_width]
        // image size
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        // intersections
        const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
        const torch::Tensor &flatten_ids,  // [n_isects]

        const uint32_t num_texture_samples,
        const float sample_alpha_threshold)
    {
        GSPLAT_DEVICE_GUARD(means2d);
        GSPLAT_CHECK_INPUT(means2d);
        GSPLAT_CHECK_INPUT(ray_transforms);
        GSPLAT_CHECK_INPUT(viewmats);
        GSPLAT_CHECK_INPUT(Ks);
        GSPLAT_CHECK_INPUT(opacities);
        GSPLAT_CHECK_INPUT(tile_offsets);
        GSPLAT_CHECK_INPUT(flatten_ids);
        if (masks.has_value())
        {
            GSPLAT_CHECK_INPUT(masks.value());
        }
        bool packed = means2d.dim() == 2;

        uint32_t C = tile_offsets.size(0);
        uint32_t N = packed ? 0 : means2d.size(1);
        uint32_t tile_height = tile_offsets.size(1);
        uint32_t tile_width = tile_offsets.size(2);
        uint32_t n_isects = flatten_ids.size(0);

        dim3 threads = {tile_size, tile_size, 1};
        dim3 blocks = {C, tile_height, tile_width};

        torch::Tensor sample_counts = torch::ones(
            {C, image_height, image_width},
            means2d.options().dtype(torch::kInt32));
        torch::Tensor sample_gaussian_ids = torch::full(
            {C, image_height, image_width}, -1,
            means2d.options().dtype(torch::kInt32));
        // 6 channels per sample: [x, y, z, vx, vy, vz]
        torch::Tensor texture_inputs = torch::zeros(
            {C, image_height, image_width, num_texture_samples, 6},
            means2d.options().dtype(torch::kFloat32));

        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        // shared memory unchanged: id_batch + xy_opacity_batch + u_Ms + v_Ms + w_Ms
        // (camera position and view_dir are computed in registers from per-camera data)
        const uint32_t shared_mem =
            tile_size * tile_size *
            (sizeof(int32_t) + sizeof(vec3<float>) + sizeof(vec3<float>) +
             sizeof(vec3<float>) + sizeof(vec3<float>));

        if (cudaFuncSetAttribute(
                rasterize_to_samples_world_and_view_fwd_implicit_textured_gaussians_kernel<float>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                shared_mem) != cudaSuccess)
        {
            AT_ERROR(
                "Failed to set maximum shared memory size (requested ",
                shared_mem,
                " bytes), try lowering tile_size.");
        }
        rasterize_to_samples_world_and_view_fwd_implicit_textured_gaussians_kernel<float>
            <<<blocks, threads, shared_mem, stream>>>(
                C,
                N,
                n_isects,
                packed,
                reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
                ray_transforms.data_ptr<float>(),
                viewmats.data_ptr<float>(),
                Ks.data_ptr<float>(),
                opacities.data_ptr<float>(),
                masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
                image_width,
                image_height,
                tile_size,
                tile_width,
                tile_height,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),

                num_texture_samples,
                sample_alpha_threshold,

                sample_counts.data_ptr<int32_t>(),
                sample_gaussian_ids.data_ptr<int32_t>(),
                texture_inputs.data_ptr<float>());

        return std::make_tuple(
            sample_counts,
            sample_gaussian_ids,
            texture_inputs);
    }
} // namespace gsplat
