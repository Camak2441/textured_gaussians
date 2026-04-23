#ifndef GSPLAT_CUDA_BILINEAR_S_FILTER_H
#define GSPLAT_CUDA_BILINEAR_S_FILTER_H

#include "../helpers.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/TensorAccessor.h>

#define FILTER_INV_SQUARE 2.0f

namespace gsplat::bilinear_s
{
    // Helper function for bilinear interpolation coordinate and weight calculation
    template <typename T>
    inline __device__ int32_t precompute(
        T s_x, T s_y, int texture_res_x, int texture_res_y,
        int32_t (&ucoords)[4], int32_t (&vcoords)[4], T (&bilerp_weights)[4])
    {
        if (s_x < T(-1.2) || s_x > T(1.2) || s_y < T(-1.2) || s_y > T(1.2))
        {
            return -1;
        }
        // Map s_x, s_y in [-3, 3] to texture coordinates
        T u = (s_x + T(1.2)) / T(2.4) * (texture_res_x - 1);
        T v = (s_y + T(1.2)) / T(2.4) * (texture_res_y - 1);
        // clamp the uv coordinates to valid range
        if (u < 0)
        {
            u = 0;
        }
        else if (u > texture_res_x - 1)
        {
            u = (T)(texture_res_x - 1);
        }
        if (v < 0)
        {
            v = 0;
        }
        else if (v > texture_res_y - 1)
        {
            v = (T)(texture_res_y - 1);
        }
        int32_t u_low = (int32_t)floor(u);
        int32_t v_low = (int32_t)floor(v);
        int32_t u_high = (int32_t)ceil(u);
        int32_t v_high = (int32_t)ceil(v);

        ucoords[0] = u_low;
        ucoords[1] = u_high;
        ucoords[2] = u_low;
        ucoords[3] = u_high;
        vcoords[0] = v_low;
        vcoords[1] = v_low;
        vcoords[2] = v_high;
        vcoords[3] = v_high;
        T w_u = u - (T)u_low;
        T w_v = v - (T)v_low;
        bilerp_weights[0] = (T(1) - w_u) * (T(1) - w_v);
        bilerp_weights[1] = w_u * (T(1) - w_v);
        bilerp_weights[2] = (T(1) - w_u) * w_v;
        bilerp_weights[3] = w_u * w_v;
        return 1;
    }
} // namespace gsplat::bilinear_s

#endif // GSPLAT_CUDA_BILINEAR_S_FILTER_H