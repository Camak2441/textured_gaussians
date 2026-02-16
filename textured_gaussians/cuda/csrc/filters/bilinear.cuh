#ifndef GSPLAT_CUDA_BILINEAR_FILTER_H
#define GSPLAT_CUDA_BILINEAR_FILTER_H

#include "helpers.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/TensorAccessor.h>

#define FILTER_INV_SQUARE 2.0f

namespace gsplat
{
    // Helper function for bilinear interpolation coordinate and weight calculation
    template <typename T>
    inline __device__ int32_t compute_bilinear_coords_weights(
        T s_x, T s_y, int texture_res_x, int texture_res_y,
        int32_t (&ucoords)[4], int32_t (&vcoords)[4], T (&bilerp_weights)[4])
    {
        if (s_x < -3.0f || s_x > 3.0f || s_y < -3.0f || s_y > 3.0f)
        {
            return -1;
        }
        // Map s_x, s_y in [-3, 3] to texture coordinates
        T u = (T)((s_x + 3.0f) / 6.0f * (texture_res_x - 1));
        T v = (T)((s_y + 3.0f) / 6.0f * (texture_res_y - 1));
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
        bilerp_weights[0] = (1.0f - w_u) * (1.0f - w_v);
        bilerp_weights[1] = w_u * (1.0f - w_v);
        bilerp_weights[2] = (1.0f - w_u) * w_v;
        bilerp_weights[3] = w_u * w_v;
        return 1;
    }
}

#endif // GSPLAT_CUDA_BILINEAR_FILTER_H