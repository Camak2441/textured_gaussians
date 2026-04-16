
#ifndef GSPLAT_CUDA_MIP2_FILTER_H
#define GSPLAT_CUDA_MIP2_FILTER_H

#include "../helpers.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/TensorAccessor.h>

#define FILTER_INV_SQUARE 2.0f
#define ISQRT2 0.70710678118f

namespace gsplat::mip2
{ // Helper function for trilinear interpolation coordinate and weight calculation
    inline uint32_t t_base_index_host(
        uint32_t log_texture_res, uint32_t t)
    {
        return ((1 << ((log_texture_res - t) * 2)) - 1) / 3;
    }

    inline __device__ uint32_t t_base_index(
        int log_texture_res, int32_t t)
    {
        return ((1 << ((log_texture_res - t) * 2)) - 1) / 3;
    }

    // Helper function for trilinear interpolation coordinate and weight calculation
    template <typename T>
    inline __device__ int32_t precompute(
        T s_x, T s_y, vec2<T> dsdx, vec2<T> dsdy, int texture_res, int log_texture_res,
        int32_t (&mipcoords)[8], T (&trilerp_weights)[8])
    {
        // Map s_x, s_y in [-3, 3] to texture coordinates

        vec2<T> duvdx = vec2<T>(dsdx.x / T(6) * (texture_res - 1), dsdx.y / T(6) * (texture_res - 1));
        vec2<T> duvdy = vec2<T>(dsdy.x / T(6) * (texture_res - 1), dsdy.y / T(6) * (texture_res - 1));
        T duv = max(glm::length(duvdx), glm::length(duvdy));
        T t = log2(duv);
        if (t < 0)
        {
            t = 0;
        }
        else if (t > 31)
        {
            t = 31;
        }

        // For ease of mipmapping, coordinate textures give the tl of the pixel
        T u = (s_x + T(3)) / T(6) * (texture_res - 1);
        T v = (s_y + T(3)) / T(6) * (texture_res - 1);
        int32_t t_low = (int32_t)floor(t);
        int32_t t_high = (int32_t)ceil(t);

        uint32_t scale_small = 1 << t_low;
        uint32_t scale_large = 1 << t_high;

        T u_small = (u + T(0.5)) / scale_small - T(0.5);
        T u_large = (u + T(0.5)) / scale_large - T(0.5);
        T v_small = (v + T(0.5)) / scale_small - T(0.5);
        T v_large = (v + T(0.5)) / scale_large - T(0.5);

        int32_t u_low_small = (int32_t)floor(u_small);
        int32_t u_low_large = (int32_t)floor(u_large);
        int32_t v_low_small = (int32_t)floor(v_small);
        int32_t v_low_large = (int32_t)floor(v_large);
        int32_t u_high_small = (int32_t)ceil(u_small);
        int32_t u_high_large = (int32_t)ceil(u_large);
        int32_t v_high_small = (int32_t)ceil(v_small);
        int32_t v_high_large = (int32_t)ceil(v_large);

        if (u_low_small < 0 || u_low_large < 0 || v_low_small < 0 || v_low_large < 0 ||
            u_high_small > (texture_res - 1) >> t_low || u_high_large > (texture_res - 1) >> t_high ||
            v_high_small > (texture_res - 1) >> t_low || v_high_large > (texture_res - 1) >> t_high)
        {
            return -1;
        }

        uint32_t t_low_res = texture_res >> t_low;
        uint32_t t_high_res = texture_res >> t_high;

        uint32_t t_low_base = t_base_index(log_texture_res, t_low);
        uint32_t t_high_base = t_base_index(log_texture_res, t_high);

        mipcoords[0] = t_low_base + v_low_small * t_low_res + u_low_small;
        mipcoords[1] = t_low_base + v_low_small * t_low_res + u_high_small;
        mipcoords[2] = t_low_base + v_high_small * t_low_res + u_low_small;
        mipcoords[3] = t_low_base + v_high_small * t_low_res + u_high_small;
        mipcoords[4] = t_high_base + v_low_large * t_high_res + u_low_large;
        mipcoords[5] = t_high_base + v_low_large * t_high_res + u_high_large;
        mipcoords[6] = t_high_base + v_high_large * t_high_res + u_low_large;
        mipcoords[7] = t_high_base + v_high_large * t_high_res + u_high_large;
        T w_u_small = u_small - (T)u_low_small;
        T w_v_small = v_small - (T)v_low_small;
        T w_u_large = u_large - (T)u_low_large;
        T w_v_large = v_large - (T)v_low_large;
        T w_t = t - (T)t_low;
        trilerp_weights[0] = (T(1) - w_u_small) * (T(1) - w_v_small) * (T(1) - w_t);
        trilerp_weights[1] = w_u_small * (T(1) - w_v_small) * (T(1) - w_t);
        trilerp_weights[2] = (T(1) - w_u_small) * w_v_small * (T(1) - w_t);
        trilerp_weights[3] = w_u_small * w_v_small * (T(1) - w_t);
        trilerp_weights[4] = (T(1) - w_u_large) * (T(1) - w_v_large) * w_t;
        trilerp_weights[5] = w_u_large * (T(1) - w_v_large) * w_t;
        trilerp_weights[6] = (T(1) - w_u_large) * w_v_large * w_t;
        trilerp_weights[7] = w_u_large * w_v_large * w_t;
        return 1;
    }
} // namespace gsplat::mip2

#endif