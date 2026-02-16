#ifndef GSPLAT_CUDA_MIP_FILTER_H
#define GSPLAT_CUDA_MIP_FILTER_H

#include "helpers.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/TensorAccessor.h>

#define FILTER_INV_SQUARE 2.0f
#define ISQRT2 0.70710678118f

namespace gsplat
{
    // Helper function for trilinear interpolation coordinate and weight calculation
    template <typename T>
    inline __device__ int32_t compute_trilinear_coords_weights(
        T s_x, T s_y, vec2<T> dsdx, vec2<T> dsdy, int texture_res_x, int texture_res_y,
        int32_t (&ucoords)[8], int32_t (&vcoords)[8], int32_t (&tcoords)[8], T (&trilerp_weights)[8])
    {
        // Map s_x, s_y in [-3, 3] to texture coordinates

        vec2<T> duvdx = vec2<T>((T)(dsdx.x / 6.0f * (texture_res_x - 1) / 2.0f), (T)(dsdx.y / 6.0f * (texture_res_y - 1) / 2.0f));
        vec2<T> duvdy = vec2<T>((T)(dsdy.x / 6.0f * (texture_res_x - 1) / 2.0f), (T)(dsdy.y / 6.0f * (texture_res_y - 1) / 2.0f));
        T duv = (T)max(glm::length(duvdx), glm::length(duvdy));
        T t = (T)log2(duv);
        if (t < 0)
        {
            t = 0;
        }
        else if (t > 31)
        {
            t = 31;
        }

        // For ease of mipmapping, coordinate textures give the tl of the pixel
        T u = (T)((s_x + 3.0f) / 6.0f * (texture_res_x - 1) / 2.0f);
        T v = (T)((s_y + 3.0f) / 6.0f * (texture_res_y - 1) / 2.0f);
        int32_t t_low = (int32_t)floor(t);
        int32_t t_high = (int32_t)ceil(t);

        int32_t scale_small = 1 << t_low;
        int32_t scale_large = 1 << t_high;

        T u_small = (u + 0.5) / scale_small - 0.5;
        T u_large = (u + 0.5) / scale_large - 0.5;
        T v_small = (v + 0.5) / scale_small - 0.5;
        T v_large = (v + 0.5) / scale_large - 0.5;

        int32_t u_low_small = (int32_t)floor(u_small);
        int32_t u_low_large = (int32_t)floor(u_large);
        int32_t v_low_small = (int32_t)floor(v_small);
        int32_t v_low_large = (int32_t)floor(v_large);
        int32_t u_high_small = (int32_t)ceil(u_small);
        int32_t u_high_large = (int32_t)ceil(u_large);
        int32_t v_high_small = (int32_t)ceil(v_small);
        int32_t v_high_large = (int32_t)ceil(v_large);

        if (u_low_small < 0 || u_low_large < 0 || v_low_small < 0 || v_low_large < 0 ||
            u_high_small > (texture_res_x - 1) / scale_small || u_high_large > (texture_res_x - 1) / scale_large ||
            v_high_small > (texture_res_y - 1) / scale_small || v_high_large > (texture_res_y - 1) / scale_large)
        {
            return -1;
        }

        ucoords[0] = u_low_small;
        ucoords[1] = u_high_small;
        ucoords[2] = u_low_small;
        ucoords[3] = u_high_small;
        ucoords[4] = u_low_large;
        ucoords[5] = u_high_large;
        ucoords[6] = u_low_large;
        ucoords[7] = u_high_large;
        vcoords[0] = v_low_small;
        vcoords[1] = v_low_small;
        vcoords[2] = v_high_small;
        vcoords[3] = v_high_small;
        vcoords[4] = v_low_large;
        vcoords[5] = v_low_large;
        vcoords[6] = v_high_large;
        vcoords[7] = v_high_large;
        tcoords[0] = t_low;
        tcoords[1] = t_low;
        tcoords[2] = t_low;
        tcoords[3] = t_low;
        tcoords[4] = t_high;
        tcoords[5] = t_high;
        tcoords[6] = t_high;
        tcoords[7] = t_high;
        T w_u_small = u_small - (T)u_low_small;
        T w_v_small = v_small - (T)v_low_small;
        T w_u_large = u_large - (T)u_low_large;
        T w_v_large = v_large - (T)v_low_large;
        T w_t = t - (T)t_low;
        trilerp_weights[0] = (1.0f - w_u_small) * (1.0f - w_v_small) * (1.0f - w_t);
        trilerp_weights[1] = w_u_small * (1.0f - w_v_small) * (1.0f - w_t);
        trilerp_weights[2] = (1.0f - w_u_small) * w_v_small * (1.0f - w_t);
        trilerp_weights[3] = w_u_small * w_v_small * (1.0f - w_t);
        trilerp_weights[4] = (1.0f - w_u_large) * (1.0f - w_v_large) * w_t;
        trilerp_weights[5] = w_u_large * (1.0f - w_v_large) * w_t;
        trilerp_weights[6] = (1.0f - w_u_large) * w_v_large * w_t;
        trilerp_weights[7] = w_u_large * w_v_large * w_t;
        return 1;
    }

    template <typename T>
    inline __device__ T mip_sample(
        at::PackedTensorAccessor32<const T, 4, at::RestrictPtrTraits> textures, // [N, Texture_Resolution, Texture_Resolution, 4]
        int32_t g,
        int32_t k,
        int32_t ucoord,
        int32_t vcoord,
        int32_t tcoord)
    {
        T c = 0;

        int32_t scale = 1 << tcoord;

        for (int32_t u = ucoord * scale; u < (ucoord + 1) * scale; u++)
        {
            for (int32_t v = vcoord * scale; v < (vcoord + 1) * scale; v++)
            {
                c += textures[g][u][v][k];
            }
        }

        return c / scale / scale;
    }

    template <typename T>
    inline __device__ void mip_update(
        at::PackedTensorAccessor32<T, 4, at::RestrictPtrTraits> v_textures, // [C, N, TEXTURE_DIM] or [nnz, TEXTURE_DIM]
        int32_t g,
        int32_t k,
        int32_t ucoord,
        int32_t vcoord,
        int32_t tcoord,
        T delta)
    {
        T c = 0;

        int32_t scale = 1 << tcoord;

        for (int32_t u = ucoord * scale; u < (ucoord + 1) * scale; u++)
        {
            for (int32_t v = vcoord * scale; v < (vcoord + 1) * scale; v++)
            {
                gpuAtomicAdd(&v_textures[g][u][v][k], delta / scale / scale);
            }
        }
    }
}

#endif // GSPLAT_CUDA_MIP_FILTER_H