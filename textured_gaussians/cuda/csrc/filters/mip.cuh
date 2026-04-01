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

        vec2<T> duvdx = vec2<T>((T)(dsdx.x / 6.0f * (texture_res_x - 1)), (T)(dsdx.y / 6.0f * (texture_res_y - 1)));
        vec2<T> duvdy = vec2<T>((T)(dsdy.x / 6.0f * (texture_res_x - 1)), (T)(dsdy.y / 6.0f * (texture_res_y - 1)));
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
        T u = (T)((s_x + 3.0f) / 6.0f * (texture_res_x - 1));
        T v = (T)((s_y + 3.0f) / 6.0f * (texture_res_y - 1));
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
            u_high_small > (texture_res_x - 1) >> t_low || u_high_large > (texture_res_x - 1) >> t_high ||
            v_high_small > (texture_res_y - 1) >> t_low || v_high_large > (texture_res_y - 1) >> t_high)
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
        if (tcoord == 0)
        {
            return textures[g][vcoord][ucoord][k];
        }

        T c = 0;

        int32_t scale = 1 << tcoord;

        for (int32_t u = ucoord * scale; u < (ucoord + 1) * scale; u++)
        {
            for (int32_t v = vcoord * scale; v < (vcoord + 1) * scale; v++)
            {
                c += textures[g][v][u][k];
            }
        }

        return c * (1.0f / (scale * scale));
    }

    template <typename T>
    inline __device__ T mip_efficient_sample(
        at::PackedTensorAccessor32<const T, 4, at::RestrictPtrTraits> textures, // [N, Texture_Resolution, Texture_Resolution, 4]
        int32_t g,
        int32_t k,
        int32_t (&ucoords)[8],
        int32_t (&vcoords)[8],
        int32_t (&tcoords)[8],
        T (&trilerp_weights)[8])
    {
        if (tcoords[0] == tcoords[4])
        {
            if (ucoords[0] == ucoords[1] && vcoords[0] == vcoords[2])
            {
                return mip_sample(textures, g, k, ucoords[0], vcoords[0], tcoords[0]);
            }
            else if (ucoords[0] == ucoords[1])
            {
                T sample0 = mip_sample(textures, g, k, ucoords[0], vcoords[0], tcoords[0]);
                T sample1 = mip_sample(textures, g, k, ucoords[2], vcoords[2], tcoords[2]);
                T weight = trilerp_weights[0] + trilerp_weights[1];
                return weight * sample0 + (1.0 - weight) * sample1;
            }
            else if (vcoords[0] == vcoords[2])
            {
                T sample0 = mip_sample(textures, g, k, ucoords[0], vcoords[0], tcoords[0]);
                T sample1 = mip_sample(textures, g, k, ucoords[1], vcoords[1], tcoords[1]);
                T weight = trilerp_weights[0] + trilerp_weights[2];
                return weight * sample0 + (1.0 - weight) * sample1;
            }
            else
            {
                T sample0 = mip_sample(textures, g, k, ucoords[0], vcoords[0], tcoords[0]);
                T sample1 = mip_sample(textures, g, k, ucoords[1], vcoords[1], tcoords[1]);
                T sample2 = mip_sample(textures, g, k, ucoords[2], vcoords[2], tcoords[2]);
                T sample3 = mip_sample(textures, g, k, ucoords[3], vcoords[3], tcoords[3]);
                return trilerp_weights[0] * sample0 + trilerp_weights[1] * sample1 + trilerp_weights[2] * sample2 + trilerp_weights[3] * sample3;
            }
        }

        T samples[4][4];

        int minu = ucoords[4] * 2;
        int minv = vcoords[4] * 2;

        for (int u = ucoords[4] * 2; u < ucoords[5] * 2 + 2; ++u)
        {
            for (int v = vcoords[4] * 2; v < vcoords[6] * 2 + 2; ++v)
            {
                samples[v - minv][u - minu] = mip_sample(textures, g, k, u, v, tcoords[0]);
            }
        }

        T c = 0;
        int i = 0;

        for (; i < 4; i++)
        {
            c += trilerp_weights[i] * samples[vcoords[i] - minv][ucoords[i] - minu];
        }
        for (; i < 8; i++)
        {
            c += trilerp_weights[i] * samples[vcoords[i] * 2 - minv][ucoords[i] * 2 - minu] / 4;
            c += trilerp_weights[i] * samples[vcoords[i] * 2 + 1 - minv][ucoords[i] * 2 - minu] / 4;
            c += trilerp_weights[i] * samples[vcoords[i] * 2 - minv][ucoords[i] * 2 + 1 - minu] / 4;
            c += trilerp_weights[i] * samples[vcoords[i] * 2 + 1 - minv][ucoords[i] * 2 + 1 - minu] / 4;
        }
        return c;
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
        if (tcoord == 0)
        {
            gpuAtomicAdd(&v_textures[g][vcoord][ucoord][k], delta);
            return;
        }

        int32_t scale = 1 << tcoord;

        for (int32_t u = ucoord * scale; u < (ucoord + 1) * scale; u++)
        {
            for (int32_t v = vcoord * scale; v < (vcoord + 1) * scale; v++)
            {
                gpuAtomicAdd(&v_textures[g][v][u][k], delta * (1.0f / (scale * scale)));
            }
        }
    }

    template <typename T>
    inline __device__ void mip_efficient_update(
        at::PackedTensorAccessor32<T, 4, at::RestrictPtrTraits> v_textures, // [C, N, TEXTURE_DIM] or [nnz, TEXTURE_DIM]
        int32_t g,
        int32_t k,
        int32_t (&ucoords)[8],
        int32_t (&vcoords)[8],
        int32_t (&tcoords)[8],
        T (&trilerp_weights)[8],
        T delta)
    {
        if (tcoords[0] == tcoords[4])
        {
            if (ucoords[0] == ucoords[1] && vcoords[0] == vcoords[2])
            {
                mip_update(v_textures, g, k, ucoords[0], vcoords[0], tcoords[0], delta);
                return;
            }
            else if (ucoords[0] == ucoords[1])
            {
                T weight = trilerp_weights[0] + trilerp_weights[1];
                mip_update(v_textures, g, k, ucoords[0], vcoords[0], tcoords[0], weight * delta);
                mip_update(v_textures, g, k, ucoords[2], vcoords[2], tcoords[2], (((T)1.0) - weight) * delta);
                return;
            }
            else if (vcoords[0] == vcoords[2])
            {
                T weight = trilerp_weights[0] + trilerp_weights[2];
                mip_update(v_textures, g, k, ucoords[0], vcoords[0], tcoords[0], weight * delta);
                mip_update(v_textures, g, k, ucoords[1], vcoords[1], tcoords[1], (((T)1.0) - weight) * delta);
                return;
            }
            else
            {
                mip_update(v_textures, g, k, ucoords[0], vcoords[0], tcoords[0], trilerp_weights[0] * delta);
                mip_update(v_textures, g, k, ucoords[1], vcoords[1], tcoords[1], trilerp_weights[1] * delta);
                mip_update(v_textures, g, k, ucoords[2], vcoords[2], tcoords[2], trilerp_weights[2] * delta);
                mip_update(v_textures, g, k, ucoords[3], vcoords[3], tcoords[3], trilerp_weights[3] * delta);
                return;
            }
        }

        T deltas[4][4] = {0.0};

        int minu = ucoords[4] * 2;
        int minv = vcoords[4] * 2;

        int i = 0;

        for (; i < 4; i++)
        {
            deltas[vcoords[i] - minv][ucoords[i] - minu] += trilerp_weights[i] * delta;
        }
        for (; i < 8; i++)
        {
            deltas[vcoords[i] * 2 - minv][ucoords[i] * 2 - minu] += trilerp_weights[i] * delta / 4;
            deltas[vcoords[i] * 2 + 1 - minv][ucoords[i] * 2 - minu] += trilerp_weights[i] * delta / 4;
            deltas[vcoords[i] * 2 - minv][ucoords[i] * 2 + 1 - minu] += trilerp_weights[i] * delta / 4;
            deltas[vcoords[i] * 2 + 1 - minv][ucoords[i] * 2 + 1 - minu] += trilerp_weights[i] * delta / 4;
        }

        for (int u = ucoords[4] * 2; u < ucoords[5] * 2 + 2; ++u)
        {
            for (int v = vcoords[4] * 2; v < vcoords[6] * 2 + 2; ++v)
            {
                mip_update(v_textures, g, k, u, v, tcoords[0], deltas[v - minv][u - minu]);
            }
        }

        return;
    }
}

#endif // GSPLAT_CUDA_MIP_FILTER_H