#ifndef GSPLAT_CUDA_DCT_FILTER_H
#define GSPLAT_CUDA_DCT_FILTER_H

#include "../helpers.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/TensorAccessor.h>

#define FILTER_INV_SQUARE 2.0f
#define ISQRT2 0.70710678118f

namespace gsplat
{
    // Fast cosine approximation
    // Only works for x > - pi/2
    // Please do not use for anything else
    template <typename T>
    inline __device__ T dct_cos(T x)
    {
        T k = fmodf(x + M_PI_2, M_PI) - M_PI_2;
        k = k * k;
        T m = 1 - k / 2 + k * k * 0.921279629f / 24;
        return copysignf(m, fmodf(x + 3 * M_PI_2, 2 * M_PI) - M_PI);
    }

    template <typename T>
    inline __device__ void precompute_dct_factors(
        int texture_res_x,
        int texture_res_y,
        T u, // from 0 to 1
        T v,
        T *ucos,
        T *vcos)
    {
        ucos[0] = 1.f;
        vcos[0] = 1.f;
        T rsqrti;
        T pii = 0;
        for (int i = 1; i < texture_res_x; ++i)
        {
            rsqrti = rsqrtf((float)(i + 1));
            pii += M_PI;
            ucos[i] = dct_cos(pii * u) * rsqrti;
            vcos[i] = dct_cos(pii * v) * rsqrti;
        }
    }

    // Helper function for trilinear interpolation coordinate and weight calculation
    template <typename T>
    inline __device__ T dct_sample(
        at::PackedTensorAccessor32<const T, 4, at::RestrictPtrTraits> textures, // [N, Texture_Resolution, Texture_Resolution, 4]
        int texture_res_x,
        int texture_res_y,
        uint32_t g,
        T u, // from 0 to 1
        T v, // from 0 to 1
        T *ucos,
        T *vcos,
        uint32_t k)
    {
        T col = 0;
        T vj;
        for (int j = 0; j < texture_res_y; ++j)
        {
            vj = vcos[j];
            for (int i = 0; i < texture_res_x; ++i)
            {
                col += textures[g][j][i][k] * ucos[i] * vj;
            }
        }
        return col;
    }

    // Helper function for trilinear interpolation coordinate and weight calculation
    template <uint32_t COLOR_DIM, typename T>
    inline __device__ void dct_color_sample(
        at::PackedTensorAccessor32<const T, 4, at::RestrictPtrTraits> textures, // [N, Texture_Resolution, Texture_Resolution, 4]
        int texture_res_x,
        int texture_res_y,
        uint32_t g,
        T u, // from 0 to 1
        T v, // from 0 to 1
        T *ucos,
        T *vcos,
        T col[COLOR_DIM])
    {
        T vj;
        T uivj;
        for (int j = 0; j < texture_res_y; ++j)
        {
            vj = vcos[j];
            for (int i = 0; i < texture_res_x; ++i)
            {
                uivj = ucos[i] * vj;
                GSPLAT_PRAGMA_UNROLL
                for (int k = 0; k < COLOR_DIM; ++k)
                {
                    col[k] += textures[g][j][i][k] * uivj;
                }
            }
        }
    }

    template <typename T>
    inline __device__ void dct_update(
        at::PackedTensorAccessor32<T, 4, at::RestrictPtrTraits> v_textures, // [C, N, TEXTURE_DIM] or [nnz, TEXTURE_DIM]
        int texture_res_x,
        int texture_res_y,
        uint32_t g,
        T u, // u from 0 to 1
        T v, // v from 0 to 1
        T *ucos,
        T *vcos,
        uint32_t k,
        T delta)
    {
        T vj;
        for (int j = 0; j < texture_res_y; ++j)
        {
            vj = vcos[j];
            for (int i = 0; i < texture_res_x; ++i)
            {
                gpuAtomicAdd(&v_textures[g][j][i][k], delta * ucos[i] * vj);
            }
        }
        return;
    }

    template <uint32_t COLOR_DIM, typename T>
    inline __device__ void dct_color_sample_and_update(
        at::PackedTensorAccessor32<const T, 4, at::RestrictPtrTraits> textures, // [C, N, TEXTURE_DIM] or [nnz, TEXTURE_DIM]
        at::PackedTensorAccessor32<T, 4, at::RestrictPtrTraits> v_textures,     // [C, N, TEXTURE_DIM] or [nnz, TEXTURE_DIM]
        int texture_res_x,
        int texture_res_y,
        uint32_t g,
        T u, // u from 0 to 1
        T v, // v from 0 to 1
        T *ucos,
        T *vcos,
        T col[COLOR_DIM],
        T deltas[COLOR_DIM])
    {
        T vj;
        T uivj;
        for (int j = 0; j < texture_res_y; ++j)
        {
            vj = vcos[j];
            for (int i = 0; i < texture_res_x; ++i)
            {
                uivj = ucos[i] * vj;
                GSPLAT_PRAGMA_UNROLL
                for (int k = 0; k < COLOR_DIM; ++k)
                {
                    gpuAtomicAdd(&v_textures[g][j][i][k], deltas[k] * uivj);
                    col[k] += textures[g][j][i][k] * uivj;
                }
            }
        }
        return;
    }
}

#endif // GSPLAT_CUDA_DCT_FILTER_H