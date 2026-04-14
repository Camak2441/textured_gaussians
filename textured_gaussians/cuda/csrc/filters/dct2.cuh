#ifndef GSPLAT_CUDA_DCT2_FILTER_H
#define GSPLAT_CUDA_DCT2_FILTER_H

#include "../helpers.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/TensorAccessor.h>

#define FILTER_INV_SQUARE 2.0f
#define ISQRT2 0.70710678118f

namespace gsplat
{
    // Fast cosine approximation
    // Has at most ~3% error
    template <typename T>
    inline __device__ constexpr T dct2_cos(T x)
    {
        x = abs(x);
        T k = fmodf(x + M_PI_2, M_PI) - M_PI_2;
        k = k * k;
        T m = 1.f - k / 2.f + k * k * 0.921279629f / 24.f;
        return copysignf(m, fmodf(x + 3 * M_PI_2, 2 * M_PI) - M_PI);
    }

    template <typename T>
    inline __device__ void precompute_dct2_factors(
        int texture_res_x,
        int texture_res_y,
        T u, // from 0 to 1
        T v,
        T *ucos,
        T *vcos)
    {
        ucos[0] = ISQRT2;
        vcos[0] = ISQRT2;
        T pii = 0;
        for (int i = 1; i < texture_res_x; ++i)
        {
            pii += M_PI;
            ucos[i] = dct2_cos(pii * u) * ISQRT2;
            vcos[i] = dct2_cos(pii * v) * ISQRT2;
        }
    }

    // Helper function for trilinear interpolation coordinate and weight calculation
    template <typename T>
    inline __device__ T dct2_sample(
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
        T norm = 1.f / (texture_res_x * texture_res_y);
        for (int j = 0; j < texture_res_y; ++j)
        {
            vj = vcos[j];
            for (int i = 0; i < texture_res_x; ++i)
            {
                T tex = textures[g][j][i][k];
                col += abs(tex * (ucos[i] * vj + copysignf(0.5f, tex))) * norm * (i + 1) * (j + 1);
            }
        }
        return col;
    }

    // Helper function for trilinear interpolation coordinate and weight calculation
    template <uint32_t COLOR_DIM, typename T>
    inline __device__ void dct2_color_sample(
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
        T norm = 1.f / (texture_res_x * texture_res_y);
        T freq_norm;
        for (int j = 0; j < texture_res_y; ++j)
        {
            vj = vcos[j];
            for (int i = 0; i < texture_res_x; ++i)
            {
                uivj = ucos[i] * vj;
                freq_norm = norm * (i + 1) * (j + 1);
                GSPLAT_PRAGMA_UNROLL
                for (int k = 0; k < COLOR_DIM; ++k)
                {
                    T tex = textures[g][j][i][k];
                    col[k] += abs(tex * (uivj + copysignf(0.5f, tex))) * freq_norm;
                }
            }
        }
    }

    template <typename T>
    inline __device__ void dct2_update(
        at::PackedTensorAccessor32<const T, 4, at::RestrictPtrTraits> textures, // [C, N, TEXTURE_DIM] or [nnz, TEXTURE_DIM]
        at::PackedTensorAccessor32<T, 4, at::RestrictPtrTraits> v_textures,     // [C, N, TEXTURE_DIM] or [nnz, TEXTURE_DIM]
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
        T norm = 1.f / (texture_res_x * texture_res_y);
        for (int j = 0; j < texture_res_y; ++j)
        {
            vj = vcos[j];
            for (int i = 0; i < texture_res_x; ++i)
            {
                T tex = textures[g][j][i][k];
                gpuAtomicAdd(&v_textures[g][j][i][k], delta * (ucos[i] * vj + copysignf(0.5f, tex)) * norm * (i + 1) * (j + 1));
            }
        }
        return;
    }

    template <uint32_t COLOR_DIM, typename T>
    inline __device__ void dct2_color_sample_and_update(
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
        T norm = 1.f / (texture_res_x * texture_res_y);
        T freq_norm;
        for (int j = 0; j < texture_res_y; ++j)
        {
            vj = vcos[j];
            for (int i = 0; i < texture_res_x; ++i)
            {
                uivj = ucos[i] * vj;
                freq_norm = norm * (i + 1) * (j + 1);
                GSPLAT_PRAGMA_UNROLL
                for (int k = 0; k < COLOR_DIM; ++k)
                {
                    T tex = textures[g][j][i][k];
                    gpuAtomicAdd(&v_textures[g][j][i][k], deltas[k] * (uivj + copysignf(0.5f, tex)) * freq_norm);
                    col[k] += abs(tex * (uivj + copysignf(0.5f, tex))) * freq_norm;
                }
            }
        }
        return;
    }
}

#endif // GSPLAT_CUDA_DCT2_FILTER_H