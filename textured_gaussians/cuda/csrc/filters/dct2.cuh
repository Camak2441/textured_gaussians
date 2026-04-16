#ifndef GSPLAT_CUDA_DCT2_FILTER_H
#define GSPLAT_CUDA_DCT2_FILTER_H

#include "../helpers.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/TensorAccessor.h>

#define FILTER_INV_SQUARE 2.0f
#define ISQRT2 0.70710678118f

namespace gsplat::dct2
{
    // Fast cosine approximation
    // Has at most ~3% error
    template <typename T>
    inline __device__ constexpr T dct_cos(T x)
    {
        x = abs(x);
        T k = fmod(x + T(M_PI_2), T(M_PI)) - T(M_PI_2);
        k = k * k;
        T m = T(1) - k / T(2) + k * k * T(0.921279629) / T(24);
        return copysign(m, fmod(x + T(3) * T(M_PI_2), T(2) * T(M_PI)) - T(M_PI));
    }

    template <typename T>
    inline __device__ void precompute(
        int texture_res_x,
        int texture_res_y,
        T u, // from 0 to 1
        T v,
        T *ucos,
        T *vcos)
    {
        ucos[0] = T(ISQRT2);
        vcos[0] = T(ISQRT2);
        T pii = 0;
        for (int i = 1; i < texture_res_x; ++i)
        {
            pii += T(M_PI);
            ucos[i] = dct_cos(pii * u) * T(ISQRT2);
            vcos[i] = dct_cos(pii * v) * T(ISQRT2);
        }
    }

    // Helper function for trilinear interpolation coordinate and weight calculation
    template <typename T>
    inline __device__ T sample(
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
    inline __device__ void color_sample(
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
                    col[k] += abs(tex * (uivj + copysign(T(0.5), tex))) * freq_norm;
                }
            }
        }
    }

    template <typename T>
    inline __device__ void update(
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
        T norm = T(1) / (texture_res_x * texture_res_y);
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
    inline __device__ void color_sample_and_update(
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
        T norm = T(1) / (texture_res_x * texture_res_y);
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
                    gpuAtomicAdd(&v_textures[g][j][i][k], deltas[k] * (uivj + copysign(T(0.5), tex)) * freq_norm);
                    col[k] += abs(tex * (uivj + copysign(T(0.5), tex))) * freq_norm;
                }
            }
        }
        return;
    }
} // namespace gsplat::dct2

#endif // GSPLAT_CUDA_DCT2_FILTER_H