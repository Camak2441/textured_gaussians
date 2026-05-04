#ifndef GSPLAT_CUDA_DCT3_FILTER_H
#define GSPLAT_CUDA_DCT3_FILTER_H

#include "../helpers.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/TensorAccessor.h>

#define FILTER_INV_SQUARE 2.0f
#define ISQRT2 0.70710678118f

namespace gsplat::dct3
{
    // Fast cosine approximation
    // Only works for x > - pi/2
    template <typename T>
    inline __device__ constexpr T dct_cos(T x)
    {
        T k = fmod(x + T(M_PI_2), T(M_PI)) - T(M_PI_2);
        k = k * k;
        T m = T(1) - k / T(2) + k * k * T(0.921279629) / T(24);
        return copysign(m, fmod(x + T(3) * T(M_PI_2), T(2) * T(M_PI)) - T(M_PI));
    }

    template <typename T>
    inline __device__ void precompute(
        int texture_res,
        T u, // from 0 to 1
        T v,
        T *ucos,
        T *vcos)
    {
        ucos[0] = T(1);
        vcos[0] = T(1);
        T rsqrti;
        T pii = 0;
        for (int i = 1; i < texture_res; ++i)
        {
            rsqrti = rsqrt((T)(i + 1));
            pii += M_PI;
            ucos[i] = dct_cos(pii * u) * rsqrti;
            vcos[i] = dct_cos(pii * v) * rsqrti;
        }
    }

    template <typename T>
    inline __device__ void grad_precompute(
        int texture_res,
        T u, // from 0 to 1
        T v,
        T *ucos,
        T *vcos,
        T *ducos,
        T *dvcos)
    {
        ucos[0] = T(1);
        vcos[0] = T(1);
        T rsqrti;
        T pii = 0;
        for (int i = 1; i < texture_res; ++i)
        {
            rsqrti = rsqrt((T)(i + 1));
            pii += M_PI;
            ucos[i] = dct_cos(pii * u) * rsqrti;
            vcos[i] = dct_cos(pii * v) * rsqrti;
            ducos[i] = pii * dct_cos(pii * u + 3 * M_PI_2) * rsqrti;
            dvcos[i] = pii * dct_cos(pii * v + 3 * M_PI_2) * rsqrti;
        }
    }

    // Helper function for trilinear interpolation coordinate and weight calculation
    template <typename T>
    inline __device__ T sample(
        at::PackedTensorAccessor32<const T, 3, at::RestrictPtrTraits> textures, // [N, t * (t - 1) / 2, 4]
        int texture_res,
        uint32_t g,
        T u, // from 0 to 1
        T v, // from 0 to 1
        T *ucos,
        T *vcos,
        uint32_t k)
    {
        T col = 0;
        T vj;
        int index = 0;
        for (int j = 0; j < texture_res; ++j)
        {
            vj = vcos[j];
            for (int i = 0; i < texture_res - j; ++i)
            {
                col += textures[g][index][k] * ucos[i] * vj;
                index++;
            }
        }
        return col;
    }

    // Helper function for trilinear interpolation coordinate and weight calculation
    template <uint32_t COLOR_DIM, typename T>
    inline __device__ void color_sample(
        at::PackedTensorAccessor32<const T, 4, at::RestrictPtrTraits> textures, // [N, Texture_Resolution, Texture_Resolution, 4]
        int texture_res,
        uint32_t g,
        T u, // from 0 to 1
        T v, // from 0 to 1
        T *ucos,
        T *vcos,
        T col[COLOR_DIM])
    {
        T vj;
        T uivj;
        int index = 0;
        for (int j = 0; j < texture_res; ++j)
        {
            vj = vcos[j];
            for (int i = 0; i < texture_res - j; ++i)
            {
                uivj = ucos[i] * vj;
                GSPLAT_PRAGMA_UNROLL
                for (int k = 0; k < COLOR_DIM; ++k)
                {
                    col[k] += textures[g][index][k] * uivj;
                }
                index++;
            }
        }
    }

    template <typename T>
    inline __device__ void update(
        at::PackedTensorAccessor32<T, 4, at::RestrictPtrTraits> v_textures, // [C, N, TEXTURE_DIM] or [nnz, TEXTURE_DIM]
        int texture_res,
        uint32_t g,
        T u, // u from 0 to 1
        T v, // v from 0 to 1
        T *ucos,
        T *vcos,
        uint32_t k,
        T delta)
    {
        T vj;
        int index = 0;
        for (int j = 0; j < texture_res; ++j)
        {
            vj = vcos[j];
            for (int i = 0; i < texture_res - j; ++i)
            {
                gpuAtomicAdd(&v_textures[g][index][k], delta * ucos[i] * vj);
            }
            index++;
        }
        return;
    }

    template <uint32_t COLOR_DIM, typename T>
    inline __device__ void color_sample_and_update(
        at::PackedTensorAccessor32<const T, 4, at::RestrictPtrTraits> textures, // [C, N, TEXTURE_DIM] or [nnz, TEXTURE_DIM]
        at::PackedTensorAccessor32<T, 4, at::RestrictPtrTraits> v_textures,     // [C, N, TEXTURE_DIM] or [nnz, TEXTURE_DIM]
        int texture_res,
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
        int index = 0;
        for (int j = 0; j < texture_res; ++j)
        {
            vj = vcos[j];
            for (int i = 0; i < texture_res - j; ++i)
            {
                uivj = ucos[i] * vj;
                GSPLAT_PRAGMA_UNROLL
                for (int k = 0; k < COLOR_DIM; ++k)
                {
                    gpuAtomicAdd(&v_textures[g][index][k], deltas[k] * uivj);
                    col[k] += textures[g][index][k] * uivj;
                }
                index++;
            }
        }
        return;
    }
} // namespace gsplat::dct3

#endif // GSPLAT_CUDA_DCT3_FILTER_H