#ifndef GSPLAT_CUDA_DCT_FILTER_H
#define GSPLAT_CUDA_DCT_FILTER_H

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
    inline __device__ T dct_sample(
        at::PackedTensorAccessor32<const T, 4, at::RestrictPtrTraits> textures, // [N, Texture_Resolution, Texture_Resolution, 4]
        int texture_res_x,
        int texture_res_y,
        uint32_t g,
        T u, // from 0 to 1
        T v, // from 0 to 1
        uint32_t k)
    {
        T col = 0;
        for (int i = 0; i < texture_res_x; ++i)
        {
            for (int j = 0; j < texture_res_y; ++j)
            {
                col += textures[g][i][j][k] * cos(M_PI * i * u) * cos(M_PI * j * v);
            }
        }
        return col;
    }

    template <typename T>
    inline __device__ void dct_update(
        at::PackedTensorAccessor32<T, 4, at::RestrictPtrTraits> v_textures, // [C, N, TEXTURE_DIM] or [nnz, TEXTURE_DIM]
        int texture_res_x,
        int texture_res_y,
        uint32_t g,
        T u, // u from 0 to 1
        T v, // v from 0 to 1
        uint32_t k,
        T delta)
    {
        for (int i = 0; i < texture_res_x; ++i)
        {
            for (int j = 0; j < texture_res_y; ++j)
            {
                gpuAtomicAdd(&v_textures[g][i][j][k], delta * cos(M_PI * i * u) * cos(M_PI * j * v));
            }
        }
        return;
    }
}

#endif // GSPLAT_CUDA_DCT_FILTER_H