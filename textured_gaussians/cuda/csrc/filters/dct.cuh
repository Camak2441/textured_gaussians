#ifndef GSPLAT_CUDA_DCT_FILTER_H
#define GSPLAT_CUDA_DCT_FILTER_H

#include "../helpers.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/TensorAccessor.h>

#define FILTER_INV_SQUARE 2.0f
#define ISQRT2 0.70710678118f

namespace gsplat::dct
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

    // stride_x / stride_y allow column-major shared-memory layout.
    // Default stride=1 preserves row-major behaviour for callers that don't opt in.

    template <typename T>
    inline __device__ void precompute(
        int texture_res_x,
        int texture_res_y,
        T u, // from 0 to 1
        T v,
        T *ucos,
        T *vcos,
        int stride_x = 1,
        int stride_y = 1)
    {
        ucos[0] = T(1);
        vcos[0] = T(1);
        T rsqrti;
        T pii = 0;
        for (int i = 1; i < texture_res_x; ++i)
        {
            rsqrti = rsqrt((T)(i + 1));
            pii += M_PI;
            ucos[i * stride_x] = dct_cos(pii * u) * rsqrti;
        }
        pii = 0;
        for (int i = 1; i < texture_res_y; ++i)
        {
            rsqrti = rsqrt((T)(i + 1));
            pii += M_PI;
            vcos[i * stride_y] = dct_cos(pii * v) * rsqrti;
        }
    }

    template <typename T>
    inline __device__ void grad_precompute(
        int texture_res_x,
        int texture_res_y,
        T u, // from 0 to 1
        T v,
        T *ucos,
        T *vcos,
        T *ducos,
        T *dvcos,
        int stride_x = 1,
        int stride_y = 1)
    {
        ucos[0] = T(1);
        vcos[0] = T(1);
        ducos[0] = T(0);
        dvcos[0] = T(0);
        T rsqrti;
        T pii = 0;
        T piiuv;
        for (int i = 1; i < texture_res_x; ++i)
        {
            rsqrti = rsqrt((T)(i + 1));
            pii += M_PI;
            piiuv = pii * u;
            ucos[i * stride_x] = dct_cos(piiuv) * rsqrti;
            ducos[i * stride_x] = pii * dct_cos(piiuv + M_PI_2) * rsqrti;
        }
        pii = 0;
        for (int i = 1; i < texture_res_y; ++i)
        {
            rsqrti = rsqrt((T)(i + 1));
            pii += M_PI;
            piiuv = pii * v;
            vcos[i * stride_y] = dct_cos(piiuv) * rsqrti;
            dvcos[i * stride_y] = pii * dct_cos(piiuv + M_PI_2) * rsqrti;
        }
    }

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
        uint32_t k,
        int stride_x = 1,
        int stride_y = 1)
    {
        T col = 0;
        T vj;
        for (int j = 0; j < texture_res_y; ++j)
        {
            vj = vcos[j * stride_y];
            for (int i = 0; i < texture_res_x; ++i)
            {
                col += textures[g][j][i][k] * ucos[i * stride_x] * vj;
            }
        }
        return col;
    }

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
        T col[COLOR_DIM],
        int stride_x = 1,
        int stride_y = 1)
    {
        T vj;
        T uivj;
        for (int j = 0; j < texture_res_y; ++j)
        {
            vj = vcos[j * stride_y];
            for (int i = 0; i < texture_res_x; ++i)
            {
                uivj = ucos[i * stride_x] * vj;
                GSPLAT_PRAGMA_UNROLL
                for (int k = 0; k < COLOR_DIM; ++k)
                {
                    col[k] += textures[g][j][i][k] * uivj;
                }
            }
        }
    }

    template <typename T>
    inline __device__ void update(
        at::PackedTensorAccessor32<T, 4, at::RestrictPtrTraits> v_textures, // [C, N, TEXTURE_DIM] or [nnz, TEXTURE_DIM]
        int texture_res_x,
        int texture_res_y,
        uint32_t g,
        T u, // u from 0 to 1
        T v, // v from 0 to 1
        T *ucos,
        T *vcos,
        uint32_t k,
        T delta,
        int stride_x = 1,
        int stride_y = 1)
    {
        T vj;
        for (int j = 0; j < texture_res_y; ++j)
        {
            vj = vcos[j * stride_y];
            for (int i = 0; i < texture_res_x; ++i)
            {
                gpuAtomicAdd(&v_textures[g][j][i][k], delta * ucos[i * stride_x] * vj);
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
        T deltas[COLOR_DIM],
        int stride_x = 1,
        int stride_y = 1)
    {
        T vj;
        T uivj;
        for (int j = 0; j < texture_res_y; ++j)
        {
            vj = vcos[j * stride_y];
            for (int i = 0; i < texture_res_x; ++i)
            {
                uivj = ucos[i * stride_x] * vj;
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

    template <typename T>
    inline __device__ void sample_grad(
        at::PackedTensorAccessor32<const T, 4, at::RestrictPtrTraits> textures, // [C, N, TEXTURE_DIM] or [nnz, TEXTURE_DIM]
        int texture_res_x,
        int texture_res_y,
        uint32_t g,
        T u, // u from 0 to 1
        T v, // v from 0 to 1
        T *ucos,
        T *vcos,
        T *ducos,
        T *dvcos,
        uint32_t k,
        vec2<T> *v_s_tex,
        T v_ck,
        int stride_x = 1,
        int stride_y = 1)
    {
        T vj;
        T dvj;
        T duivj;
        T uidvj;
        for (int j = 0; j < texture_res_y; ++j)
        {
            vj = vcos[j * stride_y];
            dvj = dvcos[j * stride_y];
            for (int i = 0; i < texture_res_x; ++i)
            {
                duivj = ducos[i * stride_x] * vj;
                uidvj = ucos[i * stride_x] * dvj;
                const T v_tex_k = v_ck;
                const T tex_val = textures[g][j][i][k];
                v_s_tex->x += duivj * tex_val * v_tex_k;
                v_s_tex->y += uidvj * tex_val * v_tex_k;
            }
        }
        return;
    }

    template <uint32_t COLOR_DIM, typename T>
    inline __device__ void color_sample_grad(
        at::PackedTensorAccessor32<const T, 4, at::RestrictPtrTraits> textures, // [C, N, TEXTURE_DIM] or [nnz, TEXTURE_DIM]
        int texture_res_x,
        int texture_res_y,
        uint32_t g,
        T u, // u from 0 to 1
        T v, // v from 0 to 1
        T *ucos,
        T *vcos,
        T *ducos,
        T *dvcos,
        vec2<T> *v_s_tex,
        T *v_render_c,
        T fac,
        int stride_x = 1,
        int stride_y = 1)
    {
        T vj;
        T dvj;
        T duivj;
        T uidvj;
        for (int j = 0; j < texture_res_y; ++j)
        {
            vj = vcos[j * stride_y];
            dvj = dvcos[j * stride_y];
            for (int i = 0; i < texture_res_x; ++i)
            {
                duivj = ducos[i * stride_x] * vj;
                uidvj = ucos[i * stride_x] * dvj;
                GSPLAT_PRAGMA_UNROLL
                for (int k = 0; k < COLOR_DIM; ++k)
                {
                    const T v_tex_k = fac * v_render_c[k];
                    const T tex_val = textures[g][j][i][k];
                    v_s_tex->x += duivj * tex_val * v_tex_k;
                    v_s_tex->y += uidvj * tex_val * v_tex_k;
                }
            }
        }
        return;
    }
} // namespace gsplat::dct

#endif // GSPLAT_CUDA_DCT_FILTER_H