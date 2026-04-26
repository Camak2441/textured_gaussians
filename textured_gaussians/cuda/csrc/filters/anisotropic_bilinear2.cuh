
#ifndef GSPLAT_CUDA_ANISOTROPIC_BILINEAR2_FILTER_H
#define GSPLAT_CUDA_ANISOTROPIC_BILINEAR2_FILTER_H

#include "../helpers.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/TensorAccessor.h>

#define FILTER_INV_SQUARE 2.0f

namespace gsplat::anisotropic_bilinear2
{

    template <typename T>
    inline __device__ vec2<T> s_to_uv(vec2<T> s, int texture_res_x, int texture_res_y, T x_range, T y_range)
    {
        return vec2<T>((s.x + x_range) / (x_range * 2) * (texture_res_x - 1),
                       (s.y + y_range) / (y_range * 2) * (texture_res_y - 1));
    }

    template <typename T>
    inline __device__ bool edge_normal(vec2<T> s0, vec2<T> s01, vec2<T> center, vec2<T> *n01)
    {
        if (s01.x == 0 && s01.y == 0)
        {
            *n01 = s0 - center;
        }
        else
        {
            *n01 = vec2<T>(s01.y, -s01.x);
        }
        T l = glm::length(*n01);
        if (l == 0)
            return false;
        *n01 /= l;
        return true;
    }

    template <typename T>
    inline __device__ T max4(T v0, T v1, T v2, T v3)
    {
        return max(max(v0, v1), max(v2, v3));
    }

    template <typename T>
    inline __device__ T min4(T v0, T v1, T v2, T v3)
    {
        return min(min(v0, v1), min(v2, v3));
    }

    // Clips the gaussian quad against the unit pixel [uv, uv+(1,1)] using
    // Sutherland-Hodgman, then computes polygon moments in pixel-local coords.
    // Fast path: if pixel centre is fully inside the gaussian, moments are set
    // analytically (A=1, Sx=0.5, Sy=0.5, Sxy=0.25) and returns true immediately.
    // Returns false when the clipped polygon has fewer than 3 vertices.
    template <typename T>
    inline __device__ bool clip_and_compute_moments(
        vec2<T> s0, vec2<T> s1, vec2<T> s2, vec2<T> s3,
        vec2<T> n01, vec2<T> n12, vec2<T> n23, vec2<T> n30,
        T n01max, T n12max, T n23max, T n30max,
        vec2<T> uv,
        T *A, T *Sx, T *Sy, T *Sxy)
    {
        // Fast path: pixel entirely inside gaussian
        vec2<T> s_center = uv + vec2<T>(T(0.5), T(0.5));
        if (glm::dot(s0 - s_center, n01) >= n01max &&
            glm::dot(s1 - s_center, n12) >= n12max &&
            glm::dot(s2 - s_center, n23) >= n23max &&
            glm::dot(s3 - s_center, n30) >= n30max)
        {
            *A = T(1);
            *Sx = T(0.5);
            *Sy = T(0.5);
            *Sxy = T(0.25);
            return true;
        }

        // Sutherland-Hodgman clip against [0,1]x[0,1] in pixel-local coords
        vec2<T> poly[8], tmp[8];
        int n = 4;
        poly[0] = s0 - uv;
        poly[1] = s1 - uv;
        poly[2] = s2 - uv;
        poly[3] = s3 - uv;

        // Clip x >= 0
        int m = 0;
        for (int i = 0; i < n; i++)
        {
            vec2<T> a = poly[i], b = poly[(i + 1) % n];
            bool a_in = (a.x >= T(0)), b_in = (b.x >= T(0));
            if (a_in)
                tmp[m++] = a;
            if (a_in != b_in)
            {
                T t = a.x / (a.x - b.x);
                tmp[m++] = vec2<T>(T(0), a.y + t * (b.y - a.y));
            }
        }
        n = m;
        if (n < 3)
            return false;

        // Clip x <= 1
        m = 0;
        for (int i = 0; i < n; i++)
        {
            vec2<T> a = tmp[i], b = tmp[(i + 1) % n];
            bool a_in = (a.x <= T(1)), b_in = (b.x <= T(1));
            if (a_in)
                poly[m++] = a;
            if (a_in != b_in)
            {
                T t = (a.x - T(1)) / (a.x - b.x);
                poly[m++] = vec2<T>(T(1), a.y + t * (b.y - a.y));
            }
        }
        n = m;
        if (n < 3)
            return false;

        // Clip y >= 0
        m = 0;
        for (int i = 0; i < n; i++)
        {
            vec2<T> a = poly[i], b = poly[(i + 1) % n];
            bool a_in = (a.y >= T(0)), b_in = (b.y >= T(0));
            if (a_in)
                tmp[m++] = a;
            if (a_in != b_in)
            {
                T t = a.y / (a.y - b.y);
                tmp[m++] = vec2<T>(a.x + t * (b.x - a.x), T(0));
            }
        }
        n = m;
        if (n < 3)
            return false;

        // Clip y <= 1
        m = 0;
        for (int i = 0; i < n; i++)
        {
            vec2<T> a = tmp[i], b = tmp[(i + 1) % n];
            bool a_in = (a.y <= T(1)), b_in = (b.y <= T(1));
            if (a_in)
                poly[m++] = a;
            if (a_in != b_in)
            {
                T t = (a.y - T(1)) / (a.y - b.y);
                poly[m++] = vec2<T>(a.x + t * (b.x - a.x), T(1));
            }
        }
        n = m;
        if (n < 3)
            return false;

        // Single-pass polygon moment computation
        // A   = 1/2  * sum c_i
        // Sx  = 1/6  * sum c_i*(x_i + x_{i+1})
        // Sy  = 1/6  * sum c_i*(y_i + y_{i+1})
        // Sxy = 1/24 * sum c_i*(2*x_i*y_i + x_i*y_{i+1} + x_{i+1}*y_i + 2*x_{i+1}*y_{i+1})
        // where c_i = cross2d(p_i, p_{i+1})
        *A = T(0);
        *Sx = T(0);
        *Sy = T(0);
        *Sxy = T(0);
        for (int i = 0; i < n; i++)
        {
            vec2<T> a = poly[i], b = poly[(i + 1) % n];
            T c = cross2d(a, b);
            *A += c;
            *Sx += c * (a.x + b.x);
            *Sy += c * (a.y + b.y);
            *Sxy += c * (T(2) * a.x * a.y + a.x * b.y + b.x * a.y + T(2) * b.x * b.y);
        }
        *A *= T(0.5);
        *Sx *= T(1) / T(6);
        *Sy *= T(1) / T(6);
        *Sxy *= T(1) / T(24);

        return true;
    }

    template <typename T>
    inline __device__ T precompute(
        vec2<T> *s0, vec2<T> *s1, vec2<T> *s2, vec2<T> *s3,
        vec2<T> *n01, vec2<T> *n12, vec2<T> *n23, vec2<T> *n30,
        T *n01max, T *n12max, T *n23max, T *n30max,
        int32_t *minu, int32_t *minv, int32_t *maxu, int32_t *maxv,
        int texture_res_x, int texture_res_y)
    {
        vec2<T> s01 = *s1 - *s0;
        vec2<T> s12 = *s2 - *s1;
        vec2<T> s23 = *s3 - *s2;
        vec2<T> s30 = *s0 - *s3;

        T area = T(0.5) * (cross2d(s01, s12) + cross2d(s23, s30));
        if (area < 0)
        {
            area *= -1;
            vec2<T> temp = *s1;
            *s1 = *s3;
            *s3 = temp;
            s01 = *s1 - *s0;
            s12 = *s2 - *s1;
            s23 = *s3 - *s2;
            s30 = *s0 - *s3;
        }

        *minu = max(-texture_res_y + 1, (int32_t)floor(min4(s0->x, s1->x, s2->x, s3->x)));
        *maxu = min(2 * texture_res_x - 2, (int32_t)ceil(max4(s0->x, s1->x, s2->x, s3->x)));
        *minv = max(-texture_res_y + 1, (int32_t)floor(min4(s0->y, s1->y, s2->y, s3->y)));
        *maxv = min(2 * texture_res_y - 2, (int32_t)ceil(max4(s0->y, s1->y, s2->y, s3->y)));

        vec2<T> center = (*s0 + *s1 + *s2 + *s3) / T(4);

        edge_normal(*s0, s01, center, n01);
        edge_normal(*s1, s12, center, n12);
        edge_normal(*s2, s23, center, n23);
        edge_normal(*s3, s30, center, n30);

        T n01c0 = abs(glm::dot(*n01, vec2<T>(T(0.5), T(0.5))));
        T n01c1 = abs(glm::dot(*n01, vec2<T>(T(0.5), T(-0.5))));
        T n12c0 = abs(glm::dot(*n12, vec2<T>(T(0.5), T(0.5))));
        T n12c1 = abs(glm::dot(*n12, vec2<T>(T(0.5), T(-0.5))));
        T n23c0 = abs(glm::dot(*n23, vec2<T>(T(0.5), T(0.5))));
        T n23c1 = abs(glm::dot(*n23, vec2<T>(T(0.5), T(-0.5))));
        T n30c0 = abs(glm::dot(*n30, vec2<T>(T(0.5), T(0.5))));
        T n30c1 = abs(glm::dot(*n30, vec2<T>(T(0.5), T(-0.5))));

        *n01max = max(n01c0, n01c1);
        *n12max = max(n12c0, n12c1);
        *n23max = max(n23c0, n23c1);
        *n30max = max(n30c0, n30c1);

        return area;
    }

    template <typename T>
    inline __device__ T sample(
        at::PackedTensorAccessor32<const T, 4, at::RestrictPtrTraits> textures,
        int32_t g, int32_t k,
        vec2<T> s0, vec2<T> s1, vec2<T> s2, vec2<T> s3,
        vec2<T> n01, vec2<T> n12, vec2<T> n23, vec2<T> n30,
        T n01max, T n12max, T n23max, T n30max,
        int32_t minu, int32_t maxu, int32_t minv, int32_t maxv,
        T area, T iarea,
        int texture_res_x, int texture_res_y)
    {
        T value = T(0);
        for (int v = minv; v < maxv; v++)
        {
            for (int u = minu; u < maxu; u++)
            {
                T A, Sx, Sy, Sxy;
                if (!clip_and_compute_moments(
                        s0, s1, s2, s3, n01, n12, n23, n30,
                        n01max, n12max, n23max, n30max,
                        vec2<T>(T(u), T(v)), &A, &Sx, &Sy, &Sxy))
                    continue;
                int u0 = max(0, min(u, texture_res_x - 1));
                int v0 = max(0, min(v, texture_res_y - 1));
                int u1 = max(0, min(u + 1, texture_res_x - 1));
                int v1 = max(0, min(u + 1, texture_res_y - 1));
                value += textures[g][v0][u0][k] * (A - Sx - Sy + Sxy) + textures[g][v0][u1][k] * (Sx - Sxy) + textures[g][v1][u0][k] * (Sy - Sxy) + textures[g][v1][u1][k] * Sxy;
            }
        }
        return value * iarea;
    }

    template <uint32_t COLOR_DIM, uint32_t alphai, typename T>
    inline __device__ void alpha_color_sample(
        at::PackedTensorAccessor32<const T, 4, at::RestrictPtrTraits> textures,
        int32_t g,
        vec2<T> s0, vec2<T> s1, vec2<T> s2, vec2<T> s3,
        vec2<T> n01, vec2<T> n12, vec2<T> n23, vec2<T> n30,
        T n01max, T n12max, T n23max, T n30max,
        int32_t minu, int32_t maxu, int32_t minv, int32_t maxv,
        T area, T iarea,
        int texture_res_x, int texture_res_y,
        T *alpha, T col[COLOR_DIM])
    {
        for (int v = minv; v < maxv; v++)
        {
            for (int u = minu; u < maxu; u++)
            {
                T A, Sx, Sy, Sxy;
                if (!clip_and_compute_moments(
                        s0, s1, s2, s3, n01, n12, n23, n30,
                        n01max, n12max, n23max, n30max,
                        vec2<T>(T(u), T(v)), &A, &Sx, &Sy, &Sxy))
                    continue;

                int u0 = max(0, min(u, texture_res_x - 1));
                int v0 = max(0, min(v, texture_res_y - 1));
                int u1 = max(0, min(u + 1, texture_res_x - 1));
                int v1 = max(0, min(u + 1, texture_res_y - 1));
                T w00 = (A - Sx - Sy + Sxy) * iarea;
                T w10 = (Sx - Sxy) * iarea;
                T w01 = (Sy - Sxy) * iarea;
                T w11 = Sxy * iarea;

                GSPLAT_PRAGMA_UNROLL
                for (int k = 0; k < COLOR_DIM; ++k)
                {
                    col[k] += textures[g][v0][u0][k] * w00 + textures[g][v0][u1][k] * w10 + textures[g][v1][u0][k] * w01 + textures[g][v1][u1][k] * w11;
                }
                *alpha += textures[g][v0][u0][alphai] * w00 + textures[g][v0][u1][alphai] * w10 + textures[g][v1][u0][alphai] * w01 + textures[g][v1][u1][alphai] * w11;
            }
        }
    }

    template <typename T>
    inline __device__ void update(
        at::PackedTensorAccessor32<T, 4, at::RestrictPtrTraits> v_textures,
        int32_t g, int32_t k,
        vec2<T> s0, vec2<T> s1, vec2<T> s2, vec2<T> s3,
        vec2<T> n01, vec2<T> n12, vec2<T> n23, vec2<T> n30,
        T n01max, T n12max, T n23max, T n30max,
        int32_t minu, int32_t maxu, int32_t minv, int32_t maxv,
        T area, T iarea,
        int texture_res_x, int texture_res_y, T delta)
    {
        T ndelta = delta * iarea;

        for (int v = minv; v < maxv; v++)
        {
            for (int u = minu; u < maxu; u++)
            {
                T A, Sx, Sy, Sxy;
                if (!clip_and_compute_moments(
                        s0, s1, s2, s3, n01, n12, n23, n30,
                        n01max, n12max, n23max, n30max,
                        vec2<T>(T(u), T(v)), &A, &Sx, &Sy, &Sxy))
                    continue;

                int u0 = max(0, min(u, texture_res_x - 1));
                int v0 = max(0, min(v, texture_res_y - 1));
                int u1 = max(0, min(u + 1, texture_res_x - 1));
                int v1 = max(0, min(u + 1, texture_res_y - 1));
                gpuAtomicAdd(&v_textures[g][v0][u0][k], ndelta * (A - Sx - Sy + Sxy));
                gpuAtomicAdd(&v_textures[g][v0][u1][k], ndelta * (Sx - Sxy));
                gpuAtomicAdd(&v_textures[g][v1][u0][k], ndelta * (Sy - Sxy));
                gpuAtomicAdd(&v_textures[g][v1][u1][k], ndelta * Sxy);
            }
        }
    }

    template <uint32_t COLOR_DIM, typename T>
    inline __device__ void color_sample_and_update(
        at::PackedTensorAccessor32<const T, 4, at::RestrictPtrTraits> textures,
        at::PackedTensorAccessor32<T, 4, at::RestrictPtrTraits> v_textures,
        int32_t g,
        vec2<T> s0, vec2<T> s1, vec2<T> s2, vec2<T> s3,
        vec2<T> n01, vec2<T> n12, vec2<T> n23, vec2<T> n30,
        T n01max, T n12max, T n23max, T n30max,
        int32_t minu, int32_t maxu, int32_t minv, int32_t maxv,
        T area, T iarea,
        int texture_res_x, int texture_res_y,
        T col[COLOR_DIM], T deltas[COLOR_DIM])
    {
        T ndeltas[COLOR_DIM];
        GSPLAT_PRAGMA_UNROLL
        for (int k = 0; k < COLOR_DIM; k++)
        {
            ndeltas[k] = deltas[k] * iarea;
        }

        for (int v = minv; v < maxv; v++)
        {
            for (int u = minu; u < maxu; u++)
            {
                T A, Sx, Sy, Sxy;
                if (!clip_and_compute_moments(
                        s0, s1, s2, s3, n01, n12, n23, n30,
                        n01max, n12max, n23max, n30max,
                        vec2<T>(T(u), T(v)), &A, &Sx, &Sy, &Sxy))
                    continue;

                int u0 = max(0, min(u, texture_res_x - 1));
                int v0 = max(0, min(v, texture_res_y - 1));
                int u1 = max(0, min(u + 1, texture_res_x - 1));
                int v1 = max(0, min(u + 1, texture_res_y - 1));
                T w00 = (A - Sx - Sy + Sxy);
                T w10 = (Sx - Sxy);
                T w01 = (Sy - Sxy);
                T w11 = Sxy;

                GSPLAT_PRAGMA_UNROLL
                for (int k = 0; k < COLOR_DIM; ++k)
                {
                    col[k] += textures[g][v0][u0][k] * w00 + textures[g][v0][u1][k] * w10 + textures[g][v1][u0][k] * w01 + textures[g][v1][u1][k] * w11;
                    gpuAtomicAdd(&v_textures[g][v0][u0][k], ndeltas[k] * w00);
                    gpuAtomicAdd(&v_textures[g][v0][u1][k], ndeltas[k] * w10);
                    gpuAtomicAdd(&v_textures[g][v1][u0][k], ndeltas[k] * w01);
                    gpuAtomicAdd(&v_textures[g][v1][u1][k], ndeltas[k] * w11);
                }
            }
        }
    }

} // namespace gsplat::anisotropic_bilinear2

#endif // GSPLAT_CUDA_ANISOTROPIC_BILINEAR2_FILTER_H
