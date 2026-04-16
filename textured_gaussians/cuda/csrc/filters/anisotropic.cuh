#ifndef GSPLAT_CUDA_ANISOTROPIC_FILTER_H
#define GSPLAT_CUDA_ANISOTROPIC_FILTER_H

#include "../helpers.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/TensorAccessor.h>

#define FILTER_INV_SQUARE 2.0f
#define ISQRT2 0.70710678118f

namespace gsplat
{
    template <typename T>
    inline __device__ T line_ratio_outside_pixel(
        vec2<T> n, T min_axis, T max_axis, T d)
    {
        if (n.x == T(0) || n.y == T(0))
        {
            return max(T(0), min(T(0.5) - d, T(1)));
        }

        T ratio_out = T(0);
        T height;
        T max_height = max_axis - min_axis;
        T inxy = T(1) / abs(n.x * n.y);
        height = max(T(0), min(max_axis - d, max_height));
        ratio_out += T(0.5) * height * height * inxy;
        height = max(T(0), min(min_axis - d, T(2) * min_axis));
        ratio_out += height * max_height * inxy;
        height = max(T(0), min(-min_axis - d, max_height));
        ratio_out += T(0.5) * height * (T(2) * max_height - height) * inxy;
        return ratio_out;
    }

    template <typename T>
    inline __device__ T ratio_inside_pixel(
        vec2<T> s0, vec2<T> s1, vec2<T> s2, vec2<T> s3,
        vec2<T> n01, vec2<T> n12, vec2<T> n23, vec2<T> n30,
        T n01min, T n01max, T n12min, T n12max, T n23min, T n23max, T n30min, T n30max,
        vec2<T> s)
    {
        vec2<T> ps0 = s0 - s;
        vec2<T> ps1 = s1 - s;
        vec2<T> ps2 = s2 - s;
        vec2<T> ps3 = s3 - s;
        T d01 = glm::dot(ps0, n01);
        T d12 = glm::dot(ps1, n12);
        T d23 = glm::dot(ps2, n23);
        T d30 = glm::dot(ps3, n30);

        if (d01 >= n01max && d12 >= n12max && d23 >= n23max && d30 >= n30max)
        {
            return T(1);
        }
        if ((d01 <= -n01max || d01 >= n01max) && (d12 <= -n12max || d12 >= n12max) && (d23 <= -n23max || d23 >= n23max) && (d30 <= -n30max || d30 >= n30max))
        {
            return T(0);
        }

        // Remove the regions outside the pixel part by part
        T ratio_in = T(1);

        ratio_in -= line_ratio_outside_pixel(n01, n01min, n01max, d01);
        ratio_in -= line_ratio_outside_pixel(n12, n12min, n12max, d12);
        ratio_in -= line_ratio_outside_pixel(n23, n23min, n23max, d23);
        ratio_in -= line_ratio_outside_pixel(n30, n30min, n30max, d30);

        return max(T(0), min(ratio_in, T(1)));
    }

    template <typename T>
    inline __device__ T ratio_corners_inside_pixel(vec2<T> s0, vec2<T> s1, vec2<T> s2, vec2<T> s3, vec2<T> uv)
    {
        vec2<T> poly[8];
        vec2<T> tmp[8];
        int n = 4;
        poly[0] = s0 - uv;
        poly[1] = s1 - uv;
        poly[2] = s2 - uv;
        poly[3] = s3 - uv;

        int m = 0;
        for (int i = 0; i < n; i++)
        {
            vec2<T> a = poly[i], b = poly[(i + 1) % n];
            bool a_in = (a.x >= 0), b_in = (b.x >= 0);
            if (a_in)
                tmp[m++] = a;
            if (a_in != b_in)
            {
                T t = a.x / (a.x - b.x);
                tmp[m++] = vec2<T>(T(0), a.y + t * (b.y - a.y));
            }
        }
        n = m;
        if (n == 0)
            return T(0);

        m = 0;
        for (int i = 0; i < n; i++)
        {
            vec2<T> a = tmp[i], b = tmp[(i + 1) % n];
            bool a_in = (a.x <= 1), b_in = (b.x <= 1);
            if (a_in)
                poly[m++] = a;
            if (a_in != b_in)
            {
                T t = (a.x - T(1)) / (a.x - b.x);
                poly[m++] = vec2<T>(T(1), a.y + t * (b.y - a.y));
            }
        }
        n = m;
        if (n == 0)
            return T(0);

        m = 0;
        for (int i = 0; i < n; i++)
        {
            vec2<T> a = poly[i], b = poly[(i + 1) % n];
            bool a_in = (a.y >= 0), b_in = (b.y >= 0);
            if (a_in)
                tmp[m++] = a;
            if (a_in != b_in)
            {
                T t = a.y / (a.y - b.y);
                tmp[m++] = vec2<T>(a.x + t * (b.x - a.x), T(0));
            }
        }
        n = m;
        if (n == 0)
            return T(0);

        m = 0;
        for (int i = 0; i < n; i++)
        {
            vec2<T> a = tmp[i], b = tmp[(i + 1) % n];
            bool a_in = (a.y <= 1), b_in = (b.y <= 1);
            if (a_in)
                poly[m++] = a;
            if (a_in != b_in)
            {
                T t = (a.y - T(1)) / (a.y - b.y);
                poly[m++] = vec2<T>(a.x + t * (b.x - a.x), T(1));
            }
        }
        n = m;
        if (n == 0)
            return T(0);

        T area = T(0);
        for (int i = 1; i < n - 1; i++)
            area += cross2d(poly[i] - poly[0], poly[i + 1] - poly[0]);
        return area * T(0.5);
    }

    template <typename T>
    inline __device__ vec2<T> convert_s_to_uv(vec2<T> s, int texture_res_x, int texture_res_y)
    {
        return vec2<T>((s.x + T(3)) / T(6) * texture_res_x, (s.y + T(3)) / T(6) * texture_res_y);
    }

    template <typename T>
    inline __device__ bool edge_normal(vec2<T> s0, vec2<T> s01, vec2<T> center, vec2<T> *n01)
    {
        *n01 = vec2<T>(s01.y, -s01.x);
        if (glm::length(*n01) == 0)
        {
            *n01 = s0 - center;
            if (glm::length(*n01) == 0)
            {
                return false;
            }
        }
        *n01 /= glm::length(*n01);
        // The winding should always be CCW so the normal should always point away from the centre.
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

    template <typename T>
    inline __device__ T precompute_aniso_data(
        vec2<T> *s0, vec2<T> *s1, vec2<T> *s2, vec2<T> *s3,
        vec2<T> *n01, vec2<T> *n12, vec2<T> *n23, vec2<T> *n30,
        T *n01min, T *n01max, T *n12min, T *n12max, T *n23min, T *n23max, T *n30min, T *n30max,
        int32_t *minu, int32_t *minv, int32_t *maxu, int32_t *maxv,
        vec2<int> *s0texel, vec2<int> *s1texel, vec2<int> *s2texel, vec2<int> *s3texel,
        int texture_res_x, int texture_res_y)
    {
        vec2<T> s01 = *s1 - *s0;
        vec2<T> s12 = *s2 - *s1;
        vec2<T> s23 = *s3 - *s2;
        vec2<T> s30 = *s0 - *s3;

        T area = T(0.5) * (cross2d(s01, s12) + cross2d(s23, s30));
        if (area < 0)
        {
            // Ensure the winding is CCW
            area *= -1;

            vec2<T> temp = *s1;
            *s1 = *s3;
            *s3 = temp;

            s01 = *s1 - *s0;
            s12 = *s2 - *s1;
            s23 = *s3 - *s2;
            s30 = *s0 - *s3;
        }

        *s0texel = vec2<int>(floor(s0->x), floor(s0->y));
        *s1texel = vec2<int>(floor(s1->x), floor(s1->y));
        *s2texel = vec2<int>(floor(s2->x), floor(s2->y));
        *s3texel = vec2<int>(floor(s3->x), floor(s3->y));

        *minu = max(0, (int32_t)floor(min4(s0->x, s1->x, s2->x, s3->x)));
        *maxu = min(texture_res_x, (int32_t)ceil(max4(s0->x, s1->x, s2->x, s3->x)));
        *minv = max(0, (int32_t)floor(min4(s0->y, s1->y, s2->y, s3->y)));
        *maxv = min(texture_res_y, (int32_t)ceil(max4(s0->y, s1->y, s2->y, s3->y)));

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

        *n01min = min(n01c0, n01c1);
        *n01max = max(n01c0, n01c1);
        *n12min = min(n12c0, n12c1);
        *n12max = max(n12c0, n12c1);
        *n23min = min(n23c0, n23c1);
        *n23max = max(n23c0, n23c1);
        *n30min = min(n30c0, n30c1);
        *n30max = max(n30c0, n30c1);

        return area;
    }

    // Helper function for anisotropic sampling
    template <typename T>
    inline __device__ T anisotropic_sample(
        at::PackedTensorAccessor32<const T, 4, at::RestrictPtrTraits> textures,
        int32_t g, int32_t k,
        vec2<T> s0, vec2<T> s1, vec2<T> s2, vec2<T> s3,
        vec2<T> n01, vec2<T> n12, vec2<T> n23, vec2<T> n30,
        T n01min, T n01max, T n12min, T n12max, T n23min, T n23max, T n30min, T n30max,
        int32_t minu, int32_t maxu, int32_t minv, int32_t maxv,
        vec2<int> s0texel, vec2<int> s1texel, vec2<int> s2texel, vec2<int> s3texel,
        T area, T iarea,
        int texture_res_x, int texture_res_y)
    {
        if (minu + 1 == maxu && minv + 1 == maxv)
        {
            return textures[g][minv][minu][k];
        }

        T value = T(0);
        for (int v = minv; v < maxv; v++)
        {
            for (int u = minu; u < maxu; u++)
            {
                T pixel_area;
                if ((u == s0texel.x && v == s0texel.y) || (u == s1texel.x && v == s1texel.y) || (u == s2texel.x && v == s2texel.y) || (u == s3texel.x && v == s3texel.y))
                {
                    pixel_area = ratio_corners_inside_pixel(s0, s1, s2, s3, vec2<T>(u, v));
                }
                else
                {
                    pixel_area = ratio_inside_pixel(s0, s1, s2, s3, n01, n12, n23, n30, n01min, n01max, n12min, n12max, n23min, n23max, n30min, n30max, vec2<T>(T(u) + T(0.5), T(v) + T(0.5)));
                }
                value += textures[g][v][u][k] * pixel_area;
            }
        }

        return value * iarea;
    }

    // Helper function for anisotropic sampling
    template <uint32_t COLOR_DIM, uint32_t alphai, typename T>
    inline __device__ void anisotropic_alpha_color_sample(
        at::PackedTensorAccessor32<const T, 4, at::RestrictPtrTraits> textures,
        int32_t g,
        vec2<T> s0, vec2<T> s1, vec2<T> s2, vec2<T> s3,
        vec2<T> n01, vec2<T> n12, vec2<T> n23, vec2<T> n30,
        T n01min, T n01max, T n12min, T n12max, T n23min, T n23max, T n30min, T n30max,
        int32_t minu, int32_t maxu, int32_t minv, int32_t maxv,
        vec2<int> s0texel, vec2<int> s1texel, vec2<int> s2texel, vec2<int> s3texel,
        T area, T iarea,
        int texture_res_x, int texture_res_y,
        T *alpha,
        T col[COLOR_DIM])
    {
        if (minu + 1 == maxu && minv + 1 == maxv)
        {
            GSPLAT_PRAGMA_UNROLL
            for (int k = 0; k < COLOR_DIM; ++k)
            {
                col[k] += textures[g][minv][minu][k];
            }
            *alpha += textures[g][minv][minu][alphai];
            return;
        }

        for (int v = minv; v < maxv; v++)
        {
            for (int u = minu; u < maxu; u++)
            {
                T pixel_area;
                if ((u == s0texel.x && v == s0texel.y) || (u == s1texel.x && v == s1texel.y) || (u == s2texel.x && v == s2texel.y) || (u == s3texel.x && v == s3texel.y))
                {
                    pixel_area = ratio_corners_inside_pixel(s0, s1, s2, s3, vec2<T>(u, v)) * iarea;
                }
                else
                {
                    pixel_area = ratio_inside_pixel(s0, s1, s2, s3, n01, n12, n23, n30, n01min, n01max, n12min, n12max, n23min, n23max, n30min, n30max, vec2<T>(T(u) + T(0.5), T(v) + T(0.5))) * iarea;
                }
                GSPLAT_PRAGMA_UNROLL
                for (int k = 0; k < COLOR_DIM; ++k)
                {
                    col[k] += textures[g][v][u][k] * pixel_area;
                }
                *alpha += textures[g][v][u][alphai] * pixel_area;
            }
        }

        return;
    }

    // Helper function for anisotropic sampling
    // Please pre-multiply delta by the area scale factor
    template <typename T>
    inline __device__ void anisotropic_update(
        at::PackedTensorAccessor32<T, 4, at::RestrictPtrTraits> v_textures,
        int32_t g, int32_t k,
        vec2<T> s0, vec2<T> s1, vec2<T> s2, vec2<T> s3,
        vec2<T> n01, vec2<T> n12, vec2<T> n23, vec2<T> n30,
        T n01min, T n01max, T n12min, T n12max, T n23min, T n23max, T n30min, T n30max,
        int32_t minu, int32_t maxu, int32_t minv, int32_t maxv,
        vec2<int> s0texel, vec2<int> s1texel, vec2<int> s2texel, vec2<int> s3texel,
        T area, T iarea,
        int texture_res_x, int texture_res_y, T delta)
    {
        if (minu + 1 == maxu && minv + 1 == maxv)
        {
            gpuAtomicAdd(&v_textures[g][minv][minu][k], delta);
            return;
        }

        for (int v = minv; v < maxv; v++)
        {
            for (int u = minu; u < maxu; u++)
            {
                T pixel_area;
                if ((u == s0texel.x && v == s0texel.y) || (u == s1texel.x && v == s1texel.y) || (u == s2texel.x && v == s2texel.y) || (u == s3texel.x && v == s3texel.y))
                {
                    pixel_area = ratio_corners_inside_pixel(s0, s1, s2, s3, vec2<T>(u, v));
                }
                else
                {
                    pixel_area = ratio_inside_pixel(s0, s1, s2, s3, n01, n12, n23, n30, n01min, n01max, n12min, n12max, n23min, n23max, n30min, n30max, vec2<T>(T(u) + T(0.5), T(v) + T(0.5)));
                }
                gpuAtomicAdd(&v_textures[g][v][u][k], delta * pixel_area * iarea);
            }
        }

        return;
    }

    // Helper function for anisotropic update
    template <uint32_t COLOR_DIM, typename T>
    inline __device__ void anisotropic_color_sample_and_update(
        at::PackedTensorAccessor32<const T, 4, at::RestrictPtrTraits> textures,
        at::PackedTensorAccessor32<T, 4, at::RestrictPtrTraits> v_textures,
        int32_t g,
        vec2<T> s0, vec2<T> s1, vec2<T> s2, vec2<T> s3,
        vec2<T> n01, vec2<T> n12, vec2<T> n23, vec2<T> n30,
        T n01min, T n01max, T n12min, T n12max, T n23min, T n23max, T n30min, T n30max,
        int32_t minu, int32_t maxu, int32_t minv, int32_t maxv,
        vec2<int> s0texel, vec2<int> s1texel, vec2<int> s2texel, vec2<int> s3texel,
        T area, T iarea,
        int texture_res_x, int texture_res_y,
        T col[COLOR_DIM],
        T deltas[COLOR_DIM])
    {
        if (minu + 1 == maxu && minv + 1 == maxv)
        {
            GSPLAT_PRAGMA_UNROLL
            for (int k = 0; k < COLOR_DIM; ++k)
            {
                col[k] += textures[g][minv][minu][k];
                gpuAtomicAdd(&v_textures[g][minv][minu][k], deltas[k]);
            }
            return;
        }

        for (int v = minv; v < maxv; v++)
        {
            for (int u = minu; u < maxu; u++)
            {
                T pixel_area;
                if ((u == s0texel.x && v == s0texel.y) || (u == s1texel.x && v == s1texel.y) || (u == s2texel.x && v == s2texel.y) || (u == s3texel.x && v == s3texel.y))
                {
                    pixel_area = ratio_corners_inside_pixel(s0, s1, s2, s3, vec2<T>(u, v)) * iarea;
                }
                else
                {
                    pixel_area = ratio_inside_pixel(s0, s1, s2, s3, n01, n12, n23, n30, n01min, n01max, n12min, n12max, n23min, n23max, n30min, n30max, vec2<T>(T(u) + T(0.5), T(v) + T(0.5))) * iarea;
                }
                GSPLAT_PRAGMA_UNROLL
                for (int k = 0; k < COLOR_DIM; ++k)
                {
                    col[k] += textures[g][v][u][k] * pixel_area;
                    gpuAtomicAdd(&v_textures[g][v][u][k], deltas[k] * pixel_area);
                }
            }
        }

        return;
    }
}

#endif // GSPLAT_CUDA_ANISOTROPIC_FILTER_H