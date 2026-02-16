#ifndef GSPLAT_CUDA_ANISOTROPIC_FILTER_H
#define GSPLAT_CUDA_ANISOTROPIC_FILTER_H

#include "helpers.cuh"

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
        if (n.x == 0 || n.y == 0)
        {
            return 0.5f - d;
        }

        T ratio_out = 0.0f;
        T height;
        T max_height = max_axis - min_axis;
        T nxy = n.x * n.y;
        height = max(0.0f, min(max_axis - d, max_height));
        ratio_out += 0.5f * height * height / nxy;
        height = max(0.0f, min(min_axis - d, 2 * min_axis));
        ratio_out += height * max_height / nxy;
        height = max(0.0f, min(-min_axis - d, max_height));
        ratio_out += 0.5f * (max_height * max_height - height * height) / nxy;
        return ratio_out;
    }

    template <typename T>
    inline __device__ T ratio_inside_pixel(
        vec2<T> s0, vec2<T> s1, vec2<T> s2, vec2<T> s3,
        vec2<T> n01, vec2<T> n12, vec2<T> n23, vec2<T> n30,
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

        vec2<T> v01 = d01 * n01;
        vec2<T> v12 = d12 * n12;
        vec2<T> v23 = d23 * n23;
        vec2<T> v30 = d30 * n30;

        const T irt2 = (T)(1 / sqrtf(2));

        T n01c0 = abs(glm::dot(n01, vec2<T>(0.5f, 0.5f)));
        T n01c1 = abs(glm::dot(n01, vec2<T>(0.5f, -0.5f)));
        T n12c0 = abs(glm::dot(n12, vec2<T>(0.5f, 0.5f)));
        T n12c1 = abs(glm::dot(n12, vec2<T>(0.5f, -0.5f)));
        T n23c0 = abs(glm::dot(n23, vec2<T>(0.5f, 0.5f)));
        T n23c1 = abs(glm::dot(n23, vec2<T>(0.5f, -0.5f)));
        T n30c0 = abs(glm::dot(n30, vec2<T>(0.5f, 0.5f)));
        T n30c1 = abs(glm::dot(n30, vec2<T>(0.5f, -0.5f)));

        T n01min = min(n01c0, n01c1);
        T n01max = max(n01c0, n01c1);
        T n12min = min(n12c0, n12c1);
        T n12max = max(n12c0, n12c1);
        T n23min = min(n23c0, n23c1);
        T n23max = max(n23c0, n23c1);
        T n30min = min(n30c0, n30c1);
        T n30max = max(n30c0, n30c1);

        if (d01 >= n01max && d12 >= n12max && d23 >= n23max && d30 >= n30max)
        {
            return 1.0f;
        }
        if ((d01 <= -n01max || d01 >= n01max) && (d12 <= -n12max || d12 >= n12max) && (d23 <= -n23max || d23 >= n23max) && (d30 <= -n30max || d30 >= n30max))
        {
            return 0.0f;
        }

        // Remove the regions outside the pixel part by part
        T ratio_in = 1.0f;

        ratio_in -= line_ratio_outside_pixel(n01, n01min, n01max, d01);
        ratio_in -= line_ratio_outside_pixel(n12, n12min, n12max, d12);
        ratio_in -= line_ratio_outside_pixel(n23, n23min, n23max, d23);
        ratio_in -= line_ratio_outside_pixel(n30, n30min, n30max, d30);

        return max(0.0f, min(ratio_in, 1.0f));
    }

    template <typename T>
    inline __device__ vec2<T> convert_s_to_uv(vec2<T> s, int texture_res_x, int texture_res_y)
    {
        return vec2<T>((T)((s.x + 3.0f) / 6.0f * (texture_res_x - 1) / 2.0f), (T)((s.y + 3.0f) / 6.0f * (texture_res_y - 1) / 2.0f));
    }

    template <typename T>
    inline __device__ bool edge_normal(vec2<T> s0, vec2<T> s1, vec2<T> center, vec2<T> *n01)
    {
        *n01 = vec2<T>(-(s1.y - s0.y), s1.x - s0.x);
        if (glm::length(*n01) == 0)
        {
            *n01 = s0 - center;
            if (glm::length(*n01) == 0)
            {
                return false;
            }
        }
        *n01 /= glm::length(*n01);
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

    // Helper function for anisotropic sampling
    template <typename T>
    inline __device__ T anisotropic_sample(
        at::PackedTensorAccessor32<const T, 4, at::RestrictPtrTraits> textures,
        int32_t g, int32_t k,
        vec2<T> s0, vec2<T> s1, vec2<T> s2, vec2<T> s3,
        int texture_res_x, int texture_res_y)
    {
        s0 = convert_s_to_uv(s0, texture_res_x, texture_res_y);
        s1 = convert_s_to_uv(s1, texture_res_x, texture_res_y);
        s2 = convert_s_to_uv(s2, texture_res_x, texture_res_y);
        s3 = convert_s_to_uv(s3, texture_res_x, texture_res_y);

        int32_t minu = floor(max(0.0f, min4(s0.x, s1.x, s2.x, s3.x) - 1.0f));
        int32_t maxu = ceil(min((T)texture_res_x - 1, max4(s0.x, s1.x, s2.x, s3.x) + 1.0f));
        int32_t minv = floor(max(0.0f, min4(s0.y, s1.y, s2.y, s3.y) - 1.0f));
        int32_t maxv = ceil(min((T)texture_res_y - 1, max4(s0.y, s1.y, s2.y, s3.y) + 1.0f));

        vec2<T> center = (s0 + s1 + s2 + s3) / 4.0f;

        vec2<T> n01;
        vec2<T> n12;
        vec2<T> n23;
        vec2<T> n30;

        if (!(edge_normal(s0, s1, center, &n01) && edge_normal(s1, s2, center, &n12) && edge_normal(s2, s3, center, &n23) && edge_normal(s3, s0, center, &n30)))
        {
            return 0.0f;
        }

        T color = 0.0f;
        T area = 0.0f;

        for (int u = minu; u <= maxu; u++)
        {
            for (int v = minv; v <= maxv; v++)
            {
                T pixel_area = ratio_inside_pixel(s0, s1, s2, s3, n01, n12, n23, n30, vec2<T>(u, v));
                color += textures[g][u][v][k] * pixel_area;
                area += pixel_area;
            }
        }

        if (area > 0)
        {
            return color / area;
        }
        return 0.0f;
    }

    // Helper function for anisotropic sampling
    template <typename T>
    inline __device__ void anisotropic_update(
        at::PackedTensorAccessor32<T, 4, at::RestrictPtrTraits> v_textures,
        int32_t g, int32_t k,
        vec2<T> s0, vec2<T> s1, vec2<T> s2, vec2<T> s3,
        int texture_res_x, int texture_res_y, T delta)
    {
        s0 = convert_s_to_uv(s0, texture_res_x, texture_res_y);
        s1 = convert_s_to_uv(s1, texture_res_x, texture_res_y);
        s2 = convert_s_to_uv(s2, texture_res_x, texture_res_y);
        s3 = convert_s_to_uv(s3, texture_res_x, texture_res_y);

        int32_t minu = floor(max(0.0f, min4(s0.x, s1.x, s2.x, s3.x) - 1));
        int32_t maxu = ceil(min((T)texture_res_x - 1, max4(s0.x, s1.x, s2.x, s3.x) + 1));
        int32_t minv = floor(max(0.0f, min4(s0.y, s1.y, s2.y, s3.y) - 1));
        int32_t maxv = ceil(min((T)texture_res_y - 1, max4(s0.y, s1.y, s2.y, s3.y) + 1));

        vec2<T> center = (s0 + s1 + s2 + s3) / 4.0f;

        vec2<T> n01;
        vec2<T> n12;
        vec2<T> n23;
        vec2<T> n30;

        if (!(edge_normal(s0, s1, center, &n01) && edge_normal(s1, s2, center, &n12) && edge_normal(s2, s3, center, &n23) && edge_normal(s3, s0, center, &n30)))
        {
            return;
        }

        T area = 0.0f;

        for (int u = minu; u <= maxu; u++)
        {
            for (int v = minv; v <= maxv; v++)
            {
                area += ratio_inside_pixel(s0, s1, s2, s3, n01, n12, n23, n30, vec2<T>(u, v));
            }
        }

        if (area <= 0)
        {
            return;
        }

        for (int u = minu; u <= maxu; u++)
        {
            for (int v = minv; v <= maxv; v++)
            {
                gpuAtomicAdd(&v_textures[g][u][v][k], delta * ratio_inside_pixel(s0, s1, s2, s3, n01, n12, n23, n30, vec2<T>(u, v)) / area);
            }
        }
        return;
    }
}

#endif // GSPLAT_CUDA_ANISOTROPIC_FILTER_H