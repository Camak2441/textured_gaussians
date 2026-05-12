
#ifndef GSPLAT_CUDA_ANISOTROPIC_BILINEAR_FILTER_H
#define GSPLAT_CUDA_ANISOTROPIC_BILINEAR_FILTER_H

#include "../helpers.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/TensorAccessor.h>

#define FILTER_INV_SQUARE 2.0f

namespace gsplat::anisotropic_bilinear
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
            *n01 = s0 - center;
        else
            *n01 = vec2<T>(s01.y, -s01.x);
        T l = glm::length(*n01);
        if (l == 0) return false;
        *n01 /= l;
        return true;
    }

    template <typename T> inline __device__ T max4(T v0, T v1, T v2, T v3) { return max(max(v0,v1),max(v2,v3)); }
    template <typename T> inline __device__ T min4(T v0, T v1, T v2, T v3) { return min(min(v0,v1),min(v2,v3)); }

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
        vec2<T> s_center = uv + vec2<T>(T(0.5), T(0.5));
        if (glm::dot(s0 - s_center, n01) >= n01max &&
            glm::dot(s1 - s_center, n12) >= n12max &&
            glm::dot(s2 - s_center, n23) >= n23max &&
            glm::dot(s3 - s_center, n30) >= n30max)
        {
            *A = T(1); *Sx = T(0.5); *Sy = T(0.5); *Sxy = T(0.25);
            return true;
        }

        vec2<T> poly[8], tmp[8];
        int n = 4;
        poly[0] = s0 - uv; poly[1] = s1 - uv; poly[2] = s2 - uv; poly[3] = s3 - uv;

        // Clip x >= 0
        int m = 0;
        for (int i = 0; i < n; i++) {
            vec2<T> a = poly[i], b = poly[(i+1)%n];
            bool ai = (a.x >= T(0)), bi = (b.x >= T(0));
            if (ai) tmp[m++] = a;
            if (ai != bi) { T t = a.x/(a.x-b.x); tmp[m++] = vec2<T>(T(0), a.y+t*(b.y-a.y)); }
        }
        n = m; if (n < 3) return false;

        // Clip x <= 1
        m = 0;
        for (int i = 0; i < n; i++) {
            vec2<T> a = tmp[i], b = tmp[(i+1)%n];
            bool ai = (a.x <= T(1)), bi = (b.x <= T(1));
            if (ai) poly[m++] = a;
            if (ai != bi) { T t = (a.x-T(1))/(a.x-b.x); poly[m++] = vec2<T>(T(1), a.y+t*(b.y-a.y)); }
        }
        n = m; if (n < 3) return false;

        // Clip y >= 0
        m = 0;
        for (int i = 0; i < n; i++) {
            vec2<T> a = poly[i], b = poly[(i+1)%n];
            bool ai = (a.y >= T(0)), bi = (b.y >= T(0));
            if (ai) tmp[m++] = a;
            if (ai != bi) { T t = a.y/(a.y-b.y); tmp[m++] = vec2<T>(a.x+t*(b.x-a.x), T(0)); }
        }
        n = m; if (n < 3) return false;

        // Clip y <= 1
        m = 0;
        for (int i = 0; i < n; i++) {
            vec2<T> a = tmp[i], b = tmp[(i+1)%n];
            bool ai = (a.y <= T(1)), bi = (b.y <= T(1));
            if (ai) poly[m++] = a;
            if (ai != bi) { T t = (a.y-T(1))/(a.y-b.y); poly[m++] = vec2<T>(a.x+t*(b.x-a.x), T(1)); }
        }
        n = m; if (n < 3) return false;

        *A = T(0); *Sx = T(0); *Sy = T(0); *Sxy = T(0);
        for (int i = 0; i < n; i++) {
            vec2<T> a = poly[i], b = poly[(i+1)%n];
            T c = cross2d(a, b);
            *A += c;
            *Sx += c * (a.x + b.x);
            *Sy += c * (a.y + b.y);
            *Sxy += c * (T(2)*a.x*a.y + a.x*b.y + b.x*a.y + T(2)*b.x*b.y);
        }
        *A *= T(0.5); *Sx *= T(1)/T(6); *Sy *= T(1)/T(6); *Sxy *= T(1)/T(24);
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
        vec2<T> s01 = *s1 - *s0, s12 = *s2 - *s1, s23 = *s3 - *s2, s30 = *s0 - *s3;
        T area = T(0.5) * (cross2d(s01, s12) + cross2d(s23, s30));
        if (area < 0) {
            area *= -1;
            vec2<T> temp = *s1; *s1 = *s3; *s3 = temp;
            s01 = *s1-*s0; s12 = *s2-*s1; s23 = *s3-*s2; s30 = *s0-*s3;
        }

        *minu = max(0, (int32_t)floor(min4(s0->x, s1->x, s2->x, s3->x)));
        *maxu = min(texture_res_x-1, (int32_t)ceil(max4(s0->x, s1->x, s2->x, s3->x)));
        *minv = max(0, (int32_t)floor(min4(s0->y, s1->y, s2->y, s3->y)));
        *maxv = min(texture_res_y-1, (int32_t)ceil(max4(s0->y, s1->y, s2->y, s3->y)));

        vec2<T> center = (*s0 + *s1 + *s2 + *s3) / T(4);
        edge_normal(*s0, s01, center, n01);
        edge_normal(*s1, s12, center, n12);
        edge_normal(*s2, s23, center, n23);
        edge_normal(*s3, s30, center, n30);

        *n01max = max(abs(glm::dot(*n01, vec2<T>(T(0.5), T( 0.5)))), abs(glm::dot(*n01, vec2<T>(T(0.5), T(-0.5)))));
        *n12max = max(abs(glm::dot(*n12, vec2<T>(T(0.5), T( 0.5)))), abs(glm::dot(*n12, vec2<T>(T(0.5), T(-0.5)))));
        *n23max = max(abs(glm::dot(*n23, vec2<T>(T(0.5), T( 0.5)))), abs(glm::dot(*n23, vec2<T>(T(0.5), T(-0.5)))));
        *n30max = max(abs(glm::dot(*n30, vec2<T>(T(0.5), T( 0.5)))), abs(glm::dot(*n30, vec2<T>(T(0.5), T(-0.5)))));
        return area;
    }

    // Returns [x_lo, x_hi] = the x-extent of the convex polygon s0..s3 within
    // the horizontal strip [v, v+1].  If the polygon misses the strip entirely,
    // x_lo > x_hi (empty interval).
    // Cost: 4 vertices × (1 vertex-in-strip check + 2 edge-crossing checks) = O(12).
    template <typename T>
    inline __device__ void strip_x_extent(
        vec2<T> s0, vec2<T> s1, vec2<T> s2, vec2<T> s3,
        int v, T *x_lo, T *x_hi)
    {
        const T y0 = T(v), y1 = T(v + 1);
        T lo = T(1e30f), hi = T(-1e30f);
        const vec2<T> verts[4] = {s0, s1, s2, s3};

        GSPLAT_PRAGMA_UNROLL
        for (int i = 0; i < 4; i++)
        {
            const vec2<T> a = verts[i], b = verts[(i + 1) & 3];

            // Vertex a inside strip
            if (a.y >= y0 && a.y <= y1)
            {
                lo = min(lo, a.x); hi = max(hi, a.x);
            }

            // Edge (a,b) crossings at y = y0 and y = y1
            if (a.y != b.y)
            {
                const T inv_dy = T(1) / (b.y - a.y);
                const T t0 = (y0 - a.y) * inv_dy;
                if (t0 >= T(0) && t0 <= T(1))
                {
                    const T x = a.x + t0 * (b.x - a.x);
                    lo = min(lo, x); hi = max(hi, x);
                }
                const T t1 = (y1 - a.y) * inv_dy;
                if (t1 >= T(0) && t1 <= T(1))
                {
                    const T x = a.x + t1 * (b.x - a.x);
                    lo = min(lo, x); hi = max(hi, x);
                }
            }
        }
        *x_lo = lo; *x_hi = hi;
    }

    // Compute the bilinear tent weight for texel (tu,tv) from four surrounding
    // unit-quadrant clips.  The strip x-extents (lo_bot/hi_bot for [tv-1,tv] and
    // lo_top/hi_top for [tv,tv+1]) gate each quadrant clip: if the quadrant cell's
    // x-interval does not intersect the polygon's x-range in that strip the clip is
    // skipped entirely, saving the Sutherland-Hodgman work.
    //
    // Quadrant → strip → cell x-range → cell overlap condition
    //   Q++ [tu,tu+1]×[tv,tv+1]   top  tu   < hi_top && tu+1 > lo_top
    //   Q-+ [tu-1,tu]×[tv,tv+1]   top  tu-1 < hi_top && tu   > lo_top
    //   Q+- [tu,tu+1]×[tv-1,tv]   bot  tu   < hi_bot && tu+1 > lo_bot
    //   Q-- [tu-1,tu]×[tv-1,tv]   bot  tu-1 < hi_bot && tu   > lo_bot
    template <typename T>
    inline __device__ T tent_weight(
        vec2<T> s0, vec2<T> s1, vec2<T> s2, vec2<T> s3,
        vec2<T> n01, vec2<T> n12, vec2<T> n23, vec2<T> n30,
        T n01max, T n12max, T n23max, T n30max,
        int tu, int tv,
        T lo_bot, T hi_bot, T lo_top, T hi_top)
    {
        T W = T(0);
        T A, Sx, Sy, Sxy;
        const T ftu = T(tu), ftu1 = T(tu + 1), ftu_1 = T(tu - 1);
        const T ftv = T(tv), ftv_1 = T(tv - 1);

        if (ftu < hi_top && ftu1 > lo_top)
            if (clip_and_compute_moments(s0, s1, s2, s3, n01, n12, n23, n30,
                    n01max, n12max, n23max, n30max, vec2<T>(ftu, ftv), &A, &Sx, &Sy, &Sxy))
                W += (A - Sx - Sy + Sxy);

        if (ftu_1 < hi_top && ftu > lo_top)
            if (clip_and_compute_moments(s0, s1, s2, s3, n01, n12, n23, n30,
                    n01max, n12max, n23max, n30max, vec2<T>(ftu_1, ftv), &A, &Sx, &Sy, &Sxy))
                W += (Sx - Sxy);

        if (ftu < hi_bot && ftu1 > lo_bot)
            if (clip_and_compute_moments(s0, s1, s2, s3, n01, n12, n23, n30,
                    n01max, n12max, n23max, n30max, vec2<T>(ftu, ftv_1), &A, &Sx, &Sy, &Sxy))
                W += (Sy - Sxy);

        if (ftu_1 < hi_bot && ftu > lo_bot)
            if (clip_and_compute_moments(s0, s1, s2, s3, n01, n12, n23, n30,
                    n01max, n12max, n23max, n30max, vec2<T>(ftu_1, ftv_1), &A, &Sx, &Sy, &Sxy))
                W += Sxy;

        return W;
    }

    // Compute tight [tu_start, tu_end] for one row, given rolling strip x-extents.
    // Clamped before float->int conversion to avoid undefined overflow behaviour.
    template <typename T>
    inline __device__ void tu_range(
        int minu, int maxu,
        T lo_bot, T hi_bot, T lo_top, T hi_top,
        int *tu_start, int *tu_end)
    {
        const T fmin = T(minu - 2), fmax = T(maxu + 2);
        const T clo = max(fmin, min(fmax, min(lo_bot, lo_top)));
        const T chi = max(fmin, min(fmax, max(hi_bot, hi_top)));
        *tu_start = max(minu, (int)floor(clo));
        *tu_end   = min(maxu, (int)ceil (chi));
    }

    // ---- Sampling functions ------------------------------------------------
    // All five functions share the same rolling-strip loop structure:
    //   1. Initialise bot/top strips for the first row.
    //   2. Per row: compute tight tu range, iterate, slide the strip window.

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
        T lo_bot, hi_bot, lo_top, hi_top;
        strip_x_extent(s0, s1, s2, s3, minv - 1, &lo_bot, &hi_bot);
        strip_x_extent(s0, s1, s2, s3, minv,     &lo_top, &hi_top);

        for (int tv = minv; tv <= maxv; tv++)
        {
            int tu_start, tu_end;
            tu_range(minu, maxu, lo_bot, hi_bot, lo_top, hi_top, &tu_start, &tu_end);

            for (int tu = tu_start; tu <= tu_end; tu++)
            {
                T W = tent_weight(s0, s1, s2, s3, n01, n12, n23, n30,
                                  n01max, n12max, n23max, n30max,
                                  tu, tv, lo_bot, hi_bot, lo_top, hi_top);
                if (W == T(0)) continue;
                value += textures[g][tv][tu][k] * W;
            }

            lo_bot = lo_top; hi_bot = hi_top;
            strip_x_extent(s0, s1, s2, s3, tv + 1, &lo_top, &hi_top);
        }
        return value * iarea;
    }

    template <uint32_t COLOR_DIM, typename T>
    inline __device__ void color_sample(
        at::PackedTensorAccessor32<const T, 4, at::RestrictPtrTraits> textures,
        int32_t g,
        vec2<T> s0, vec2<T> s1, vec2<T> s2, vec2<T> s3,
        vec2<T> n01, vec2<T> n12, vec2<T> n23, vec2<T> n30,
        T n01max, T n12max, T n23max, T n30max,
        int32_t minu, int32_t maxu, int32_t minv, int32_t maxv,
        T area, T iarea,
        int texture_res_x, int texture_res_y,
        T col[COLOR_DIM])
    {
        T lo_bot, hi_bot, lo_top, hi_top;
        strip_x_extent(s0, s1, s2, s3, minv - 1, &lo_bot, &hi_bot);
        strip_x_extent(s0, s1, s2, s3, minv,     &lo_top, &hi_top);

        for (int tv = minv; tv <= maxv; tv++)
        {
            int tu_start, tu_end;
            tu_range(minu, maxu, lo_bot, hi_bot, lo_top, hi_top, &tu_start, &tu_end);

            for (int tu = tu_start; tu <= tu_end; tu++)
            {
                T W = tent_weight(s0, s1, s2, s3, n01, n12, n23, n30,
                                  n01max, n12max, n23max, n30max,
                                  tu, tv, lo_bot, hi_bot, lo_top, hi_top);
                if (W == T(0)) continue;
                T wi = W * iarea;
                GSPLAT_PRAGMA_UNROLL
                for (int k = 0; k < COLOR_DIM; ++k)
                    col[k] += textures[g][tv][tu][k] * wi;
            }

            lo_bot = lo_top; hi_bot = hi_top;
            strip_x_extent(s0, s1, s2, s3, tv + 1, &lo_top, &hi_top);
        }
    }

    template <uint32_t COLOR_DIM, typename T>
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
        const int alpha_k = textures.size(3) - 1;
        T lo_bot, hi_bot, lo_top, hi_top;
        strip_x_extent(s0, s1, s2, s3, minv - 1, &lo_bot, &hi_bot);
        strip_x_extent(s0, s1, s2, s3, minv,     &lo_top, &hi_top);

        for (int tv = minv; tv <= maxv; tv++)
        {
            int tu_start, tu_end;
            tu_range(minu, maxu, lo_bot, hi_bot, lo_top, hi_top, &tu_start, &tu_end);

            for (int tu = tu_start; tu <= tu_end; tu++)
            {
                T W = tent_weight(s0, s1, s2, s3, n01, n12, n23, n30,
                                  n01max, n12max, n23max, n30max,
                                  tu, tv, lo_bot, hi_bot, lo_top, hi_top);
                if (W == T(0)) continue;
                T wi = W * iarea;
                GSPLAT_PRAGMA_UNROLL
                for (int k = 0; k < COLOR_DIM; ++k)
                    col[k] += textures[g][tv][tu][k] * wi;
                *alpha += textures[g][tv][tu][alpha_k] * wi;
            }

            lo_bot = lo_top; hi_bot = hi_top;
            strip_x_extent(s0, s1, s2, s3, tv + 1, &lo_top, &hi_top);
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
        T lo_bot, hi_bot, lo_top, hi_top;
        strip_x_extent(s0, s1, s2, s3, minv - 1, &lo_bot, &hi_bot);
        strip_x_extent(s0, s1, s2, s3, minv,     &lo_top, &hi_top);

        for (int tv = minv; tv <= maxv; tv++)
        {
            int tu_start, tu_end;
            tu_range(minu, maxu, lo_bot, hi_bot, lo_top, hi_top, &tu_start, &tu_end);

            for (int tu = tu_start; tu <= tu_end; tu++)
            {
                T W = tent_weight(s0, s1, s2, s3, n01, n12, n23, n30,
                                  n01max, n12max, n23max, n30max,
                                  tu, tv, lo_bot, hi_bot, lo_top, hi_top);
                if (W == T(0)) continue;
                gpuAtomicAdd(&v_textures[g][tv][tu][k], ndelta * W);
            }

            lo_bot = lo_top; hi_bot = hi_top;
            strip_x_extent(s0, s1, s2, s3, tv + 1, &lo_top, &hi_top);
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
            ndeltas[k] = deltas[k] * iarea;

        T lo_bot, hi_bot, lo_top, hi_top;
        strip_x_extent(s0, s1, s2, s3, minv - 1, &lo_bot, &hi_bot);
        strip_x_extent(s0, s1, s2, s3, minv,     &lo_top, &hi_top);

        for (int tv = minv; tv <= maxv; tv++)
        {
            int tu_start, tu_end;
            tu_range(minu, maxu, lo_bot, hi_bot, lo_top, hi_top, &tu_start, &tu_end);

            for (int tu = tu_start; tu <= tu_end; tu++)
            {
                T W = tent_weight(s0, s1, s2, s3, n01, n12, n23, n30,
                                  n01max, n12max, n23max, n30max,
                                  tu, tv, lo_bot, hi_bot, lo_top, hi_top);
                if (W == T(0)) continue;
                T wi = W * iarea;
                GSPLAT_PRAGMA_UNROLL
                for (int k = 0; k < COLOR_DIM; ++k)
                {
                    col[k] += textures[g][tv][tu][k] * wi;
                    gpuAtomicAdd(&v_textures[g][tv][tu][k], ndeltas[k] * W);
                }
            }

            lo_bot = lo_top; hi_bot = hi_top;
            strip_x_extent(s0, s1, s2, s3, tv + 1, &lo_top, &hi_top);
        }
    }

} // namespace gsplat::anisotropic_bilinear

#endif // GSPLAT_CUDA_ANISOTROPIC_BILINEAR_FILTER_H
