
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
    inline __device__ T max4(T v0, T v1, T v2, T v3) { return max(max(v0, v1), max(v2, v3)); }
    template <typename T>
    inline __device__ T min4(T v0, T v1, T v2, T v3) { return min(min(v0, v1), min(v2, v3)); }

    template <typename T>
    inline __device__ T precompute(
        vec2<T> *s0, vec2<T> *s1, vec2<T> *s2, vec2<T> *s3,
        int32_t *minu, int32_t *minv, int32_t *maxu, int32_t *maxv,
        int32_t *minv_full, int32_t *maxv_full,
        int texture_res_x, int texture_res_y)
    {
        const vec2<T> s01 = *s1 - *s0, s12 = *s2 - *s1, s23 = *s3 - *s2, s30 = *s0 - *s3;
        T area = T(0.5) * (cross2d(s01, s12) + cross2d(s23, s30));
        if (area < 0)
        {
            area *= -1;
            vec2<T> t = *s1;
            *s1 = *s3;
            *s3 = t;
        }
        *minu = max(0, (int32_t)floor(min4(s0->x, s1->x, s2->x, s3->x)));
        *maxu = min(texture_res_x - 1, (int32_t)ceil(max4(s0->x, s1->x, s2->x, s3->x)));
        *minv_full = (int32_t)floor(min4(s0->y, s1->y, s2->y, s3->y));
        *maxv_full = (int32_t)ceil(max4(s0->y, s1->y, s2->y, s3->y));
        *minv = max(0, *minv_full);
        *maxv = min(texture_res_y - 1, *maxv_full);
        return area;
    }

    // Area and v-moment of polygon row ∩ {u < 0}, given left edge varies linearly
    // from xlo_bot (at t=0) to xlo_top (at t=1).
    template <typename T>
    inline __device__ void left_overflow_moments(T xlo_bot, T xlo_top, T *A, T *Sy)
    {
        *A = T(0);
        *Sy = T(0);
        if (xlo_bot >= T(0) && xlo_top >= T(0))
            return;
        T t0 = T(0), t1 = T(1), w0 = -xlo_bot, w1 = -xlo_top;
        if (xlo_bot >= T(0)) { t0 = xlo_bot / (xlo_bot - xlo_top); w0 = T(0); }
        else if (xlo_top >= T(0)) { t1 = xlo_bot / (xlo_bot - xlo_top); w1 = T(0); }
        const T dt = t1 - t0;
        *A = dt * (w0 + w1) * T(0.5);
        *Sy = t0 * (*A) + dt * dt * (w0 * T(1.0f / 6) + w1 * T(1.0f / 3));
    }

    // Area and v-moment of polygon row ∩ {u > max_u}, right edge xhi_bot→xhi_top.
    template <typename T>
    inline __device__ void right_overflow_moments(T xhi_bot, T xhi_top, T max_u, T *A, T *Sy)
    {
        left_overflow_moments(-xhi_bot + max_u, -xhi_top + max_u, A, Sy);
    }

    template <typename T>
    inline __device__ void scan_active_edges(
        vec2<T> s0, vec2<T> s1, vec2<T> s2, vec2<T> s3, T y0,
        T *xlo, T *xhi, T *slo, T *shi)
    {
        const vec2<T> verts[4] = {s0, s1, s2, s3};
        T ex[2];
        T es[2];
        int cnt = 0;
        *xlo = T(1e30f);
        *xhi = T(-1e30f);
        *slo = T(0);
        *shi = T(0);
        GSPLAT_PRAGMA_UNROLL
        for (int i = 0; i < 4; i++)
        {
            if (cnt >= 2)
                break;
            const vec2<T> a = verts[i], b = verts[(i + 1) & 3];
            if (a.y == b.y)
                continue;
            const T yl = min(a.y, b.y), yh = max(a.y, b.y);
            if (y0 < yl || y0 > yh)
                continue;
            const T slope = (b.x - a.x) / (b.y - a.y);
            ex[cnt] = a.x + (y0 - a.y) * slope;
            es[cnt] = slope;
            cnt++;
        }
        if (cnt == 1)
        {
            *xlo = *xhi = ex[0];
            *slo = *shi = es[0];
        }
        else if (cnt == 2)
        {
            if (ex[0] <= ex[1])
            {
                *xlo = ex[0];
                *xhi = ex[1];
                *slo = es[0];
                *shi = es[1];
            }
            else
            {
                *xlo = ex[1];
                *xhi = ex[0];
                *slo = es[1];
                *shi = es[0];
            }
        }
    }

    template <typename T>
    inline __device__ void advance_scanline(
        vec2<T> s0, vec2<T> s1, vec2<T> s2, vec2<T> s3,
        T y_curr, T y_next,
        T xlo_in, T xhi_in, T slo_in, T shi_in,
        T *xlo_out, T *xhi_out, T *slo_out, T *shi_out,
        int *nv_out)
    {
        bool rescan = false;
        *nv_out = 0;
        {
            const T _vy = s0.y;
            if (_vy > y_curr && _vy <= y_next)
            {
                rescan = true;
                if (_vy < y_next)
                    (*nv_out)++;
            }
        }
        {
            const T _vy = s1.y;
            if (_vy > y_curr && _vy <= y_next)
            {
                rescan = true;
                if (_vy < y_next)
                    (*nv_out)++;
            }
        }
        {
            const T _vy = s2.y;
            if (_vy > y_curr && _vy <= y_next)
            {
                rescan = true;
                if (_vy < y_next)
                    (*nv_out)++;
            }
        }
        {
            const T _vy = s3.y;
            if (_vy > y_curr && _vy <= y_next)
            {
                rescan = true;
                if (_vy < y_next)
                    (*nv_out)++;
            }
        }
        if (rescan)
            scan_active_edges(s0, s1, s2, s3, y_next, xlo_out, xhi_out, slo_out, shi_out);
        else
        {
            *xlo_out = xlo_in + slo_in;
            *xhi_out = xhi_in + shi_in;
            *slo_out = slo_in;
            *shi_out = shi_in;
        }
    }

    // ---- Analytical moment integration ----------------------------------------

    template <typename T>
    inline __device__ void trapezoid_moments(
        T al, T ar, T bl, T br,
        T *A, T *Sx, T *Sy, T *Sxy)
    {
        if (al <= T(0) && ar >= T(1) && bl <= T(0) && br >= T(1))
        {
            *A = T(1);
            *Sx = T(0.5);
            *Sy = T(0.5);
            *Sxy = T(0.25);
            return;
        }
        if (al >= T(0) && ar <= T(1) && bl >= T(0) && br <= T(1))
        {
            const T qa = bl - al, qb = br - ar, C = ar - al, D = qb - qa;
            *A = C + D * T(0.5);
            *Sy = C * T(0.5) + D * (T(1) / T(3));
            const T E = ar * ar - al * al, F = T(2) * (ar * qb - al * qa), G = qb * qb - qa * qa;
            *Sx = (E + F * T(0.5) + G * (T(1) / T(3))) * T(0.5);
            *Sxy = (E * T(0.5) + F * (T(1) / T(3)) + G * T(0.25)) * T(0.5);
            return;
        }
        *A = T(0);
        *Sx = T(0);
        *Sy = T(0);
        *Sxy = T(0);
        const T dxl = bl - al, dxr = br - ar;
        T bp[7];
        int nb = 2;
        bp[0] = T(0);
        bp[1] = T(1);
#define TM_ADD(t_)                  \
    do                              \
    {                               \
        T _t = (t_);                \
        if (_t > T(0) && _t < T(1)) \
            bp[nb++] = _t;          \
    } while (0)
        if (dxl != T(0))
        {
            TM_ADD(-al / dxl);
            TM_ADD((T(1) - al) / dxl);
        }
        if (dxr != T(0))
        {
            TM_ADD(-ar / dxr);
            TM_ADD((T(1) - ar) / dxr);
        }
        {
            T den = dxl - dxr;
            if (den != T(0))
                TM_ADD((ar - al) / den);
        }
#undef TM_ADD
        for (int i = 1; i < nb; i++)
        {
            T key = bp[i];
            int j = i - 1;
            while (j >= 0 && bp[j] > key)
            {
                bp[j + 1] = bp[j];
                j--;
            }
            bp[j + 1] = key;
        }
        for (int i = 0; i < nb - 1; i++)
        {
            const T t0 = bp[i], t1 = bp[i + 1], dt = t1 - t0;
            if (dt <= T(0))
                continue;
            const T la = min(max(al + dxl * t0, T(0)), T(1));
            const T lb = min(max(al + dxl * t1, T(0)), T(1));
            const T ra = min(max(ar + dxr * t0, T(0)), T(1));
            const T rb = min(max(ar + dxr * t1, T(0)), T(1));
            const T C = ra - la;
            if (C <= T(0) && (rb - lb) <= T(0))
                continue;
            const T qa = (lb - la) / dt, qb = (rb - ra) / dt, D = qb - qa;
            const T dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt3 * dt;
            const T dA = C * dt + D * dt2 * T(0.5);
            *A += dA;
            *Sy += t0 * dA + C * dt2 * T(0.5) + D * dt3 * (T(1) / T(3));
            const T E = ra * ra - la * la, F = T(2) * (ra * qb - la * qa), G = qb * qb - qa * qa;
            const T dSx = (E * dt + F * dt2 * T(0.5) + G * dt3 * (T(1) / T(3))) * T(0.5);
            *Sx += dSx;
            *Sxy += t0 * dSx + (E * dt2 * T(0.5) + F * dt3 * (T(1) / T(3)) + G * dt4 * T(0.25)) * T(0.5);
        }
    }

    template <typename T>
    inline __device__ void trapezoid_sx_sxy(T al, T ar, T bl, T br, T *Sx, T *Sxy)
    {
        if (al <= T(0) && ar >= T(1) && bl <= T(0) && br >= T(1))
        {
            *Sx = T(0.5);
            *Sxy = T(0.25);
            return;
        }
        if (al >= T(0) && ar <= T(1) && bl >= T(0) && br <= T(1))
        {
            const T qa = bl - al, qb = br - ar;
            const T E = ar * ar - al * al, F = T(2) * (ar * qb - al * qa), G = qb * qb - qa * qa;
            *Sx = (E + F * T(0.5) + G * (T(1) / T(3))) * T(0.5);
            *Sxy = (E * T(0.5) + F * (T(1) / T(3)) + G * T(0.25)) * T(0.5);
            return;
        }
        *Sx = T(0);
        *Sxy = T(0);
        const T dxl = bl - al, dxr = br - ar;
        T bp[7];
        int nb = 2;
        bp[0] = T(0);
        bp[1] = T(1);
#define TSX_ADD(t_)                 \
    do                              \
    {                               \
        T _t = (t_);                \
        if (_t > T(0) && _t < T(1)) \
            bp[nb++] = _t;          \
    } while (0)
        if (dxl != T(0))
        {
            TSX_ADD(-al / dxl);
            TSX_ADD((T(1) - al) / dxl);
        }
        if (dxr != T(0))
        {
            TSX_ADD(-ar / dxr);
            TSX_ADD((T(1) - ar) / dxr);
        }
        {
            T den = dxl - dxr;
            if (den != T(0))
                TSX_ADD((ar - al) / den);
        }
#undef TSX_ADD
        for (int i = 1; i < nb; i++)
        {
            T key = bp[i];
            int j = i - 1;
            while (j >= 0 && bp[j] > key)
            {
                bp[j + 1] = bp[j];
                j--;
            }
            bp[j + 1] = key;
        }
        for (int i = 0; i < nb - 1; i++)
        {
            const T t0 = bp[i], t1 = bp[i + 1], dt = t1 - t0;
            if (dt <= T(0))
                continue;
            const T la = min(max(al + dxl * t0, T(0)), T(1));
            const T lb = min(max(al + dxl * t1, T(0)), T(1));
            const T ra = min(max(ar + dxr * t0, T(0)), T(1));
            const T rb = min(max(ar + dxr * t1, T(0)), T(1));
            const T C = ra - la;
            if (C <= T(0) && (rb - lb) <= T(0))
                continue;
            const T qa = (lb - la) / dt, qb = (rb - ra) / dt;
            const T dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt3 * dt;
            const T E = ra * ra - la * la, F = T(2) * (ra * qb - la * qa), G = qb * qb - qa * qa;
            const T dSx = (E * dt + F * dt2 * T(0.5) + G * dt3 * (T(1) / T(3))) * T(0.5);
            *Sx += dSx;
            *Sxy += t0 * dSx + (E * dt2 * T(0.5) + F * dt3 * (T(1) / T(3)) + G * dt4 * T(0.25)) * T(0.5);
        }
    }

    template <typename T>
    inline __device__ void accumulate_strip(
        T al, T ar, T bl, T br, T t_lo, T t_hi,
        T *A, T *Sx, T *Sy, T *Sxy)
    {
        T An, Sxn, Syn, Sxyn;
        trapezoid_moments(al, ar, bl, br, &An, &Sxn, &Syn, &Sxyn);
        const T dt = t_hi - t_lo;
        *A += dt * An;
        *Sx += dt * Sxn;
        *Sy += dt * (t_lo * An + dt * Syn);
        *Sxy += dt * (t_lo * Sxn + dt * Sxyn);
    }

    template <typename T>
    inline __device__ void polygon_x_at_y(
        vec2<T> s0, vec2<T> s1, vec2<T> s2, vec2<T> s3,
        T y, T *xl, T *xr)
    {
        *xl = T(1e30f);
        *xr = T(-1e30f);
        const vec2<T> v[4] = {s0, s1, s2, s3};
        GSPLAT_PRAGMA_UNROLL
        for (int i = 0; i < 4; i++)
        {
            const vec2<T> a = v[i], b = v[(i + 1) & 3];
            if (a.y == b.y)
                continue;
            const T yl = min(a.y, b.y), yh = max(a.y, b.y);
            if (y < yl || y > yh)
                continue;
            const T x = a.x + (y - a.y) / (b.y - a.y) * (b.x - a.x);
            *xl = min(*xl, x);
            *xr = max(*xr, x);
        }
    }

    template <typename T>
    inline __device__ void single_cell_moments(
        vec2<T> s0, vec2<T> s1, vec2<T> s2, vec2<T> s3,
        T y_lo, T y_hi, T xu,
        T xl_lo, T xr_lo, T xl_hi, T xr_hi,
        T *A, T *Sx, T *Sy, T *Sxy)
    {
        if (xl_lo > xr_lo)
            polygon_x_at_y(s0, s1, s2, s3, y_lo, &xl_lo, &xr_lo);
        if (xl_hi > xr_hi)
            polygon_x_at_y(s0, s1, s2, s3, y_hi, &xl_hi, &xr_hi);
        T vy[4];
        int nv = 0;
        const vec2<T> verts[4] = {s0, s1, s2, s3};
        GSPLAT_PRAGMA_UNROLL
        for (int i = 0; i < 4; i++)
        {
            const T y = verts[i].y;
            if (y > y_lo && y < y_hi)
            {
                int j = nv;
                while (j > 0 && vy[j - 1] > y)
                {
                    vy[j] = vy[j - 1];
                    j--;
                }
                vy[j] = y;
                nv++;
            }
        }
        if (nv == 0)
        {
            if (!(xl_lo > xr_lo) && !(xl_hi > xr_hi))
                trapezoid_moments(xl_lo - xu, xr_lo - xu, xl_hi - xu, xr_hi - xu, A, Sx, Sy, Sxy);
            else
                *A = *Sx = *Sy = *Sxy = T(0);
            return;
        }
        *A = *Sx = *Sy = *Sxy = T(0);
        T ypts[6];
        int ny = 0;
        ypts[ny++] = y_lo;
        for (int i = 0; i < nv; i++)
            ypts[ny++] = vy[i];
        ypts[ny++] = y_hi;
        T xlo_a = xl_lo, xhi_a = xr_lo;
        for (int i = 0; i < ny - 1; i++)
        {
            T xlo_b, xhi_b;
            if (i == ny - 2)
            {
                xlo_b = xl_hi;
                xhi_b = xr_hi;
            }
            else
                polygon_x_at_y(s0, s1, s2, s3, ypts[i + 1], &xlo_b, &xhi_b);
            if (!(xlo_a > xhi_a) && !(xlo_b > xhi_b))
            {
                const T t_lo = ypts[i] - y_lo, t_hi = ypts[i + 1] - y_lo;
                accumulate_strip(xlo_a - xu, xhi_a - xu, xlo_b - xu, xhi_b - xu, t_lo, t_hi, A, Sx, Sy, Sxy);
            }
            xlo_a = xlo_b;
            xhi_a = xhi_b;
        }
    }

    template <typename T>
    inline __device__ void single_cell_moments_simple(
        T xl_lo, T xr_lo, T xl_hi, T xr_hi, T xu,
        T *A, T *Sx, T *Sy, T *Sxy)
    {
        if (xl_lo <= xr_lo && xl_hi <= xr_hi)
            trapezoid_moments(xl_lo - xu, xr_lo - xu, xl_hi - xu, xr_hi - xu, A, Sx, Sy, Sxy);
        else
            *A = *Sx = *Sy = *Sxy = T(0);
    }

    template <typename T>
    inline __device__ void single_cell_sx_sxy_simple(
        T xl_lo, T xr_lo, T xl_hi, T xr_hi, T xu,
        T *Sx, T *Sxy)
    {
        if (xl_lo <= xr_lo && xl_hi <= xr_hi)
            trapezoid_sx_sxy(xl_lo - xu, xr_lo - xu, xl_hi - xu, xr_hi - xu, Sx, Sxy);
        else
            *Sx = *Sxy = T(0);
    }

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
        *tu_end = min(maxu, (int)ceil(chi));
    }

#define AB2_INIT_SCANLINE()                                                                            \
    T _row_Delta_s[33], _row_Sxyb_s[33];                                                               \
    GSPLAT_PRAGMA_UNROLL                                                                               \
    for (int _i = 0; _i < 33; _i++)                                                                    \
        _row_Delta_s[_i] = _row_Sxyb_s[_i] = T(0);                                                    \
    T *const _row_Delta = _row_Delta_s + 1, *const _row_Sxyb = _row_Sxyb_s + 1;                       \
    T _rb_extra = T(0);                                                                                \
    T _left_carry = T(0), _right_carry = T(0);                                                         \
    /* Bottom-overflow pre-pass uses a SEPARATE scanline so the main-loop state stays clean */         \
    if (minv_full < minv)                                                                              \
    {                                                                                                  \
        T _xlo_p, _xhi_p, _slo_p, _shi_p;                                                             \
        scan_active_edges(s0, s1, s2, s3, T(minv_full - 1), &_xlo_p, &_xhi_p, &_slo_p, &_shi_p);      \
        for (int _tv_ovf = minv_full; _tv_ovf < minv; _tv_ovf++)                                      \
        {                                                                                              \
            T _xlo_n, _xhi_n, _slo_n, _shi_n;                                                         \
            int _nv_n;                                                                                 \
            advance_scanline(s0, s1, s2, s3, T(_tv_ovf), T(_tv_ovf + 1),                               \
                             _xlo_p, _xhi_p, _slo_p, _shi_p,                                           \
                             &_xlo_n, &_xhi_n, &_slo_n, &_shi_n, &_nv_n);                             \
            /* LB corner: left overflow area → _row_Sxyb_s[0], which is _pSxyb for texel (0,0) */     \
            T _dA_lb, _dSy_lb;                                                                         \
            left_overflow_moments(_xlo_p, _xlo_n, &_dA_lb, &_dSy_lb);                                  \
            _row_Sxyb_s[0] += _dA_lb;                                                                  \
            /* CB strip: accumulate per-column (A-Sx) into _row_Delta, Sx into _row_Sxyb */            \
            const int _tu_s_ovf = max(minu, (int)floorf((float)min(_xlo_p, _xlo_n)));                  \
            const int _tu_e_ovf = min(maxu, (int)ceilf((float)max(_xhi_p, _xhi_n)));                   \
            for (int _tu_ovf = _tu_s_ovf; _tu_ovf <= _tu_e_ovf; _tu_ovf++)                            \
            {                                                                                          \
                T _A_c, _Sx_c, _Sy_c, _Sxy_c;                                                         \
                if (_nv_n == 0)                                                                        \
                    single_cell_moments_simple(_xlo_p, _xhi_p, _xlo_n, _xhi_n,                         \
                                               T(_tu_ovf), &_A_c, &_Sx_c, &_Sy_c, &_Sxy_c);           \
                else                                                                                   \
                    single_cell_moments(s0, s1, s2, s3, T(_tv_ovf), T(_tv_ovf + 1), T(_tu_ovf),        \
                                        _xlo_p, _xhi_p, _xlo_n, _xhi_n,                               \
                                        &_A_c, &_Sx_c, &_Sy_c, &_Sxy_c);                              \
                _row_Delta[_tu_ovf - minu] += _A_c - _Sx_c;                                            \
                _row_Sxyb[_tu_ovf - minu] += _Sx_c;                                                    \
            }                                                                                          \
            /* RB corner: right overflow area + orphaned Sx of column maxu → _rb_extra */              \
            T _dA_rb, _dSy_rb;                                                                         \
            right_overflow_moments(_xhi_p, _xhi_n, T(maxu), &_dA_rb, &_dSy_rb);                        \
            _rb_extra += _dA_rb + _row_Sxyb[maxu - minu];                                              \
            _row_Sxyb[maxu - minu] = T(0);                                                             \
            _xlo_p = _xlo_n; _xhi_p = _xhi_n; _slo_p = _slo_n; _shi_p = _shi_n;                       \
        }                                                                                              \
    }                                                                                                  \
    /* Main-loop scanline: (re-)initialise independently from T(minv-1) */                             \
    T _xlo, _xhi, _slo, _shi;                                                                         \
    scan_active_edges(s0, s1, s2, s3, T(minv - 1), &_xlo, &_xhi, &_slo, &_shi);                       \
    T _lo_bot_carry, _hi_bot_carry;                                                                    \
    {                                                                                                  \
        T _xp = _xlo, _xhp = _xhi;                                                                    \
        int _pnv;                                                                                      \
        advance_scanline(s0, s1, s2, s3, T(minv - 1), T(minv),                                         \
                         _xlo, _xhi, _slo, _shi, &_xlo, &_xhi, &_slo, &_shi, &_pnv);                  \
        _lo_bot_carry = min(_xp, _xlo);                                                                \
        _hi_bot_carry = max(_xhp, _xhi);                                                               \
        if (_pnv != 0)                                                                                 \
        {                                                                                              \
            const T _fi_ = T(minv);                                                                    \
            if (s0.y > _fi_ - T(1) && s0.y < _fi_)                                                    \
            {                                                                                          \
                _lo_bot_carry = min(_lo_bot_carry, s0.x);                                              \
                _hi_bot_carry = max(_hi_bot_carry, s0.x);                                              \
            }                                                                                          \
            if (s1.y > _fi_ - T(1) && s1.y < _fi_)                                                    \
            {                                                                                          \
                _lo_bot_carry = min(_lo_bot_carry, s1.x);                                              \
                _hi_bot_carry = max(_hi_bot_carry, s1.x);                                              \
            }                                                                                          \
            if (s2.y > _fi_ - T(1) && s2.y < _fi_)                                                    \
            {                                                                                          \
                _lo_bot_carry = min(_lo_bot_carry, s2.x);                                              \
                _hi_bot_carry = max(_hi_bot_carry, s2.x);                                              \
            }                                                                                          \
            if (s3.y > _fi_ - T(1) && s3.y < _fi_)                                                    \
            {                                                                                          \
                _lo_bot_carry = min(_lo_bot_carry, s3.x);                                              \
                _hi_bot_carry = max(_hi_bot_carry, s3.x);                                              \
            }                                                                                          \
        }                                                                                              \
    }

#define AB2_ROW_EXTENTS(tv_)                                                    \
    T _xlo_next, _xhi_next, _slo_next, _shi_next;                               \
    int _nv_top;                                                                \
    advance_scanline(s0, s1, s2, s3, T(tv_), T((tv_) + 1),                      \
                     _xlo, _xhi, _slo, _shi,                                    \
                     &_xlo_next, &_xhi_next, &_slo_next, &_shi_next, &_nv_top); \
    T lo_bot = _lo_bot_carry, hi_bot = _hi_bot_carry;                           \
    T lo_top = min(_xlo, _xlo_next), hi_top = max(_xhi, _xhi_next);             \
    if (_nv_top != 0)                                                           \
    {                                                                           \
        const T _f_ = T(tv_);                                                   \
        if (s0.y > _f_ && s0.y < _f_ + T(1))                                    \
        {                                                                       \
            lo_top = min(lo_top, s0.x);                                         \
            hi_top = max(hi_top, s0.x);                                         \
        }                                                                       \
        if (s1.y > _f_ && s1.y < _f_ + T(1))                                    \
        {                                                                       \
            lo_top = min(lo_top, s1.x);                                         \
            hi_top = max(hi_top, s1.x);                                         \
        }                                                                       \
        if (s2.y > _f_ && s2.y < _f_ + T(1))                                    \
        {                                                                       \
            lo_top = min(lo_top, s2.x);                                         \
            hi_top = max(hi_top, s2.x);                                         \
        }                                                                       \
        if (s3.y > _f_ && s3.y < _f_ + T(1))                                    \
        {                                                                       \
            lo_top = min(lo_top, s3.x);                                         \
            hi_top = max(hi_top, s3.x);                                         \
        }                                                                       \
    }                                                                           \
    const T _ftv_ = T(tv_), _ftv1_ = _ftv_ + T(1);

#define AB2_SLIDE_WINDOW()  \
    _xlo = _xlo_next;       \
    _xhi = _xhi_next;       \
    _slo = _slo_next;       \
    _shi = _shi_next;       \
    _lo_bot_carry = lo_top; \
    _hi_bot_carry = hi_top;

// Per-row left/right overflow moments (call after AB2_ROW_EXTENTS).
// Produces _dA_L/_dSy_L (left strip) and _dA_R/_dSy_R (right strip).
#define AB2_ROW_OVERFLOW()                                                          \
    T _dA_L, _dSy_L, _dA_R, _dSy_R;                                                \
    left_overflow_moments(_xlo, _xlo_next, &_dA_L, &_dSy_L);                       \
    right_overflow_moments(_xhi, _xhi_next, T(maxu), &_dA_R, &_dSy_R);

// _pSxmSxy = Sx-Sxy of left-neighbour top cell (current row).
// _pSxyb   = Sxyt of left-neighbour bot cell (previous row), carried as column register.
#define AB2_COL_CACHE_INIT(tv_)                                                                 \
    T _pSxmSxy, _pSxyb;                                                                         \
    if (tu_s == minu)                                                                           \
    {                                                                                           \
        _pSxmSxy = T(0);                                                                        \
        _pSxyb = T(0);                                                                          \
    }                                                                                           \
    else                                                                                        \
    {                                                                                           \
        T _tsx, _tsxy;                                                                          \
        if (_nv_top == 0)                                                                       \
            single_cell_sx_sxy_simple(_xlo, _xhi, _xlo_next, _xhi_next,                         \
                                      T(tu_s - 1), &_tsx, &_tsxy);                              \
        else                                                                                    \
        {                                                                                       \
            T _dam, _Syt;                                                                       \
            single_cell_moments(s0, s1, s2, s3, _ftv_, _ftv1_, T(tu_s - 1),                     \
                                _xlo, _xhi, _xlo_next, _xhi_next, &_dam, &_tsx, &_Syt, &_tsxy); \
        }                                                                                       \
        _pSxmSxy = _tsx - _tsxy;                                                                \
        _pSxyb = _row_Sxyb[tu_s - 1 - minu];                                                    \
    }                                                                                           \
    const int _tfl = (int)ceilf((float)max(_xlo, _xlo_next));                                   \
    const int _tfh = (int)floorf((float)min(_xhi, _xhi_next)) - 1;

#define AB2_COMPUTE_W(tu_, tv_, W_)                                                            \
    {                                                                                          \
        const T _bd = _row_Delta[(tu_) - minu];                                                \
        const T _Sxyb_c = _row_Sxyb[(tu_) - minu];                                             \
        T _At, _Sxt, _Syt, _Sxyt;                                                              \
        if ((tu_) >= _tfl && (tu_) <= _tfh)                                                    \
        {                                                                                      \
            _At = T(1);                                                                        \
            _Sxt = T(0.5);                                                                     \
            _Syt = T(0.5);                                                                     \
            _Sxyt = T(0.25);                                                                   \
        }                                                                                      \
        else if (_nv_top == 0)                                                                 \
            single_cell_moments_simple(_xlo, _xhi, _xlo_next, _xhi_next,                       \
                                       T(tu_), &_At, &_Sxt, &_Syt, &_Sxyt);                    \
        else                                                                                   \
            single_cell_moments(s0, s1, s2, s3, _ftv_, _ftv1_, T(tu_),                         \
                                _xlo, _xhi, _xlo_next, _xhi_next, &_At, &_Sxt, &_Syt, &_Sxyt); \
        _row_Delta[(tu_) - minu] = _Syt - _Sxyt;                                               \
        _row_Sxyb[(tu_) - minu] = _Sxyt;                                                       \
        (W_) = (_At - _Sxt - _Syt + _Sxyt) + _pSxmSxy + _bd + _pSxyb;                          \
        _pSxmSxy = _Sxt - _Sxyt;                                                               \
        _pSxyb = _Sxyb_c;                                                                      \
    }

// Top-overflow post-pass: rows [maxv+1, maxv_full] contribute to texels in row maxv.
// acc_ is a statement that uses `_tu_top` (column) and `_W_top` (weight).
#define AB2_TOP_OVERFLOW(acc_)                                                                        \
    /* Flush left/right carries from the last main-loop row to their clamped destinations */          \
    if (maxv >= minv) {                                                                               \
        { const int _tu_top = 0;    const T _W_top = _left_carry;  acc_ }                            \
        { const int _tu_top = maxu; const T _W_top = _right_carry; acc_ }                            \
    }                                                                                                 \
    for (int _tv_top = maxv + 1; _tv_top <= maxv_full; _tv_top++)                                    \
    {                                                                                                 \
        T _xlo_tn, _xhi_tn, _slo_tn, _shi_tn; int _nv_tn;                                            \
        advance_scanline(s0, s1, s2, s3, T(_tv_top), T(_tv_top + 1),                                  \
                         _xlo, _xhi, _slo, _shi, &_xlo_tn, &_xhi_tn, &_slo_tn, &_shi_tn, &_nv_tn);  \
        /* LT corner: left overflow area → texel (0, maxv) */                                         \
        { T _dA_lt, _dSy_lt; left_overflow_moments(_xlo, _xlo_tn, &_dA_lt, &_dSy_lt);                \
          const int _tu_top = 0; const T _W_top = _dA_lt; acc_ }                                     \
        /* CT strip */                                                                                \
        const int _tu_s_tn = max(minu, (int)floorf((float)min(_xlo, _xlo_tn)));                       \
        const int _tu_e_tn = min(maxu, (int)ceilf((float)max(_xhi, _xhi_tn)));                        \
        T _sx_c_tn = T(0);                                                                            \
        for (int _tu_top = _tu_s_tn; _tu_top <= _tu_e_tn; _tu_top++)                                 \
        {                                                                                             \
            T _At_tn, _Sxt_tn, _Syt_tn, _Sxyt_tn;                                                    \
            if (_nv_tn == 0)                                                                          \
                single_cell_moments_simple(_xlo, _xhi, _xlo_tn, _xhi_tn,                              \
                                           T(_tu_top), &_At_tn, &_Sxt_tn, &_Syt_tn, &_Sxyt_tn);      \
            else                                                                                      \
                single_cell_moments(s0, s1, s2, s3, T(_tv_top), T(_tv_top + 1), T(_tu_top),           \
                                    _xlo, _xhi, _xlo_tn, _xhi_tn,                                     \
                                    &_At_tn, &_Sxt_tn, &_Syt_tn, &_Sxyt_tn);                         \
            { const T _W_top = _At_tn - _Sxt_tn + _sx_c_tn; acc_ }                                   \
            _sx_c_tn = _Sxt_tn;                                                                       \
        }                                                                                             \
        /* Flush Sx carry (clamped) */                                                                \
        if (_tu_e_tn >= _tu_s_tn) {                                                                   \
            const int _tu_top = min(maxu, _tu_e_tn + 1);                                              \
            const T _W_top = _sx_c_tn; acc_                                                           \
        }                                                                                             \
        /* RT corner: right overflow area → texel (maxu, maxv) */                                     \
        { T _dA_rt, _dSy_rt; right_overflow_moments(_xhi, _xhi_tn, T(maxu), &_dA_rt, &_dSy_rt);      \
          const int _tu_top = maxu; const T _W_top = _dA_rt; acc_ }                                   \
        _xlo = _xlo_tn; _xhi = _xhi_tn; _slo = _slo_tn; _shi = _shi_tn;                               \
    }

    template <typename T>
    inline __device__ T sample(
        at::PackedTensorAccessor32<const T, 4, at::RestrictPtrTraits> textures,
        int32_t g, int32_t k,
        vec2<T> s0, vec2<T> s1, vec2<T> s2, vec2<T> s3,
        int32_t minu, int32_t maxu, int32_t minv, int32_t maxv,
        int32_t minv_full, int32_t maxv_full,
        T area, T iarea, int texture_res_x, int texture_res_y)
    {
        AB2_INIT_SCANLINE()
        T value = T(0);
        for (int tv = minv; tv <= maxv; tv++)
        {
            AB2_ROW_EXTENTS(tv)
            AB2_ROW_OVERFLOW()
            int tu_s, tu_e;
            tu_range(minu, maxu, lo_bot, hi_bot, lo_top, hi_top, &tu_s, &tu_e);
            AB2_COL_CACHE_INIT(tv)
            for (int tu = tu_s; tu <= tu_e; tu++)
            {
                T W;
                AB2_COMPUTE_W(tu, tv, W)
                if (tu == 0)    W += _left_carry + _dA_L - _dSy_L;
                if (tu == maxu) { W += _right_carry + _dA_R - _dSy_R; if (tv == minv) W += _rb_extra; }
                value += textures[g][tv][tu][k] * W;
            }
            _left_carry = _dSy_L;
            _right_carry = _dSy_R;
            AB2_SLIDE_WINDOW()
        }
        AB2_TOP_OVERFLOW(value += textures[g][maxv][_tu_top][k] * _W_top;)
        return value * iarea;
    }

    template <uint32_t COLOR_DIM, typename T>
    inline __device__ void alpha_color_sample(
        at::PackedTensorAccessor32<const T, 4, at::RestrictPtrTraits> textures,
        int32_t g,
        vec2<T> s0, vec2<T> s1, vec2<T> s2, vec2<T> s3,
        int32_t minu, int32_t maxu, int32_t minv, int32_t maxv,
        int32_t minv_full, int32_t maxv_full,
        T area, T iarea, int texture_res_x, int texture_res_y,
        T *alpha, T col[COLOR_DIM])
    {
        const int alpha_k = textures.size(3) - 1;
        AB2_INIT_SCANLINE()
        for (int tv = minv; tv <= maxv; tv++)
        {
            AB2_ROW_EXTENTS(tv)
            AB2_ROW_OVERFLOW()
            int tu_s, tu_e;
            tu_range(minu, maxu, lo_bot, hi_bot, lo_top, hi_top, &tu_s, &tu_e);
            AB2_COL_CACHE_INIT(tv)
            for (int tu = tu_s; tu <= tu_e; tu++)
            {
                T W;
                AB2_COMPUTE_W(tu, tv, W)
                if (tu == 0)    W += _left_carry + _dA_L - _dSy_L;
                if (tu == maxu) { W += _right_carry + _dA_R - _dSy_R; if (tv == minv) W += _rb_extra; }
                GSPLAT_PRAGMA_UNROLL
                for (int k = 0; k < COLOR_DIM; ++k)
                    col[k] += textures[g][tv][tu][k] * W;
                *alpha += textures[g][tv][tu][alpha_k] * W;
            }
            _left_carry = _dSy_L;
            _right_carry = _dSy_R;
            AB2_SLIDE_WINDOW()
        }
        AB2_TOP_OVERFLOW(
            GSPLAT_PRAGMA_UNROLL for (int k = 0; k < COLOR_DIM; ++k) col[k] += textures[g][maxv][_tu_top][k] * _W_top;
            *alpha += textures[g][maxv][_tu_top][alpha_k] * _W_top;)
        GSPLAT_PRAGMA_UNROLL
        for (int k = 0; k < COLOR_DIM; ++k)
            col[k] *= iarea;
        *alpha *= iarea;
    }

    template <uint32_t COLOR_DIM, typename T>
    inline __device__ void color_sample(
        at::PackedTensorAccessor32<const T, 4, at::RestrictPtrTraits> textures,
        int32_t g,
        vec2<T> s0, vec2<T> s1, vec2<T> s2, vec2<T> s3,
        int32_t minu, int32_t maxu, int32_t minv, int32_t maxv,
        int32_t minv_full, int32_t maxv_full,
        T area, T iarea, int texture_res_x, int texture_res_y,
        T col[COLOR_DIM])
    {
        AB2_INIT_SCANLINE()
        for (int tv = minv; tv <= maxv; tv++)
        {
            AB2_ROW_EXTENTS(tv)
            AB2_ROW_OVERFLOW()
            int tu_s, tu_e;
            tu_range(minu, maxu, lo_bot, hi_bot, lo_top, hi_top, &tu_s, &tu_e);
            AB2_COL_CACHE_INIT(tv)
            for (int tu = tu_s; tu <= tu_e; tu++)
            {
                T W;
                AB2_COMPUTE_W(tu, tv, W)
                if (tu == 0)    W += _left_carry + _dA_L - _dSy_L;
                if (tu == maxu) { W += _right_carry + _dA_R - _dSy_R; if (tv == minv) W += _rb_extra; }
                GSPLAT_PRAGMA_UNROLL
                for (int k = 0; k < COLOR_DIM; ++k)
                    col[k] += textures[g][tv][tu][k] * W;
            }
            _left_carry = _dSy_L;
            _right_carry = _dSy_R;
            AB2_SLIDE_WINDOW()
        }
        AB2_TOP_OVERFLOW(
            GSPLAT_PRAGMA_UNROLL for (int k = 0; k < COLOR_DIM; ++k) col[k] += textures[g][maxv][_tu_top][k] * _W_top;)
        GSPLAT_PRAGMA_UNROLL
        for (int k = 0; k < COLOR_DIM; ++k)
            col[k] *= iarea;
    }

    template <typename T>
    inline __device__ void update(
        at::PackedTensorAccessor32<T, 4, at::RestrictPtrTraits> v_textures,
        int32_t g, int32_t k,
        vec2<T> s0, vec2<T> s1, vec2<T> s2, vec2<T> s3,
        int32_t minu, int32_t maxu, int32_t minv, int32_t maxv,
        int32_t minv_full, int32_t maxv_full,
        T area, T iarea, int texture_res_x, int texture_res_y, T delta)
    {
        T ndelta = delta * iarea;
        AB2_INIT_SCANLINE()
        for (int tv = minv; tv <= maxv; tv++)
        {
            AB2_ROW_EXTENTS(tv)
            AB2_ROW_OVERFLOW()
            int tu_s, tu_e;
            tu_range(minu, maxu, lo_bot, hi_bot, lo_top, hi_top, &tu_s, &tu_e);
            AB2_COL_CACHE_INIT(tv)
            for (int tu = tu_s; tu <= tu_e; tu++)
            {
                T W;
                AB2_COMPUTE_W(tu, tv, W)
                if (tu == 0)    W += _left_carry + _dA_L - _dSy_L;
                if (tu == maxu) { W += _right_carry + _dA_R - _dSy_R; if (tv == minv) W += _rb_extra; }
                gpuAtomicAdd(&v_textures[g][tv][tu][k], ndelta * W);
            }
            _left_carry = _dSy_L;
            _right_carry = _dSy_R;
            AB2_SLIDE_WINDOW()
        }
        AB2_TOP_OVERFLOW(gpuAtomicAdd(&v_textures[g][maxv][_tu_top][k], ndelta * _W_top);)
    }

    template <uint32_t COLOR_DIM, typename T>
    inline __device__ void color_sample_and_update(
        at::PackedTensorAccessor32<const T, 4, at::RestrictPtrTraits> textures,
        at::PackedTensorAccessor32<T, 4, at::RestrictPtrTraits> v_textures,
        int32_t g,
        vec2<T> s0, vec2<T> s1, vec2<T> s2, vec2<T> s3,
        int32_t minu, int32_t maxu, int32_t minv, int32_t maxv,
        int32_t minv_full, int32_t maxv_full,
        T area, T iarea, int texture_res_x, int texture_res_y,
        T col[COLOR_DIM], T deltas[COLOR_DIM])
    {
        T ndeltas[COLOR_DIM];
        GSPLAT_PRAGMA_UNROLL
        for (int k = 0; k < COLOR_DIM; k++)
            ndeltas[k] = deltas[k] * iarea;
        AB2_INIT_SCANLINE()
        for (int tv = minv; tv <= maxv; tv++)
        {
            AB2_ROW_EXTENTS(tv)
            AB2_ROW_OVERFLOW()
            int tu_s, tu_e;
            tu_range(minu, maxu, lo_bot, hi_bot, lo_top, hi_top, &tu_s, &tu_e);
            AB2_COL_CACHE_INIT(tv)
            for (int tu = tu_s; tu <= tu_e; tu++)
            {
                T W;
                AB2_COMPUTE_W(tu, tv, W)
                if (tu == 0)    W += _left_carry + _dA_L - _dSy_L;
                if (tu == maxu) { W += _right_carry + _dA_R - _dSy_R; if (tv == minv) W += _rb_extra; }
                GSPLAT_PRAGMA_UNROLL
                for (int k = 0; k < COLOR_DIM; ++k)
                {
                    col[k] += textures[g][tv][tu][k] * W;
                    gpuAtomicAdd(&v_textures[g][tv][tu][k], ndeltas[k] * W);
                }
            }
            _left_carry = _dSy_L;
            _right_carry = _dSy_R;
            AB2_SLIDE_WINDOW()
        }
        AB2_TOP_OVERFLOW(
            GSPLAT_PRAGMA_UNROLL for (int k = 0; k < COLOR_DIM; ++k) {
                col[k] += textures[g][maxv][_tu_top][k] * _W_top;
                gpuAtomicAdd(&v_textures[g][maxv][_tu_top][k], ndeltas[k] * _W_top);
            })
        GSPLAT_PRAGMA_UNROLL
        for (int k = 0; k < COLOR_DIM; ++k)
            col[k] *= iarea;
    }

#undef AB2_INIT_SCANLINE
#undef AB2_ROW_EXTENTS
#undef AB2_SLIDE_WINDOW
#undef AB2_ROW_OVERFLOW
#undef AB2_COL_CACHE_INIT
#undef AB2_COMPUTE_W
#undef AB2_TOP_OVERFLOW

} // namespace gsplat::anisotropic_bilinear2

#endif // GSPLAT_CUDA_ANISOTROPIC_BILINEAR2_FILTER_H
