
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
    inline __device__ T max4(T v0, T v1, T v2, T v3) { return max(max(v0, v1), max(v2, v3)); }
    template <typename T>
    inline __device__ T min4(T v0, T v1, T v2, T v3) { return min(min(v0, v1), min(v2, v3)); }

    template <typename T>
    inline __device__ T precompute(
        vec2<T> *s0, vec2<T> *s1, vec2<T> *s2, vec2<T> *s3,
        int32_t *minu, int32_t *minv, int32_t *maxu, int32_t *maxv,
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
        *minv = max(0, (int32_t)floor(min4(s0->y, s1->y, s2->y, s3->y)));
        *maxv = min(texture_res_y - 1, (int32_t)ceil(max4(s0->y, s1->y, s2->y, s3->y)));
        return area;
    }

    // ---- Rolling scanline helpers ----------------------------------------
    //
    // A convex polygon has at most 2 x-intersections with any horizontal line.
    // We maintain these two x-values (xlo ≤ xhi) as rolling state that is
    // advanced one row at a time by adding the edge slope (x += slope).
    // A full re-scan (O(4)) is triggered only when a polygon vertex is crossed,
    // which happens at most twice (for the ≤2 interior vertices) over the entire
    // column height H.  All other rows cost O(1).

    // Find the (at most 2) polygon edges active at y=y0, return their x-values
    // (sorted lo ≤ hi) and slopes.  Called once at init and on vertex crossings.
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

    // Advance x-boundary state from y=y_curr to y=y_next (= y_curr + 1).
    // If any vertex lies in (y_curr, y_next] a full re-scan is done (O(4));
    // otherwise just x += slope  (O(1)).
    template <typename T>
    inline __device__ void advance_scanline(
        vec2<T> s0, vec2<T> s1, vec2<T> s2, vec2<T> s3,
        T y_curr, T y_next,
        T xlo_in, T xhi_in, T slo_in, T shi_in,
        T *xlo_out, T *xhi_out, T *slo_out, T *shi_out)
    {
        const vec2<T> verts[4] = {s0, s1, s2, s3};
        bool rescan = false;
        GSPLAT_PRAGMA_UNROLL
        for (int i = 0; i < 4; i++)
            if (verts[i].y > y_curr && verts[i].y <= y_next)
            {
                rescan = true;
                break;
            }

        if (rescan)
        {
            scan_active_edges(s0, s1, s2, s3, y_next, xlo_out, xhi_out, slo_out, shi_out);
        }
        else
        {
            *xlo_out = xlo_in + slo_in;
            *xhi_out = xhi_in + shi_in;
            *slo_out = slo_in;
            *shi_out = shi_in;
        }
    }

    // Extend [*xlo, *xhi] with any vertex whose y is strictly inside (y_lo, y_hi).
    // Needed because a convex polygon vertex can be the extreme x within a strip
    // even though it doesn't lie on an integer y-boundary.
    template <typename T>
    inline __device__ void fold_vertices(
        vec2<T> s0, vec2<T> s1, vec2<T> s2, vec2<T> s3,
        T y_lo, T y_hi, T *xlo, T *xhi)
    {
        const vec2<T> verts[4] = {s0, s1, s2, s3};
        GSPLAT_PRAGMA_UNROLL
        for (int i = 0; i < 4; i++)
            if (verts[i].y > y_lo && verts[i].y < y_hi)
            {
                *xlo = min(*xlo, verts[i].x);
                *xhi = max(*xhi, verts[i].x);
            }
    }

    // ---- Analytical moment integration ----------------------------------------

    // trapezoid_moments: integrate polygon x-span over normalised cell [0,1]x[0,1].
    // Three fast paths before falling back to the general breakpoint loop.
    template <typename T>
    inline __device__ void trapezoid_moments(
        T al, T ar, T bl, T br,
        T *A, T *Sx, T *Sy, T *Sxy)
    {
        // Fast path 1: cell fully covered — polygon spans all of [0,1] in x
        if (al<=T(0)&&ar>=T(1)&&bl<=T(0)&&br>=T(1)) {
            *A=T(1);*Sx=T(0.5);*Sy=T(0.5);*Sxy=T(0.25); return;
        }
        // Fast path 2: no clamping — all boundaries already in [0,1]
        if (al>=T(0)&&ar<=T(1)&&bl>=T(0)&&br<=T(1)) {
            const T qa=bl-al,qb=br-ar,C=ar-al,D=qb-qa;
            *A=C+D*T(0.5);
            *Sy=C*T(0.5)+D*(T(1)/T(3));
            const T E=ar*ar-al*al,F=T(2)*(ar*qb-al*qa),G=qb*qb-qa*qa;
            *Sx=(E+F*T(0.5)+G*(T(1)/T(3)))*T(0.5);
            *Sxy=(E*T(0.5)+F*(T(1)/T(3))+G*T(0.25))*T(0.5);
            return;
        }
        // General case: insert breakpoints where xl/xr cross 0 or 1, sort, integrate
        *A=T(0);*Sx=T(0);*Sy=T(0);*Sxy=T(0);
        const T dxl=bl-al,dxr=br-ar;
        T bp[7]; int nb=2; bp[0]=T(0); bp[1]=T(1);
        #define TM_ADD(t_) do{T _t=(t_);if(_t>T(0)&&_t<T(1))bp[nb++]=_t;}while(0)
        if(dxl!=T(0)){TM_ADD(-al/dxl);TM_ADD((T(1)-al)/dxl);}
        if(dxr!=T(0)){TM_ADD(-ar/dxr);TM_ADD((T(1)-ar)/dxr);}
        {T den=dxl-dxr;if(den!=T(0))TM_ADD((ar-al)/den);}
        #undef TM_ADD
        for(int i=1;i<nb;i++){T key=bp[i];int j=i-1;while(j>=0&&bp[j]>key){bp[j+1]=bp[j];j--;}bp[j+1]=key;}
        for(int i=0;i<nb-1;i++){
            const T t0=bp[i],t1=bp[i+1],dt=t1-t0;
            if(dt<=T(0))continue;
            const T la=min(max(al+dxl*t0,T(0)),T(1));
            const T lb=min(max(al+dxl*t1,T(0)),T(1));
            const T ra=min(max(ar+dxr*t0,T(0)),T(1));
            const T rb=min(max(ar+dxr*t1,T(0)),T(1));
            const T C=ra-la;
            if(C<=T(0)&&(rb-lb)<=T(0))continue;
            const T qa=(lb-la)/dt,qb=(rb-ra)/dt,D=qb-qa;
            const T dt2=dt*dt,dt3=dt2*dt,dt4=dt3*dt;
            const T dA=C*dt+D*dt2*T(0.5);
            *A+=dA;
            *Sy+=t0*dA+C*dt2*T(0.5)+D*dt3*(T(1)/T(3));
            const T E=ra*ra-la*la,F=T(2)*(ra*qb-la*qa),G=qb*qb-qa*qa;
            const T dSx=(E*dt+F*dt2*T(0.5)+G*dt3*(T(1)/T(3)))*T(0.5);
            *Sx+=dSx;
            *Sxy+=t0*dSx+(E*dt2*T(0.5)+F*dt3*(T(1)/T(3))+G*dt4*T(0.25))*T(0.5);
        }
    }

    // accumulate_strip: add sub-strip contribution (y_local in [t_lo,t_hi]) to moments.
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

    // polygon_x_at_y: exact xl (min) and xr (max) of polygon at y.
    // Returns xl>xr if the polygon does not span y.
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

    // process_strip: moments for cells [xu,xu+1] (_pp) and [xu-1,xu] (_mp) clipped
    // to strip [y_lo,y_hi].  xl_lo/xr_lo and xl_hi/xr_hi are pre-computed scanline
    // x-bounds (inverted sentinel if the polygon doesn't reach that y).  Falls back
    // to polygon_x_at_y only when the scanline value is degenerate or for interior
    // y-vertices.  Fast path for the common nv=0 case skips all loop overhead.
    template <typename T>
    inline __device__ void process_strip(
        vec2<T> s0, vec2<T> s1, vec2<T> s2, vec2<T> s3,
        T y_lo, T y_hi, T xu,
        T xl_lo, T xr_lo,   // scanline polygon x at y_lo (xl>xr = degenerate)
        T xl_hi, T xr_hi,   // scanline polygon x at y_hi
        T *A_pp, T *Sx_pp, T *Sy_pp, T *Sxy_pp,
        T *A_mp, T *Sx_mp, T *Sy_mp, T *Sxy_mp)
    {
        *A_pp=*Sx_pp=*Sy_pp=*Sxy_pp=T(0);
        *A_mp=*Sx_mp=*Sy_mp=*Sxy_mp=T(0);
        // Validate scanline bounds; fall back to exact scan when degenerate
        if(xl_lo>xr_lo) polygon_x_at_y(s0,s1,s2,s3,y_lo,&xl_lo,&xr_lo);
        if(xl_hi>xr_hi) polygon_x_at_y(s0,s1,s2,s3,y_hi,&xl_hi,&xr_hi);
        // Collect interior y-vertices (insertion-sorted, up to 4)
        T vy[4]; int nv=0;
        const vec2<T> verts[4]={s0,s1,s2,s3};
        GSPLAT_PRAGMA_UNROLL
        for(int i=0;i<4;i++){
            const T y=verts[i].y;
            if(y>y_lo&&y<y_hi){int j=nv;while(j>0&&vy[j-1]>y){vy[j]=vy[j-1];j--;}vy[j]=y;nv++;}
        }
        if(nv==0){
            // Common case: single sub-strip spanning full [0,1] y range.
            // For dt=1, t_lo=0: trapezoid_moments output is already cell-local.
            if(!(xl_lo>xr_lo)&&!(xl_hi>xr_hi)){
                trapezoid_moments(xl_lo-xu,    xr_lo-xu,    xl_hi-xu,    xr_hi-xu,    A_pp,Sx_pp,Sy_pp,Sxy_pp);
                trapezoid_moments(xl_lo-xu+T(1),xr_lo-xu+T(1),xl_hi-xu+T(1),xr_hi-xu+T(1),A_mp,Sx_mp,Sy_mp,Sxy_mp);
            }
            return;
        }
        // Slow path: multiple sub-strips
        T ypts[6]; int ny=0;
        ypts[ny++]=y_lo;
        for(int i=0;i<nv;i++) ypts[ny++]=vy[i];
        ypts[ny++]=y_hi;
        T xlo_a=xl_lo, xhi_a=xr_lo;
        for(int i=0;i<ny-1;i++){
            T xlo_b,xhi_b;
            if(i==ny-2){xlo_b=xl_hi;xhi_b=xr_hi;}
            else polygon_x_at_y(s0,s1,s2,s3,ypts[i+1],&xlo_b,&xhi_b);
            if(!(xlo_a>xhi_a)&&!(xlo_b>xhi_b)){
                const T t_lo=ypts[i]-y_lo,t_hi=ypts[i+1]-y_lo;
                accumulate_strip(xlo_a-xu,      xhi_a-xu,      xlo_b-xu,      xhi_b-xu,      t_lo,t_hi,A_pp,Sx_pp,Sy_pp,Sxy_pp);
                accumulate_strip(xlo_a-xu+T(1), xhi_a-xu+T(1), xlo_b-xu+T(1), xhi_b-xu+T(1), t_lo,t_hi,A_mp,Sx_mp,Sy_mp,Sxy_mp);
            }
            xlo_a=xlo_b; xhi_a=xhi_b;
        }
    }

    // tent_weight: bilinear tent weight for texel (tu,tv).
    // Scanline x-bounds are passed in to avoid redundant polygon_x_at_y calls
    // at the strip boundaries (they are already computed by AB_ROW_EXTENTS).
    template <typename T>
    inline __device__ T tent_weight(
        vec2<T> s0, vec2<T> s1, vec2<T> s2, vec2<T> s3,
        int tu, int tv,
        T xlo_prev, T xhi_prev,   // polygon x at y=tv-1
        T xlo_curr, T xhi_curr,   // polygon x at y=tv
        T xlo_next, T xhi_next)   // polygon x at y=tv+1
    {
        T W=T(0);
        T A_pp,Sx_pp,Sy_pp,Sxy_pp,A_mp,Sx_mp,Sy_mp,Sxy_mp;
        const T ftu=T(tu),ftv=T(tv);
        // Top strip [tv,tv+1]: Q++ (1-x)(1-y) and Q-+ x(1-y)
        process_strip(s0,s1,s2,s3,ftv,ftv+T(1),ftu,
                      xlo_curr,xhi_curr,xlo_next,xhi_next,
                      &A_pp,&Sx_pp,&Sy_pp,&Sxy_pp,
                      &A_mp,&Sx_mp,&Sy_mp,&Sxy_mp);
        W+=A_pp-Sx_pp-Sy_pp+Sxy_pp;
        W+=Sx_mp-Sxy_mp;
        // Bottom strip [tv-1,tv]: Q+- (1-x)y and Q-- xy
        process_strip(s0,s1,s2,s3,ftv-T(1),ftv,ftu,
                      xlo_prev,xhi_prev,xlo_curr,xhi_curr,
                      &A_pp,&Sx_pp,&Sy_pp,&Sxy_pp,
                      &A_mp,&Sx_mp,&Sy_mp,&Sxy_mp);
        W+=Sy_pp-Sxy_pp;
        W+=Sxy_mp;
        return W;
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

// ---- Sampling functions -----------------------------------------------
// Each function initialises the rolling scanline at y = minv-1, advances
// once to y = minv, then loops: advance to tv+1, derive strip extents,
// fold vertices, sample, slide the three-snapshot window.

// Shared scanline initialisation and first advance.
#define AB_INIT_SCANLINE()                                                      \
    T _xlo, _xhi, _slo, _shi;                                                   \
    scan_active_edges(s0, s1, s2, s3, T(minv - 1), &_xlo, &_xhi, &_slo, &_shi); \
    T _xlo_prev = _xlo, _xhi_prev = _xhi;                                       \
    advance_scanline(s0, s1, s2, s3, T(minv - 1), T(minv),                      \
                     _xlo, _xhi, _slo, _shi, &_xlo, &_xhi, &_slo, &_shi);       \
    /* _xlo_prev/xhi_prev = x at y=minv-1; _xlo/xhi = x at y=minv */

// Per-row: advance to tv+1, form strip extents, update window.
#define AB_ROW_EXTENTS(tv_)                                                \
    T _xlo_next, _xhi_next, _slo_next, _shi_next;                          \
    advance_scanline(s0, s1, s2, s3, T(tv_), T((tv_) + 1),                 \
                     _xlo, _xhi, _slo, _shi,                               \
                     &_xlo_next, &_xhi_next, &_slo_next, &_shi_next);      \
    T lo_bot = min(_xlo_prev, _xlo), hi_bot = max(_xhi_prev, _xhi);        \
    T lo_top = min(_xlo, _xlo_next), hi_top = max(_xhi, _xhi_next);        \
    fold_vertices(s0, s1, s2, s3, T((tv_) - 1), T(tv_), &lo_bot, &hi_bot); \
    fold_vertices(s0, s1, s2, s3, T(tv_), T((tv_) + 1), &lo_top, &hi_top);

#define AB_SLIDE_WINDOW() \
    _xlo_prev = _xlo;     \
    _xhi_prev = _xhi;     \
    _xlo = _xlo_next;     \
    _xhi = _xhi_next;     \
    _slo = _slo_next;     \
    _shi = _shi_next;

    template <typename T>
    inline __device__ T sample(
        at::PackedTensorAccessor32<const T, 4, at::RestrictPtrTraits> textures,
        int32_t g, int32_t k,
        vec2<T> s0, vec2<T> s1, vec2<T> s2, vec2<T> s3,
        int32_t minu, int32_t maxu, int32_t minv, int32_t maxv,
        T area, T iarea, int texture_res_x, int texture_res_y)
    {
        AB_INIT_SCANLINE()
        T value = T(0);
        for (int tv = minv; tv <= maxv; tv++)
        {
            AB_ROW_EXTENTS(tv)
            int tu_s, tu_e;
            tu_range(minu, maxu, lo_bot, hi_bot, lo_top, hi_top, &tu_s, &tu_e);
            for (int tu = tu_s; tu <= tu_e; tu++)
            {
                T W = tent_weight(s0,s1,s2,s3,tu,tv,_xlo_prev,_xhi_prev,_xlo,_xhi,_xlo_next,_xhi_next);
                if (W == T(0))
                    continue;
                value += textures[g][tv][tu][k] * W;
            }
            AB_SLIDE_WINDOW()
        }
        return value * iarea;
    }

    template <uint32_t COLOR_DIM, typename T>
    inline __device__ void color_sample(
        at::PackedTensorAccessor32<const T, 4, at::RestrictPtrTraits> textures,
        int32_t g,
        vec2<T> s0, vec2<T> s1, vec2<T> s2, vec2<T> s3,
        int32_t minu, int32_t maxu, int32_t minv, int32_t maxv,
        T area, T iarea, int texture_res_x, int texture_res_y,
        T col[COLOR_DIM])
    {
        AB_INIT_SCANLINE()
        for (int tv = minv; tv <= maxv; tv++)
        {
            AB_ROW_EXTENTS(tv)
            int tu_s, tu_e;
            tu_range(minu, maxu, lo_bot, hi_bot, lo_top, hi_top, &tu_s, &tu_e);
            for (int tu = tu_s; tu <= tu_e; tu++)
            {
                T W = tent_weight(s0,s1,s2,s3,tu,tv,_xlo_prev,_xhi_prev,_xlo,_xhi,_xlo_next,_xhi_next);
                if (W == T(0))
                    continue;
                T wi = W * iarea;
                GSPLAT_PRAGMA_UNROLL
                for (int k = 0; k < COLOR_DIM; ++k)
                    col[k] += textures[g][tv][tu][k] * wi;
            }
            AB_SLIDE_WINDOW()
        }
    }

    template <uint32_t COLOR_DIM, typename T>
    inline __device__ void alpha_color_sample(
        at::PackedTensorAccessor32<const T, 4, at::RestrictPtrTraits> textures,
        int32_t g,
        vec2<T> s0, vec2<T> s1, vec2<T> s2, vec2<T> s3,
        int32_t minu, int32_t maxu, int32_t minv, int32_t maxv,
        T area, T iarea, int texture_res_x, int texture_res_y,
        T *alpha, T col[COLOR_DIM])
    {
        const int alpha_k = textures.size(3) - 1;
        AB_INIT_SCANLINE()
        for (int tv = minv; tv <= maxv; tv++)
        {
            AB_ROW_EXTENTS(tv)
            int tu_s, tu_e;
            tu_range(minu, maxu, lo_bot, hi_bot, lo_top, hi_top, &tu_s, &tu_e);
            for (int tu = tu_s; tu <= tu_e; tu++)
            {
                T W = tent_weight(s0,s1,s2,s3,tu,tv,_xlo_prev,_xhi_prev,_xlo,_xhi,_xlo_next,_xhi_next);
                if (W == T(0))
                    continue;
                T wi = W * iarea;
                GSPLAT_PRAGMA_UNROLL
                for (int k = 0; k < COLOR_DIM; ++k)
                    col[k] += textures[g][tv][tu][k] * wi;
                *alpha += textures[g][tv][tu][alpha_k] * wi;
            }
            AB_SLIDE_WINDOW()
        }
    }

    template <typename T>
    inline __device__ void update(
        at::PackedTensorAccessor32<T, 4, at::RestrictPtrTraits> v_textures,
        int32_t g, int32_t k,
        vec2<T> s0, vec2<T> s1, vec2<T> s2, vec2<T> s3,
        int32_t minu, int32_t maxu, int32_t minv, int32_t maxv,
        T area, T iarea, int texture_res_x, int texture_res_y, T delta)
    {
        T ndelta = delta * iarea;
        AB_INIT_SCANLINE()
        for (int tv = minv; tv <= maxv; tv++)
        {
            AB_ROW_EXTENTS(tv)
            int tu_s, tu_e;
            tu_range(minu, maxu, lo_bot, hi_bot, lo_top, hi_top, &tu_s, &tu_e);
            for (int tu = tu_s; tu <= tu_e; tu++)
            {
                T W = tent_weight(s0,s1,s2,s3,tu,tv,_xlo_prev,_xhi_prev,_xlo,_xhi,_xlo_next,_xhi_next);
                if (W == T(0))
                    continue;
                gpuAtomicAdd(&v_textures[g][tv][tu][k], ndelta * W);
            }
            AB_SLIDE_WINDOW()
        }
    }

    template <uint32_t COLOR_DIM, typename T>
    inline __device__ void color_sample_and_update(
        at::PackedTensorAccessor32<const T, 4, at::RestrictPtrTraits> textures,
        at::PackedTensorAccessor32<T, 4, at::RestrictPtrTraits> v_textures,
        int32_t g,
        vec2<T> s0, vec2<T> s1, vec2<T> s2, vec2<T> s3,
        int32_t minu, int32_t maxu, int32_t minv, int32_t maxv,
        T area, T iarea, int texture_res_x, int texture_res_y,
        T col[COLOR_DIM], T deltas[COLOR_DIM])
    {
        T ndeltas[COLOR_DIM];
        GSPLAT_PRAGMA_UNROLL
        for (int k = 0; k < COLOR_DIM; k++)
            ndeltas[k] = deltas[k] * iarea;
        AB_INIT_SCANLINE()
        for (int tv = minv; tv <= maxv; tv++)
        {
            AB_ROW_EXTENTS(tv)
            int tu_s, tu_e;
            tu_range(minu, maxu, lo_bot, hi_bot, lo_top, hi_top, &tu_s, &tu_e);
            for (int tu = tu_s; tu <= tu_e; tu++)
            {
                T W = tent_weight(s0,s1,s2,s3,tu,tv,_xlo_prev,_xhi_prev,_xlo,_xhi,_xlo_next,_xhi_next);
                if (W == T(0))
                    continue;
                T wi = W * iarea;
                GSPLAT_PRAGMA_UNROLL
                for (int k = 0; k < COLOR_DIM; ++k)
                {
                    col[k] += textures[g][tv][tu][k] * wi;
                    gpuAtomicAdd(&v_textures[g][tv][tu][k], ndeltas[k] * W);
                }
            }
            AB_SLIDE_WINDOW()
        }
    }

#undef AB_INIT_SCANLINE
#undef AB_ROW_EXTENTS
#undef AB_SLIDE_WINDOW

} // namespace gsplat::anisotropic_bilinear

#endif // GSPLAT_CUDA_ANISOTROPIC_BILINEAR_FILTER_H
