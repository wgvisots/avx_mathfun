/*
   AVX-512 implementation of sin, cos, sincos, exp, log, atan and atan2

   This is a port of avx_mathfun.h (AVX/AVX2, 8 floats per vector) to
   AVX-512 (16 floats per vector).  The polynomial approximations are the
   same Cephes-derived ones used by avx_mathfun.h, so results agree with the
   AVX2 versions to within an ulp or two (the AVX-512 versions use FMA
   throughout, which changes rounding slightly).

   Based on "avx_mathfun.h" by Giovanni Garberoglio, which is in turn based on
   "sse_mathfun.h", by Julien Pommier
   http://gruntthepeon.free.fr/ssemath/

   Copyright (C) 2012 Giovanni Garberoglio
   Interdisciplinary Laboratory for Computational Science (LISC)
   Fondazione Bruno Kessler and University of Trento
   via Sommarive, 18
   I-38123 Trento (Italy)

   AVX-512 port Copyright (C) 2026 W. Greg Visotski

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)

  ---------------------------------------------------------------------------
  Requirements
  ---------------------------------------------------------------------------
  Only AVX-512F (the foundation subset) is required, so this header works on
  every AVX-512 capable CPU (Skylake-SP and later, Ice Lake, Sapphire Rapids,
  AMD Zen 4/Zen 5, ...) and on every AVX10 CPU (AVX10.1/AVX10.2 are supersets
  of AVX-512F).  Compile with e.g. -mavx512f, -march=skylake-avx512,
  -march=x86-64-v4, -mavx10.1 or -march=native on a capable machine.

  ---------------------------------------------------------------------------
  Differences from the AVX2 versions in avx_mathfun.h
  ---------------------------------------------------------------------------
  The AVX-512 instruction set has dedicated instructions for some of the
  bit-twiddling the AVX2 code does by hand (getexp/getmant for frexp, scalef
  for ldexp, mask registers for selects).  Using them also fixes several edge
  cases, so these functions follow the C standard library more closely:

    log512_ps : subnormal inputs are handled correctly (AVX2 clamps them to
                FLT_MIN); log(+-0) = -inf (AVX2: NaN); log(+inf) = +inf.
    exp512_ps : results that underflow become subnormal / 0 and results that
                overflow become +inf (AVX2 clamps to roughly +-88.38);
                NaN propagates.
    atan      : atan(-0) = -0; NaN propagates.
    atan2     : same algorithm and special-value handling as the AVX2
                version (signed zeros, infinities and NaN like std::atan2).

  sin/cos/sincos keep the Cephes range reduction, so as with the AVX2
  versions accuracy is excellent for |x| < 8192 and degrades beyond that.
*/

#ifndef AVX512_MATHFUN_H
#define AVX512_MATHFUN_H

#include <cstddef>
#include <iostream>
#include <limits>
#include <numbers>
#include <immintrin.h>

#if !defined(__AVX512F__)
#error "avx512_mathfun.h requires AVX-512F. Compile with -mavx512f (or -march=x86-64-v4, -mavx10.1, -march=native on a capable CPU, ...)."
#endif

namespace avx512_mathfun {

typedef __m512    v16sf;  // vector of 16 float  (avx512)
typedef __m512i   v16si;  // vector of 16 int32  (avx512)
typedef __mmask16 v16mask; // one bit per lane

namespace detail {

// Bitwise float ops.  _mm512_and_ps & friends need AVX-512DQ; going through
// the integer domain needs only AVX-512F and compiles to the same thing.
inline v16sf and_ps(v16sf a, v16sf b) noexcept {
    return _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(a), _mm512_castps_si512(b)));
}
inline v16sf or_ps(v16sf a, v16sf b) noexcept {
    return _mm512_castsi512_ps(_mm512_or_si512(_mm512_castps_si512(a), _mm512_castps_si512(b)));
}
inline v16sf xor_ps(v16sf a, v16sf b) noexcept {
    return _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(a), _mm512_castps_si512(b)));
}
inline v16sf sign_bits(v16sf a) noexcept {
    return and_ps(a, _mm512_castsi512_ps(_mm512_set1_epi32(static_cast<int>(0x80000000u))));
}
// lanes whose sign bit is set (including -0.0 and negative NaNs)
inline v16mask sign_set_mask(v16sf a) noexcept {
    return _mm512_test_epi32_mask(_mm512_castps_si512(a), _mm512_set1_epi32(static_cast<int>(0x80000000u)));
}

// Cephes constants (identical to avx_mathfun.h)
constexpr float cephes_SQRTHF = 0.707106781186547524F;
constexpr float cephes_log_p0 = 7.0376836292E-2F;
constexpr float cephes_log_p1 = -1.1514610310E-1F;
constexpr float cephes_log_p2 = 1.1676998740E-1F;
constexpr float cephes_log_p3 = -1.2420140846E-1F;
constexpr float cephes_log_p4 = +1.4249322787E-1F;
constexpr float cephes_log_p5 = -1.6668057665E-1F;
constexpr float cephes_log_p6 = +2.0000714765E-1F;
constexpr float cephes_log_p7 = -2.4999993993E-1F;
constexpr float cephes_log_p8 = +3.3333331174E-1F;
constexpr float cephes_log_q1 = -2.12194440e-4F;
constexpr float cephes_log_q2 = 0.693359375F;

constexpr float cephes_LOG2EF = 1.44269504088896341F;
constexpr float cephes_exp_C1 = 0.693359375F;
constexpr float cephes_exp_C2 = -2.12194440e-4F;
constexpr float cephes_exp_p0 = 1.9875691500E-4F;
constexpr float cephes_exp_p1 = 1.3981999507E-3F;
constexpr float cephes_exp_p2 = 8.3334519073E-3F;
constexpr float cephes_exp_p3 = 4.1665795894E-2F;
constexpr float cephes_exp_p4 = 1.6666665459E-1F;
constexpr float cephes_exp_p5 = 5.0000001201E-1F;
// Inputs are clamped to this range before range reduction.  Anything above
// exp_hi overflows to +inf in the final scalef, anything below exp_lo
// underflows to 0.
constexpr float exp_hi = 89.0F;
constexpr float exp_lo = -104.0F;

constexpr float minus_cephes_DP1 = -0.78515625F;
constexpr float minus_cephes_DP2 = -2.4187564849853515625e-4F;
constexpr float minus_cephes_DP3 = -3.77489497744594108e-8F;
constexpr float sincof_p0 = -1.9515295891E-4F;
constexpr float sincof_p1 = 8.3321608736E-3F;
constexpr float sincof_p2 = -1.6666654611E-1F;
constexpr float coscof_p0 = 2.443315711809948E-005F;
constexpr float coscof_p1 = -1.388731625493765E-003F;
constexpr float coscof_p2 = 4.166664568298827E-002F;
constexpr float cephes_FOPI = 1.27323954473516F; // 4 / pi

constexpr float PI   = std::numbers::pi_v<float>;
constexpr float PIO2 = std::numbers::pi_v<float> / 2.0F;
constexpr float PIO4 = std::numbers::pi_v<float> / 4.0F;
constexpr float T3P8 = 2.41421356237309504880F; // tan(3*pi/8)
constexpr float MOREBITS      = 6.123233995736765886130E-17F;
constexpr float HALF_MOREBITS = 0.5F * 6.123233995736765886130E-17F;

constexpr float atan_P0 = -8.750608600031904122785E-1F;
constexpr float atan_P1 = -1.615753718733365076637E1F;
constexpr float atan_P2 = -7.500855792314704667340E1F;
constexpr float atan_P3 = -1.228866684490136173410E2F;
constexpr float atan_P4 = -6.485021904942025371773E1F;
/* atan_Q: leading coefficient 1.0 is implicit */
constexpr float atan_Q0 = 2.485846490142306297962E1F;
constexpr float atan_Q1 = 1.650270098316988542046E2F;
constexpr float atan_Q2 = 4.328810604912902668951E2F;
constexpr float atan_Q3 = 4.853903996359136964868E2F;
constexpr float atan_Q4 = 1.945506571482613964425E2F;

inline v16sf set1(float f) noexcept { return _mm512_set1_ps(f); }

constexpr float INF  = std::numeric_limits<float>::infinity();
constexpr float QNAN = std::numeric_limits<float>::quiet_NaN();

} // namespace detail


/* natural logarithm computed for 16 simultaneous floats
   log(x < 0) = NaN, log(+-0) = -inf, log(+inf) = +inf, log(NaN) = NaN
*/
inline v16sf log512_ps(v16sf x) noexcept {
    using namespace detail;
    const v16sf one  = set1(1.0F);
    const v16sf zero = _mm512_setzero_ps();

    const v16mask is_neg  = _mm512_cmp_ps_mask(x, zero, _CMP_LT_OQ);
    const v16mask is_zero = _mm512_cmp_ps_mask(x, zero, _CMP_EQ_OQ);
    const v16mask is_inf  = _mm512_cmp_ps_mask(x, set1(INF), _CMP_EQ_OQ);

    /* frexp(): x = m * 2^e with m in [0.5, 1).  getexp returns floor(log2(x)),
       which is one less than frexp's exponent.  Both handle subnormals. */
    v16sf e = _mm512_add_ps(_mm512_getexp_ps(x), one);
    v16sf m = _mm512_getmant_ps(x, _MM_MANT_NORM_p5_1, _MM_MANT_SIGN_zero);

    /* if( m < SQRTHF ) { e -= 1; x = m + m - 1.0; } else { x = m - 1.0; } */
    const v16mask small = _mm512_cmp_ps_mask(m, set1(cephes_SQRTHF), _CMP_LT_OQ);
    e = _mm512_mask_sub_ps(e, small, e, one);
    v16sf xr = _mm512_sub_ps(m, one);
    xr = _mm512_mask_add_ps(xr, small, xr, m);

    const v16sf z = _mm512_mul_ps(xr, xr);

    v16sf y = set1(cephes_log_p0);
    y = _mm512_fmadd_ps(y, xr, set1(cephes_log_p1));
    y = _mm512_fmadd_ps(y, xr, set1(cephes_log_p2));
    y = _mm512_fmadd_ps(y, xr, set1(cephes_log_p3));
    y = _mm512_fmadd_ps(y, xr, set1(cephes_log_p4));
    y = _mm512_fmadd_ps(y, xr, set1(cephes_log_p5));
    y = _mm512_fmadd_ps(y, xr, set1(cephes_log_p6));
    y = _mm512_fmadd_ps(y, xr, set1(cephes_log_p7));
    y = _mm512_fmadd_ps(y, xr, set1(cephes_log_p8));
    y = _mm512_mul_ps(y, xr);
    y = _mm512_mul_ps(y, z);

    y = _mm512_fmadd_ps(e, set1(cephes_log_q1), y);
    y = _mm512_fnmadd_ps(z, set1(0.5F), y);

    xr = _mm512_add_ps(xr, y);
    xr = _mm512_fmadd_ps(e, set1(cephes_log_q2), xr);

    /* special cases (NaN inputs already propagate through getexp/getmant) */
    xr = _mm512_mask_mov_ps(xr, is_inf,  x);
    xr = _mm512_mask_mov_ps(xr, is_zero, set1(-INF));
    xr = _mm512_mask_mov_ps(xr, is_neg,  set1(QNAN));
    return xr;
}

/* exp() computed for 16 simultaneous floats
   exp(+inf) = +inf, exp(-inf) = 0, exp(NaN) = NaN; overflow -> +inf,
   underflow -> subnormal / 0.
*/
inline v16sf exp512_ps(v16sf x) noexcept {
    using namespace detail;
    const v16sf one = set1(1.0F);

    /* Operand order matters: min/max return the *second* operand if either
       is NaN, so this keeps NaN inputs as NaN. */
    x = _mm512_min_ps(set1(exp_hi), x);
    x = _mm512_max_ps(set1(exp_lo), x);

    /* express exp(x) as exp(g + n*log(2)), n = floor(x*log2(e) + 0.5) */
    v16sf fx = _mm512_fmadd_ps(x, set1(cephes_LOG2EF), set1(0.5F));
    fx = _mm512_roundscale_ps(fx, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);

    x = _mm512_fnmadd_ps(fx, set1(cephes_exp_C1), x);
    x = _mm512_fnmadd_ps(fx, set1(cephes_exp_C2), x);

    const v16sf z = _mm512_mul_ps(x, x);

    v16sf y = set1(cephes_exp_p0);
    y = _mm512_fmadd_ps(y, x, set1(cephes_exp_p1));
    y = _mm512_fmadd_ps(y, x, set1(cephes_exp_p2));
    y = _mm512_fmadd_ps(y, x, set1(cephes_exp_p3));
    y = _mm512_fmadd_ps(y, x, set1(cephes_exp_p4));
    y = _mm512_fmadd_ps(y, x, set1(cephes_exp_p5));
    y = _mm512_fmadd_ps(y, z, x);
    y = _mm512_add_ps(y, one);

    /* y * 2^n -- scalef handles overflow to inf and gradual underflow */
    return _mm512_scalef_ps(y, fx);
}

namespace detail {

/* The two Cephes polynomials shared by sin, cos and sincos.
   x is the reduced argument in [-Pi/4, Pi/4], z = x*x. */
inline v16sf cos_poly(v16sf z) noexcept {
    v16sf y = set1(coscof_p0);
    y = _mm512_fmadd_ps(y, z, set1(coscof_p1));
    y = _mm512_fmadd_ps(y, z, set1(coscof_p2));
    y = _mm512_mul_ps(y, z);
    y = _mm512_mul_ps(y, z);
    y = _mm512_fnmadd_ps(z, set1(0.5F), y);
    return _mm512_add_ps(y, set1(1.0F));
}
inline v16sf sin_poly(v16sf x, v16sf z) noexcept {
    v16sf y = set1(sincof_p0);
    y = _mm512_fmadd_ps(y, z, set1(sincof_p1));
    y = _mm512_fmadd_ps(y, z, set1(sincof_p2));
    y = _mm512_mul_ps(y, z);
    return _mm512_fmadd_ps(y, x, x);
}

/* Cephes range reduction.  x must be non-negative.
   Returns the reduced argument; j receives the (even) octant index. */
inline v16sf reduce(v16sf x, v16si& j) noexcept {
    /* j = (int)(x * 4/Pi); j = (j+1) & ~1 */
    j = _mm512_cvttps_epi32(_mm512_mul_ps(x, set1(cephes_FOPI)));
    j = _mm512_add_epi32(j, _mm512_set1_epi32(1));
    j = _mm512_and_si512(j, _mm512_set1_epi32(~1));
    const v16sf y = _mm512_cvtepi32_ps(j);

    /* The magic pass: "Extended precision modular arithmetic"
       x = ((x - y * DP1) - y * DP2) - y * DP3; */
    x = _mm512_fmadd_ps(y, set1(minus_cephes_DP1), x);
    x = _mm512_fmadd_ps(y, set1(minus_cephes_DP2), x);
    x = _mm512_fmadd_ps(y, set1(minus_cephes_DP3), x);
    return x;
}

} // namespace detail

/* evaluation of 16 sines at once using AVX-512 intrinsics.
   Precision is excellent as long as |x| < 8192 (see avx_mathfun.h).
*/
inline v16sf sin512_ps(v16sf x) noexcept {
    using namespace detail;
    v16sf sign_bit = sign_bits(x);
    x = _mm512_abs_ps(x);

    v16si j;
    x = reduce(x, j);

    /* swap sign flag: bit 2 of j moved to the float sign bit */
    const v16si swap = _mm512_slli_epi32(_mm512_and_si512(j, _mm512_set1_epi32(4)), 29);
    sign_bit = xor_ps(sign_bit, _mm512_castsi512_ps(swap));
    /* polynomial selection: (j & 2) != 0 -> cosine polynomial */
    const v16mask use_cos = _mm512_test_epi32_mask(j, _mm512_set1_epi32(2));

    const v16sf z = _mm512_mul_ps(x, x);
    v16sf y = _mm512_mask_blend_ps(use_cos, sin_poly(x, z), cos_poly(z));
    return xor_ps(y, sign_bit);
}

/* almost the same as sin512_ps */
inline v16sf cos512_ps(v16sf x) noexcept {
    using namespace detail;
    x = _mm512_abs_ps(x);

    v16si j;
    x = reduce(x, j);
    j = _mm512_sub_epi32(j, _mm512_set1_epi32(2));

    const v16si sign = _mm512_slli_epi32(_mm512_andnot_si512(j, _mm512_set1_epi32(4)), 29);
    const v16mask use_cos = _mm512_test_epi32_mask(j, _mm512_set1_epi32(2));

    const v16sf z = _mm512_mul_ps(x, x);
    v16sf y = _mm512_mask_blend_ps(use_cos, sin_poly(x, z), cos_poly(z));
    return xor_ps(y, _mm512_castsi512_ps(sign));
}

/* sine and cosine together, for roughly the price of one of them */
inline void sincos512_ps(v16sf x, v16sf* s, v16sf* c) noexcept {
    using namespace detail;
    v16sf sign_bit_sin = sign_bits(x);
    x = _mm512_abs_ps(x);

    v16si j;
    x = reduce(x, j);

    const v16si swap_sin = _mm512_slli_epi32(_mm512_and_si512(j, _mm512_set1_epi32(4)), 29);
    sign_bit_sin = xor_ps(sign_bit_sin, _mm512_castsi512_ps(swap_sin));

    const v16si jc = _mm512_sub_epi32(j, _mm512_set1_epi32(2));
    const v16sf sign_bit_cos = _mm512_castsi512_ps(
        _mm512_slli_epi32(_mm512_andnot_si512(jc, _mm512_set1_epi32(4)), 29));

    /* (j & 2) != 0 -> the sine comes from the cosine polynomial and vice versa */
    const v16mask swap_poly = _mm512_test_epi32_mask(j, _mm512_set1_epi32(2));

    const v16sf z  = _mm512_mul_ps(x, x);
    const v16sf yc = cos_poly(z);
    const v16sf ys = sin_poly(x, z);

    *s = xor_ps(_mm512_mask_blend_ps(swap_poly, ys, yc), sign_bit_sin);
    *c = xor_ps(_mm512_mask_blend_ps(swap_poly, yc, ys), sign_bit_cos);
}

/**
 * Vectorized Cephes atanf for 16 floats.
 * Cephes Math Library Release 2.1:  December, 1988
 * Copyright 1984, 1987, 1988 by Stephen L. Moshier
 *
 * atan(+-0) = +-0, atan(+-inf) = +-Pi/2, atan(NaN) = NaN
 */
inline v16sf atan(v16sf x) noexcept {
    using namespace detail;
    const v16sf one = set1(1.0F);

    /* make argument positive and save the sign */
    const v16sf sign = sign_bits(x);
    x = _mm512_abs_ps(x);

    /* range reduction:
         x > tan(3*Pi/8)  : y = Pi/2, x = -1/x           (flag 1)
         x > 0.66         : y = Pi/4, x = (x-1)/(x+1)    (flag 2)
         otherwise        : y = 0                        (flag 0)
       NaN fails both comparisons and ends up in flag 2, which propagates it. */
    const v16mask flag1 = _mm512_cmp_ps_mask(x, set1(T3P8), _CMP_GT_OQ);
    const v16mask flag0 = _mm512_cmp_ps_mask(x, set1(0.66F), _CMP_LE_OQ);
    const v16mask flag2 = _kandn_mask16(_kor_mask16(flag0, flag1), static_cast<v16mask>(0xFFFF));

    v16sf y = _mm512_maskz_mov_ps(flag1, set1(PIO2));
    y = _mm512_mask_mov_ps(y, flag2, set1(PIO4));

    /* fold all three reductions into a single division: num / den */
    v16sf num = _mm512_mask_blend_ps(flag1, _mm512_sub_ps(x, one), set1(-1.0F));
    v16sf den = _mm512_mask_blend_ps(flag1, _mm512_add_ps(x, one), x);
    num = _mm512_mask_mov_ps(num, flag0, x);
    den = _mm512_mask_mov_ps(den, flag0, one);
    x = _mm512_div_ps(num, den);

    const v16sf z = _mm512_mul_ps(x, x);

    /* polevl(z, P, 4) */
    v16sf p = set1(atan_P0);
    p = _mm512_fmadd_ps(p, z, set1(atan_P1));
    p = _mm512_fmadd_ps(p, z, set1(atan_P2));
    p = _mm512_fmadd_ps(p, z, set1(atan_P3));
    p = _mm512_fmadd_ps(p, z, set1(atan_P4));
    /* p1evl(z, Q, 5) */
    v16sf q = _mm512_add_ps(z, set1(atan_Q0));
    q = _mm512_fmadd_ps(q, z, set1(atan_Q1));
    q = _mm512_fmadd_ps(q, z, set1(atan_Q2));
    q = _mm512_fmadd_ps(q, z, set1(atan_Q3));
    q = _mm512_fmadd_ps(q, z, set1(atan_Q4));

    /* z = x * (z * P(z)/Q(z)) + x */
    v16sf r = _mm512_div_ps(_mm512_mul_ps(z, p), q);
    r = _mm512_fmadd_ps(x, r, x);

    r = _mm512_mask_add_ps(r, flag1, r, set1(MOREBITS));
    r = _mm512_mask_add_ps(r, flag2, r, set1(HALF_MOREBITS));

    y = _mm512_add_ps(y, r);

    /* restore sign */
    return xor_ps(y, sign);
}

/**
 * Four quadrant arctangent of y/x for 16 floats; returns -Pi <= z <= Pi.
 *
 * Matches std::atan2 for all the special cases in C99 Annex F:
 * signed zeros, infinities and NaNs.
 *
 * @param y
 * @param x
 */
inline v16sf atan2(v16sf y, v16sf x) noexcept {
    using namespace detail;
    const v16sf ysign = sign_bits(y);
    const v16sf xsign = sign_bits(x);
    const v16sf qsign = xor_ps(ysign, xsign); // sign of y/x

    v16sf ratio = _mm512_div_ps(y, x);

    /* y == +-0 and x not NaN : y/x is a signed zero (also when x == +-0, where y/x is NaN) */
    const v16mask y_zero = _mm512_mask_cmp_ps_mask(_mm512_cmp_ps_mask(y, _mm512_setzero_ps(), _CMP_EQ_OQ),
                                                   x, x, _CMP_ORD_Q);
    ratio = _mm512_mask_mov_ps(ratio, y_zero, qsign);

    /* both infinite : inf/inf is NaN, but atan2 wants +-Pi/4 or +-3Pi/4 */
    const v16sf inf = set1(INF);
    const v16mask both_inf = _kand_mask16(_mm512_cmp_ps_mask(_mm512_abs_ps(y), inf, _CMP_EQ_OQ),
                                          _mm512_cmp_ps_mask(_mm512_abs_ps(x), inf, _CMP_EQ_OQ));
    ratio = _mm512_mask_mov_ps(ratio, both_inf, or_ps(set1(1.0F), qsign));

    v16sf z = atan(ratio);

    /* x negative (sign bit set, so this includes -0.0): add copysign(Pi, y) */
    const v16mask x_neg = sign_set_mask(x);
    z = _mm512_mask_add_ps(z, x_neg, z, or_ps(set1(PI), ysign));
    return z;
}

inline void printVec(const v16sf& v) noexcept {
    alignas(64) float vArray[16];
    _mm512_store_ps(vArray, v);
    std::cout << "[ ";
    for (std::size_t i = 0; i < 16; ++i) {
        std::cout << vArray[i] << " ";
    }
    std::cout << "]" << std::endl;
}

} // namespace avx512_mathfun

#endif // AVX512_MATHFUN_H
