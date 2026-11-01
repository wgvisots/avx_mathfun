/*
   AVX implementation of sin, cos, sincos, exp and log

   Based on "sse_mathfun.h", by Julien Pommier
   http://gruntthepeon.free.fr/ssemath/

   Copyright (C) 2012 Giovanni Garberoglio
   Interdisciplinary Laboratory for Computational Science (LISC)
   Fondazione Bruno Kessler and University of Trento
   via Sommarive, 18
   I-38123 Trento (Italy)

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
*/

#ifndef AVX_MATHFUN_H
#define AVX_MATHFUN_H

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <numbers>
#include <immintrin.h>

/*
  wgvisots: Rewritten for strict C++20 conformance (2026).  Constants are
  constexpr scalars broadcast with _mm256_set1_*, instead of aligned arrays
  read through casts to vector-pointer types; lanes are read with stores
  rather than by reinterpreting a vector's address; and the non-AVX2 integer
  path splits vectors with intrinsics instead of union type punning.  The
  floating-point operations, and their order, are unchanged, so the results
  are bit-for-bit identical to the previous version.
*/

namespace avx_mathfun{ //wgvisots

typedef __m256  v8sf; // vector of 8 float (avx)
typedef __m256i v8si; // vector of 8 int   (avx)
typedef __m128i v4si; // vector of 4 int   (sse2)

namespace detail {

/* Cephes constants */
constexpr float cephes_SQRTHF = 0.707106781186547524F;
constexpr float cephes_log_p0 = 7.0376836292E-2F;
constexpr float cephes_log_p1 = - 1.1514610310E-1F;
constexpr float cephes_log_p2 = 1.1676998740E-1F;
constexpr float cephes_log_p3 = - 1.2420140846E-1F;
constexpr float cephes_log_p4 = + 1.4249322787E-1F;
constexpr float cephes_log_p5 = - 1.6668057665E-1F;
constexpr float cephes_log_p6 = + 2.0000714765E-1F;
constexpr float cephes_log_p7 = - 2.4999993993E-1F;
constexpr float cephes_log_p8 = + 3.3333331174E-1F;
constexpr float cephes_log_q1 = -2.12194440e-4F;
constexpr float cephes_log_q2 = 0.693359375F;

// exp() input clamp, the same as avx512_mathfun.h: exp(89) overflows to +inf and exp(-104) rounds
// to 0, so clamping changes no result.
constexpr float exp_hi = 89.0F;
constexpr float exp_lo = -104.0F;
constexpr float cephes_LOG2EF = 1.44269504088896341F;
constexpr float cephes_exp_C1 = 0.693359375F;
constexpr float cephes_exp_C2 = -2.12194440e-4F;
constexpr float cephes_exp_p0 = 1.9875691500E-4F;
constexpr float cephes_exp_p1 = 1.3981999507E-3F;
constexpr float cephes_exp_p2 = 8.3334519073E-3F;
constexpr float cephes_exp_p3 = 4.1665795894E-2F;
constexpr float cephes_exp_p4 = 1.6666665459E-1F;
constexpr float cephes_exp_p5 = 5.0000001201E-1F;

constexpr float minus_cephes_DP1 = -0.78515625F;
constexpr float minus_cephes_DP2 = -2.4187564849853515625e-4F;
constexpr float minus_cephes_DP3 = -3.77489497744594108e-8F;
constexpr float sincof_p0 = -1.9515295891E-4F;
constexpr float sincof_p1 =  8.3321608736E-3F;
constexpr float sincof_p2 = -1.6666654611E-1F;
constexpr float coscof_p0 =  2.443315711809948E-005F;
constexpr float coscof_p1 = -1.388731625493765E-003F;
constexpr float coscof_p2 =  4.166664568298827E-002F;
constexpr float cephes_FOPI = 1.27323954473516F; // 4 / pi

constexpr float PI   = std::numbers::pi_v<float>;
constexpr float PIO2 = std::numbers::pi_v<float> / 2.0F; // exact: scaling by a power of two
constexpr float PIO4 = std::numbers::pi_v<float> / 4.0F;

//tan(3*pi/8)
//NOTE: Cephes defines T3P8 like this for Intel/IEEE:
//static unsigned short T3P8A[] = {0x9de6,0x333f,0x504f,0x4003};
//#define T3P8 *(double *)T3P8A
//And like this for UNK:
//NOTE: we lose precision calling these floats rather than the original double
constexpr float T3P8          = 2.41421356237309504880F;
constexpr float MOREBITS      = 6.123233995736765886130E-17F;
constexpr float HALF_MOREBITS = static_cast<float>(6.123233995736765886130E-17 * 0.5);
constexpr float ATAN_SPLIT    = 0.66F;

/* bit patterns */
constexpr int sign_mask     = static_cast<int>(0x80000000U);
constexpr int inv_sign_mask = 0x7FFFFFFF;
constexpr int inv_mant_mask = ~0x7f800000;

inline v8sf set1(float f) noexcept { return _mm256_set1_ps(f); }
inline v8si set1i(int i) noexcept { return _mm256_set1_epi32(i); }
inline v8sf bits(int i) noexcept { return _mm256_castsi256_ps(_mm256_set1_epi32(i)); }

/* a * b + c: one fused instruction where FMA is available, a separate multiply and add otherwise
   (AVX-only CPUs such as Sandy Bridge have no FMA). */
inline v8sf fmadd(v8sf a, v8sf b, v8sf c) noexcept {
#ifdef __FMA__
  return _mm256_fmadd_ps(a, b, c);
#else
  return _mm256_add_ps(_mm256_mul_ps(a, b), c);
#endif
}

/*
  32-bit integer operations on 256-bit vectors.  These are single AVX2
  instructions; on AVX-only CPUs they are done as two SSE2 operations on the
  128-bit halves.
*/
#ifdef __AVX2__
inline v8si add_epi32(v8si a, v8si b) noexcept { return _mm256_add_epi32(a, b); }
inline v8si sub_epi32(v8si a, v8si b) noexcept { return _mm256_sub_epi32(a, b); }
inline v8si and_si256(v8si a, v8si b) noexcept { return _mm256_and_si256(a, b); }
inline v8si andnot_si256(v8si a, v8si b) noexcept { return _mm256_andnot_si256(a, b); }
inline v8si cmpeq_epi32(v8si a, v8si b) noexcept { return _mm256_cmpeq_epi32(a, b); }
template <int N> inline v8si slli_epi32(v8si a) noexcept { return _mm256_slli_epi32(a, N); }
template <int N> inline v8si srli_epi32(v8si a) noexcept { return _mm256_srli_epi32(a, N); }
template <int N> inline v8si srai_epi32(v8si a) noexcept { return _mm256_srai_epi32(a, N); }
#else
inline v4si lo128(v8si a) noexcept { return _mm256_castsi256_si128(a); }
inline v4si hi128(v8si a) noexcept { return _mm256_extractf128_si256(a, 1); }
inline v8si combine(v4si lo, v4si hi) noexcept { return _mm256_insertf128_si256(_mm256_castsi128_si256(lo), hi, 1); }

inline v8si add_epi32(v8si a, v8si b) noexcept {
  return combine(_mm_add_epi32(lo128(a), lo128(b)), _mm_add_epi32(hi128(a), hi128(b)));
}
inline v8si sub_epi32(v8si a, v8si b) noexcept {
  return combine(_mm_sub_epi32(lo128(a), lo128(b)), _mm_sub_epi32(hi128(a), hi128(b)));
}
inline v8si cmpeq_epi32(v8si a, v8si b) noexcept {
  return combine(_mm_cmpeq_epi32(lo128(a), lo128(b)), _mm_cmpeq_epi32(hi128(a), hi128(b)));
}
// bitwise ops can use the AVX float-domain instructions
inline v8si and_si256(v8si a, v8si b) noexcept {
  return _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
}
inline v8si andnot_si256(v8si a, v8si b) noexcept {
  return _mm256_castps_si256(_mm256_andnot_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
}
template <int N> inline v8si slli_epi32(v8si a) noexcept {
  return combine(_mm_slli_epi32(lo128(a), N), _mm_slli_epi32(hi128(a), N));
}
template <int N> inline v8si srli_epi32(v8si a) noexcept {
  return combine(_mm_srli_epi32(lo128(a), N), _mm_srli_epi32(hi128(a), N));
}
template <int N> inline v8si srai_epi32(v8si a) noexcept {
  return combine(_mm_srai_epi32(lo128(a), N), _mm_srai_epi32(hi128(a), N));
}
#endif /* __AVX2__ */

} // namespace detail


/* natural logarithm computed for 8 simultaneous floats
   log(x < 0) = NaN, log(+-0) = -inf, log(+inf) = +inf, log(NaN) = NaN; subnormals are handled
   accurately.  Same results as log512_ps in avx512_mathfun.h.
*/
//wgvisots - added inline
inline v8sf log256_ps(v8sf x) noexcept {
  using namespace detail;
  v8si imm0;
  const v8sf one   = set1(1.0F);
  const v8sf input = x;

  /* subnormals: scale by 2^24 into the normal range, and take 24 off the exponent below */
  const v8sf subnormal = _mm256_and_ps(_mm256_cmp_ps(x, set1(std::numeric_limits<float>::min()), _CMP_LT_OQ),
                                       _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_GT_OQ));
  x = _mm256_blendv_ps(x, _mm256_mul_ps(x, set1(16777216.0F)), subnormal);

  x = _mm256_max_ps(x, set1(std::numeric_limits<float>::min()));  /* zero, negatives and NaN are fixed up at the end */

  imm0 = srli_epi32<23>(_mm256_castps_si256(x));

  /* keep only the fractional part */
  x = _mm256_and_ps(x, bits(inv_mant_mask));
  x = _mm256_or_ps(x, set1(0.5F));

  imm0 = sub_epi32(imm0, set1i(0x7f));
  v8sf e = _mm256_cvtepi32_ps(imm0);

  e = _mm256_add_ps(e, one);
  e = _mm256_sub_ps(e, _mm256_and_ps(subnormal, set1(24.0F)));

  /* part2:
     if( x < SQRTHF ) {
       e -= 1;
       x = x + x - 1.0;
     } else { x = x - 1.0; }
  */
  v8sf mask = _mm256_cmp_ps(x, set1(cephes_SQRTHF), _CMP_LT_OS);
  v8sf tmp = _mm256_and_ps(x, mask);
  x = _mm256_sub_ps(x, one);
  e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
  x = _mm256_add_ps(x, tmp);

  v8sf z = _mm256_mul_ps(x,x);

  v8sf y = set1(cephes_log_p0);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, set1(cephes_log_p1));
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, set1(cephes_log_p2));
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, set1(cephes_log_p3));
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, set1(cephes_log_p4));
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, set1(cephes_log_p5));
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, set1(cephes_log_p6));
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, set1(cephes_log_p7));
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, set1(cephes_log_p8));
  y = _mm256_mul_ps(y, x);

  y = _mm256_mul_ps(y, z);

  tmp = _mm256_mul_ps(e, set1(cephes_log_q1));
  y = _mm256_add_ps(y, tmp);

  tmp = _mm256_mul_ps(z, set1(0.5F));
  y = _mm256_sub_ps(y, tmp);

  tmp = _mm256_mul_ps(e, set1(cephes_log_q2));
  x = _mm256_add_ps(x, y);
  x = _mm256_add_ps(x, tmp);

  /* special values (C99 Annex F) */
  x = _mm256_blendv_ps(x, set1(-std::numeric_limits<float>::infinity()), _mm256_cmp_ps(input, _mm256_setzero_ps(), _CMP_EQ_OQ));
  x = _mm256_blendv_ps(x, set1(std::numeric_limits<float>::infinity()), _mm256_cmp_ps(input, set1(std::numeric_limits<float>::infinity()), _CMP_EQ_OQ));
  x = _mm256_blendv_ps(x, set1(std::numeric_limits<float>::quiet_NaN()), _mm256_cmp_ps(input, _mm256_setzero_ps(), _CMP_NGE_UQ)); /* x < 0 or NaN */
  return x;
}

/* exp() computed for 8 simultaneous floats
   exp(+-0) = 1, exp(+inf) = +inf, exp(-inf) = 0, exp(NaN) = NaN; results overflow to +inf and
   underflow through the subnormals to 0, as in C.  Same results as exp512_ps in avx512_mathfun.h.
*/
//wgvisots - added inline
inline v8sf exp256_ps(v8sf x) noexcept {
  using namespace detail;
  v8sf tmp, fx;
  v8si imm0;
  const v8sf one   = set1(1.0F);
  const v8sf input = x;

  x = _mm256_min_ps(x, set1(exp_hi));
  x = _mm256_max_ps(x, set1(exp_lo));

  /* express exp(x) as exp(g + n*log(2)) */
  fx = _mm256_mul_ps(x, set1(cephes_LOG2EF));
  fx = _mm256_add_ps(fx, set1(0.5F));

  tmp = _mm256_floor_ps(fx);

  /* if greater, substract 1 */
  v8sf mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
  mask = _mm256_and_ps(mask, one);
  fx = _mm256_sub_ps(tmp, mask);

  tmp = _mm256_mul_ps(fx, set1(cephes_exp_C1));
  v8sf z = _mm256_mul_ps(fx, set1(cephes_exp_C2));
  x = _mm256_sub_ps(x, tmp);
  x = _mm256_sub_ps(x, z);

  z = _mm256_mul_ps(x,x);

  v8sf y = set1(cephes_exp_p0);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, set1(cephes_exp_p1));
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, set1(cephes_exp_p2));
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, set1(cephes_exp_p3));
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, set1(cephes_exp_p4));
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, set1(cephes_exp_p5));
  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, x);
  y = _mm256_add_ps(y, one);

  /* multiply by 2^n.  n is in [-150, 128], where 2^n is not always a normal float, so apply it in
     two halves that are: y * 2^n1 is exact, and the multiply by 2^n2 rounds once, overflowing to
     +inf or rounding to a subnormal (or 0) exactly as a correctly rounded exp would. */
  imm0 = _mm256_cvttps_epi32(fx);
  const v8si n1 = srai_epi32<1>(imm0);
  const v8si n2 = sub_epi32(imm0, n1);
  const v8sf pow2n1 = _mm256_castsi256_ps(slli_epi32<23>(add_epi32(n1, set1i(0x7f))));
  const v8sf pow2n2 = _mm256_castsi256_ps(slli_epi32<23>(add_epi32(n2, set1i(0x7f))));
  y = _mm256_mul_ps(_mm256_mul_ps(y, pow2n1), pow2n2);

  /* NaN in, NaN out (the clamps above replaced it with a number) */
  y = _mm256_blendv_ps(y, input, _mm256_cmp_ps(input, input, _CMP_UNORD_Q));
  return y;
}

namespace detail {

/* The magic pass: "Extended precision modular arithmetic"
   x = ((x - y * DP1) - y * DP2) - y * DP3; */
inline v8sf cephes_reduce(v8sf x, v8sf y) noexcept {
  v8sf xmm1 = _mm256_mul_ps(y, set1(minus_cephes_DP1));
  v8sf xmm2 = _mm256_mul_ps(y, set1(minus_cephes_DP2));
  v8sf xmm3 = _mm256_mul_ps(y, set1(minus_cephes_DP3));
  x = _mm256_add_ps(x, xmm1);
  x = _mm256_add_ps(x, xmm2);
  x = _mm256_add_ps(x, xmm3);
  return x;
}

/* first polynom (0 <= x <= Pi/4), z = x*x */
inline v8sf cephes_cos_poly(v8sf z) noexcept {
  v8sf y = set1(coscof_p0);
  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, set1(coscof_p1));
  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, set1(coscof_p2));
  y = _mm256_mul_ps(y, z);
  y = _mm256_mul_ps(y, z);
  v8sf tmp = _mm256_mul_ps(z, set1(0.5F));
  y = _mm256_sub_ps(y, tmp);
  y = _mm256_add_ps(y, set1(1.0F));
  return y;
}

/* second polynom (Pi/4 <= x <= 0), z = x*x */
inline v8sf cephes_sin_poly(v8sf x, v8sf z) noexcept {
  v8sf y2 = set1(sincof_p0);
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_add_ps(y2, set1(sincof_p1));
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_add_ps(y2, set1(sincof_p2));
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_mul_ps(y2, x);
  y2 = _mm256_add_ps(y2, x);
  return y2;
}

} // namespace detail

/* evaluation of 8 sines at onces using AVX intrisics

   The code is the exact rewriting of the cephes sinf function.
   Precision is excellent as long as x < 8192 (I did not bother to
   take into account the special handling they have for greater values
   -- it does not return garbage for arguments over 8192, though, but
   the extra precision is missing).

   Note that it is such that sinf((float)M_PI) = 8.74e-8, which is the
   surprising but correct result.

*/
//wgvisots - added inline
inline v8sf sin256_ps(v8sf x) noexcept { // any x
  using namespace detail;
  v8sf sign_bit, y;
  v8si imm0, imm2;

  sign_bit = x;
  /* take the absolute value */
  x = _mm256_and_ps(x, bits(inv_sign_mask));
  /* extract the sign bit (upper one) */
  sign_bit = _mm256_and_ps(sign_bit, bits(sign_mask));

  /* scale by 4/Pi */
  y = _mm256_mul_ps(x, set1(cephes_FOPI));

  /* store the integer part of y in mm0 */
  imm2 = _mm256_cvttps_epi32(y);
  /* j=(j+1) & (~1) (see the cephes sources) */
  imm2 = add_epi32(imm2, set1i(1));
  imm2 = and_si256(imm2, set1i(~1));
  y = _mm256_cvtepi32_ps(imm2);

  /* get the swap sign flag */
  imm0 = and_si256(imm2, set1i(4));
  imm0 = slli_epi32<29>(imm0);
  /* get the polynom selection mask
     there is one polynom for 0 <= x <= Pi/4
     and another one for Pi/4<x<=Pi/2

     Both branches will be computed.
  */
  imm2 = and_si256(imm2, set1i(2));
  imm2 = cmpeq_epi32(imm2, _mm256_setzero_si256());

  v8sf swap_sign_bit = _mm256_castsi256_ps(imm0);
  v8sf poly_mask = _mm256_castsi256_ps(imm2);
  sign_bit = _mm256_xor_ps(sign_bit, swap_sign_bit);

  x = cephes_reduce(x, y);

  v8sf z = _mm256_mul_ps(x,x);
  y = cephes_cos_poly(z);
  v8sf y2 = cephes_sin_poly(x, z);

  /* select the correct result from the two polynoms */
  y2 = _mm256_and_ps(poly_mask, y2);
  y = _mm256_andnot_ps(poly_mask, y);
  y = _mm256_add_ps(y,y2);
  /* update the sign */
  y = _mm256_xor_ps(y, sign_bit);

  return y;
}

/* almost the same as sin_ps */
//wgvisots - added inline
inline v8sf cos256_ps(v8sf x) noexcept { // any x
  using namespace detail;
  v8sf y;
  v8si imm0, imm2;

  /* take the absolute value */
  x = _mm256_and_ps(x, bits(inv_sign_mask));

  /* scale by 4/Pi */
  y = _mm256_mul_ps(x, set1(cephes_FOPI));

  /* store the integer part of y in mm0 */
  imm2 = _mm256_cvttps_epi32(y);
  /* j=(j+1) & (~1) (see the cephes sources) */
  imm2 = add_epi32(imm2, set1i(1));
  imm2 = and_si256(imm2, set1i(~1));
  y = _mm256_cvtepi32_ps(imm2);
  imm2 = sub_epi32(imm2, set1i(2));

  /* get the swap sign flag */
  imm0 = andnot_si256(imm2, set1i(4));
  imm0 = slli_epi32<29>(imm0);
  /* get the polynom selection mask */
  imm2 = and_si256(imm2, set1i(2));
  imm2 = cmpeq_epi32(imm2, _mm256_setzero_si256());

  v8sf sign_bit = _mm256_castsi256_ps(imm0);
  v8sf poly_mask = _mm256_castsi256_ps(imm2);

  x = cephes_reduce(x, y);

  v8sf z = _mm256_mul_ps(x,x);
  y = cephes_cos_poly(z);
  v8sf y2 = cephes_sin_poly(x, z);

  /* select the correct result from the two polynoms */
  y2 = _mm256_and_ps(poly_mask, y2);
  y = _mm256_andnot_ps(poly_mask, y);
  y = _mm256_add_ps(y,y2);
  /* update the sign */
  y = _mm256_xor_ps(y, sign_bit);

  return y;
}

/* since sin256_ps and cos256_ps are almost identical, sincos256_ps could replace both of them..
   it is almost as fast, and gives you a free cosine with your sine */
//wgvisots - added inline
inline void sincos256_ps(v8sf x, v8sf *s, v8sf *c) noexcept {
  using namespace detail;
  v8sf sign_bit_sin, y;
  v8si imm0, imm2, imm4;

  sign_bit_sin = x;
  /* take the absolute value */
  x = _mm256_and_ps(x, bits(inv_sign_mask));
  /* extract the sign bit (upper one) */
  sign_bit_sin = _mm256_and_ps(sign_bit_sin, bits(sign_mask));

  /* scale by 4/Pi */
  y = _mm256_mul_ps(x, set1(cephes_FOPI));

  /* store the integer part of y in imm2 */
  imm2 = _mm256_cvttps_epi32(y);

  /* j=(j+1) & (~1) (see the cephes sources) */
  imm2 = add_epi32(imm2, set1i(1));
  imm2 = and_si256(imm2, set1i(~1));

  y = _mm256_cvtepi32_ps(imm2);
  imm4 = imm2;

  /* get the swap sign flag for the sine */
  imm0 = and_si256(imm2, set1i(4));
  imm0 = slli_epi32<29>(imm0);

  /* get the polynom selection mask for the sine*/
  imm2 = and_si256(imm2, set1i(2));
  imm2 = cmpeq_epi32(imm2, _mm256_setzero_si256());

  v8sf swap_sign_bit_sin = _mm256_castsi256_ps(imm0);
  v8sf poly_mask = _mm256_castsi256_ps(imm2);

  x = cephes_reduce(x, y);

  imm4 = sub_epi32(imm4, set1i(2));
  imm4 = andnot_si256(imm4, set1i(4));
  imm4 = slli_epi32<29>(imm4);

  v8sf sign_bit_cos = _mm256_castsi256_ps(imm4);

  sign_bit_sin = _mm256_xor_ps(sign_bit_sin, swap_sign_bit_sin);

  v8sf z = _mm256_mul_ps(x,x);
  y = cephes_cos_poly(z);
  v8sf y2 = cephes_sin_poly(x, z);

  /* select the correct result from the two polynoms */
  v8sf ysin2 = _mm256_and_ps(poly_mask, y2);
  v8sf ysin1 = _mm256_andnot_ps(poly_mask, y);
  y2 = _mm256_sub_ps(y2,ysin2);
  y = _mm256_sub_ps(y, ysin1);

  v8sf xmm1 = _mm256_add_ps(ysin1,ysin2);
  v8sf xmm2 = _mm256_add_ps(y,y2);

  /* update the sign */
  *s = _mm256_xor_ps(xmm1, sign_bit_sin);
  *c = _mm256_xor_ps(xmm2, sign_bit_cos);
}

/**
 * Reference (non-vectorized) atan2: std::atan2 applied to each lane.
 * Useful as a baseline for accuracy and speed comparisons.
 *
 * @param y
 * @param x
 * @param c receives atan2(y[i], x[i]) for each lane i; -pi <= c <= pi
 */
//wgvisots
inline void atan2(const v8sf &y, const v8sf &x, v8sf &c) noexcept {
    alignas(32) float yArray[8];
    alignas(32) float xArray[8];
    alignas(32) float cArray[8];
    _mm256_store_ps(yArray, y);
    _mm256_store_ps(xArray, x);
    for (std::size_t i = 0; i < 8; ++i) {
        cArray[i] = std::atan2(yArray[i], xArray[i]);
    }
    c = _mm256_load_ps(cArray);
}

/**
 * Reference (non-vectorized) atan: std::atan applied to each lane.
 *
 * @param a
 * @param c receives atan(a[i]) for each lane i
 */
inline void atan(const v8sf &a, v8sf &c) noexcept {
    alignas(32) float aArray[8];
    alignas(32) float cArray[8];
    _mm256_store_ps(aArray, a);
    for (std::size_t i = 0; i < 8; ++i) {
        cArray[i] = std::atan(aArray[i]);
    }
    c = _mm256_load_ps(cArray);
}

inline void printVec(const v8sf& v) noexcept {
    alignas(32) float vArray[8];
    _mm256_store_ps(vArray, v);
    std::cout << "[ ";
    for (std::size_t i = 0; i < 8; ++i) {
        std::cout << vArray[i] << " ";
    }
    std::cout << "]" << std::endl;
}

//NOTE: we lose precision calling these floats rather than the original double
inline constexpr float atan_P[5] = {
    -8.750608600031904122785E-1F,
    -1.615753718733365076637E1F,
    -7.500855792314704667340E1F,
    -1.228866684490136173410E2F,
    -6.485021904942025371773E1F,
    };
inline constexpr float atan_Q[5] = {
    /* 1.000000000000000000000E0, */
    2.485846490142306297962E1F,
    1.650270098316988542046E2F,
    4.328810604912902668951E2F,
    4.853903996359136964868E2F,
    1.945506571482613964425E2F,
   };

/**
 * Vectorized version of polevl from Cephes.
 * Original copyright:
 * Cephes Math Library Release 2.1:  December, 1988
 * Copyright 1984, 1987, 1988 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 *
 * @param x
 * @param coef N+1 coefficients, highest order first
 * @param N    degree, must be >= 1
 * @return
 */
inline v8sf polevl(const v8sf& x, const float* coef, const std::size_t& N) noexcept {
    const float* p = coef;
    v8sf ans = _mm256_broadcast_ss(p++);
    std::size_t i = N;
    do {
        v8sf pVec = _mm256_broadcast_ss(p++);
        ans = detail::fmadd(ans, x, pVec);
    }while (--i);
    return ans;
}
/**
 * As polevl, but with an implicit leading coefficient of 1.0.
 *
 * @param coef N coefficients
 * @param N    degree, must be >= 2
 */
inline v8sf p1evl(const v8sf& x, const float* coef, const std::size_t& N ) noexcept {
    const float* p = coef;
    std::size_t i = N-1;

    v8sf ans = _mm256_add_ps(x, _mm256_broadcast_ss(p++));

    do
        ans = detail::fmadd(ans, x, _mm256_broadcast_ss(p++));
    while( --i );

    return ans;
}

/**
 * Vectorized Cephes atanf.  Also used by atan2 below.
 *
 * atan(+-0) = +-0, atan(+-inf) = +-pi/2, atan(NaN) = NaN
 *
 * @param ratio y/x
 * @return
 */
inline v8sf atan(v8sf ratio) noexcept {
    using namespace detail;
    //make argument positive and save the sign bit (keeping the sign bit, rather than testing x < 0,
    //means atan(-0.0) == -0.0, which atan2 relies on)
    const v8sf signMask = bits(sign_mask);
    const v8sf sign = _mm256_and_ps(signMask, ratio);
    ratio = _mm256_andnot_ps(signMask, ratio);

    // if( x > T3P8 ){
    //     y = PIO2;
    //     flag = 1;
    //     x = -( 1.0/x );
    // }
    const v8sf mask1 = _mm256_cmp_ps(ratio, set1(T3P8), _CMP_GT_OQ);
    v8sf y = _mm256_and_ps(mask1, set1(PIO2));
    ratio = _mm256_or_ps(_mm256_andnot_ps(mask1, ratio),
                         _mm256_and_ps(mask1, _mm256_div_ps(set1(-1.0F), ratio)));

    //else if( x <= 0.66 ){
    //  y = 0.0;
    //}
    //if(x <= 0.66) y = 0.0 /*we get the assignment for free*/
    const v8sf mask2 = _mm256_cmp_ps(ratio, set1(ATAN_SPLIT), _CMP_LE_OQ);

    //else{ /* x > 0.66 */
    //  y = PIO4;
    //  flag = 2;
    //  x = (x-1.0)/(x+1.0);
    //}
    y = _mm256_or_ps(y, _mm256_andnot_ps(mask2, set1(PIO4)));
    //x = (x-1.0)/(x+1.0);
    const v8sf newx = _mm256_div_ps(_mm256_sub_ps(ratio, set1(1.0F)),
                                    _mm256_add_ps(ratio, set1(1.0F)));
    const v8sf mask = _mm256_or_ps(mask1, mask2);

    ratio = _mm256_or_ps(_mm256_and_ps(ratio, mask),
                         _mm256_andnot_ps(mask, newx));

    //z = x * x;
    v8sf z = _mm256_mul_ps(ratio, ratio);

    //z = z * polevl( z, P, 4 ) / p1evl( z, Q, 5 );
    z = _mm256_div_ps(_mm256_mul_ps(z, polevl(z, atan_P, 4)),
                      p1evl(z, atan_Q, 5));

    //z = x * z + x
    z = detail::fmadd(ratio, z, ratio);

    //Cephes: flag == 2 (x > 0.66, not > T3P8) gets 0.5*MOREBITS; flag == 0 gets nothing.
    //(mask2 also covers the flag == 1 lanes, whose ratio is now negative, so use ~mask.)
    z = _mm256_add_ps(z, _mm256_andnot_ps(mask, set1(HALF_MOREBITS)));
    z = _mm256_add_ps(z, _mm256_and_ps(mask1, set1(MOREBITS)));

    y = _mm256_add_ps(y, z);

    //restore sign
    return _mm256_xor_ps(y, sign);
}

/**
 * Four quadrant arctangent of y/x; returns -PI <= z <= PI (ANSI C like).
 *
 * Matches std::atan2 for the C99 Annex F special cases: signed zeros,
 * infinities and NaNs.
 *
 * Approach: z = atan(y/x), plus copysign(PI, y) when x is negative (sign bit
 * set, so -0.0 counts as negative).  Two cases where y/x is NaN are patched
 * first: y == +-0 (y/x is replaced by a correctly signed zero) and both
 * arguments infinite (y/x is replaced by +-1).
 *
 * @param y
 * @param x
 */
inline v8sf atan2(v8sf y, v8sf x) noexcept {
    using namespace detail;
    const v8sf signMask = bits(sign_mask);
    const v8sf inf      = set1(std::numeric_limits<float>::infinity());

    const v8sf ysign = _mm256_and_ps(signMask, y);
    const v8sf xsign = _mm256_and_ps(signMask, x);
    const v8sf qsign = _mm256_xor_ps(ysign, xsign); // sign of y/x

    v8sf ratio = _mm256_div_ps(y, x);

    //y == +-0 (and x not NaN): y/x is a signed zero (also when x == +-0, where y/x would be NaN)
    const v8sf y_zero = _mm256_and_ps(_mm256_cmp_ps(y, _mm256_setzero_ps(), _CMP_EQ_OQ),
                                      _mm256_cmp_ps(x, x, _CMP_ORD_Q));
    ratio = _mm256_blendv_ps(ratio, qsign, y_zero);

    //both infinite: inf/inf is NaN, but atan2 wants +-PI/4 or +-3PI/4
    const v8sf both_inf = _mm256_and_ps(_mm256_cmp_ps(_mm256_andnot_ps(signMask, y), inf, _CMP_EQ_OQ),
                                        _mm256_cmp_ps(_mm256_andnot_ps(signMask, x), inf, _CMP_EQ_OQ));
    ratio = _mm256_blendv_ps(ratio, _mm256_or_ps(set1(1.0F), qsign), both_inf);

    v8sf z = atan(ratio);

    //x negative: add copysign(PI, y).  blendv selects on the sign bit of its mask, so passing
    //x directly picks every lane with x's sign bit set, including -0.0.
    const v8sf shifted = _mm256_add_ps(z, _mm256_or_ps(set1(PI), ysign));
    z = _mm256_blendv_ps(z, shifted, x);
    return z;
}

} // namespace avx_mathfun

#endif // AVX_MATHFUN_H
