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

#pragma once /*wgvisots*/

#include <cstring>
#include <iostream>
#include <immintrin.h>

#include "Aligned.h" //wgvisots - this is to provide ALIGN32_BEG/ALIGN32_end


/* yes I know, the top of this file is quite ugly */

namespace avx_mathfun{ //wgvisots

/* __m128 is ugly to write */
typedef __m256  v8sf; // vector of 8 float (avx)
typedef __m256i v8si; // vector of 8 int   (avx)
typedef __m128i v4si; // vector of 8 int   (avx)

#define _PI32AVX_CONST(Name, Val)                                            \
  static const int _pi32avx_##Name[4] ALIGN32_END = { Val, Val, Val, Val }

_PI32AVX_CONST(1, 1);
_PI32AVX_CONST(inv1, ~1);
_PI32AVX_CONST(2, 2);
_PI32AVX_CONST(4, 4);


/* declare some AVX constants -- why can't I figure a better way to do that? */
#define _PS256_CONST(Name, Val)                                            \
  static const float _ps256_##Name[8] ALIGN32_END = { Val, Val, Val, Val, Val, Val, Val, Val }
#define _PI32_CONST256(Name, Val)                                            \
  static const int _pi32_256_##Name[8] ALIGN32_END = { Val, Val, Val, Val, Val, Val, Val, Val }
#define _PS256_CONST_TYPE(Name, Type, Val)                                 \
  static const Type _ps256_##Name[8] ALIGN32_END = { Val, Val, Val, Val, Val, Val, Val, Val }

_PS256_CONST(1  , 1.0f);
_PS256_CONST(neg1, -1.0f);
_PS256_CONST(0p5, 0.5f);
_PS256_CONST(2  , 2.0f);
_PS256_CONST(4  , 4.0f);
/* the smallest non denormalized float number */
_PS256_CONST_TYPE(min_norm_pos, int, 0x00800000);
_PS256_CONST_TYPE(mant_mask, int, 0x7f800000);
_PS256_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);

_PS256_CONST_TYPE(sign_mask, int, (int)0x80000000);
_PS256_CONST_TYPE(inv_sign_mask, int, ~0x80000000);

_PI32_CONST256(0, 0);
_PI32_CONST256(1, 1);
_PI32_CONST256(inv1, ~1);
_PI32_CONST256(2, 2);
_PI32_CONST256(3, 3);
_PI32_CONST256(4, 4);
_PI32_CONST256(0x7f, 0x7f);
_PI32_CONST256(fp_sign_mask, ~0x7FFFFFFF);

_PS256_CONST(cephes_SQRTHF, 0.707106781186547524F);
_PS256_CONST(cephes_log_p0, 7.0376836292E-2F);
_PS256_CONST(cephes_log_p1, - 1.1514610310E-1F);
_PS256_CONST(cephes_log_p2, 1.1676998740E-1F);
_PS256_CONST(cephes_log_p3, - 1.2420140846E-1F);
_PS256_CONST(cephes_log_p4, + 1.4249322787E-1F);
_PS256_CONST(cephes_log_p5, - 1.6668057665E-1F);
_PS256_CONST(cephes_log_p6, + 2.0000714765E-1F);
_PS256_CONST(cephes_log_p7, - 2.4999993993E-1F);
_PS256_CONST(cephes_log_p8, + 3.3333331174E-1F);
_PS256_CONST(cephes_log_q1, -2.12194440e-4F);
_PS256_CONST(cephes_log_q2, 0.693359375F);

_PS256_CONST(infinity,           INFINITY);
_PS256_CONST(negative_infinity, -INFINITY);
_PS256_CONST(zero,               0.0F);
_PS256_CONST(negativeZero,      -0.0F);
_PS256_CONST(PI,                 M_PI);
_PS256_CONST(negPI,             -M_PI);
_PS256_CONST(PIO2,               M_PI_2);
_PS256_CONST(negPIO2,           -M_PI_2);
_PS256_CONST(PIO4,               M_PI_4);

//tan(3*pi/8)
//NOTE: Cephes defines T3P8 like this for Intel/IEEE:
//static unsigned short T3P8A[] = {0x9de6,0x333f,0x504f,0x4003};
//#define T3P8 *(double *)T3P8A
//And like this for UNK:
//NOTE: we lose precision calling these floats rather than the original double
_PS256_CONST(T3P8,          2.41421356237309504880);
_PS256_CONST(HALF_MOREBITS, 6.123233995736765886130E-17 * 0.5);
_PS256_CONST(MOREBITS,      6.123233995736765886130E-17);

_PS256_CONST(0_66,  0.66);

//NOTE: we lose precision calling these floats rather than the original double
static float atan_P[5] = {
    -8.750608600031904122785E-1,
    -1.615753718733365076637E1,
    -7.500855792314704667340E1,
    -1.228866684490136173410E2,
    -6.485021904942025371773E1,
    };
static float atan_Q[5] = {
    /* 1.000000000000000000000E0, */
    2.485846490142306297962E1,
    1.650270098316988542046E2,
    4.328810604912902668951E2,
    4.853903996359136964868E2,
    1.945506571482613964425E2,
   };


#ifndef __AVX2__

typedef union imm_xmm_union {
  v8si imm;
  v4si xmm[2];
} imm_xmm_union;

#define COPY_IMM_TO_XMM(imm_, xmm0_, xmm1_) {    \
    imm_xmm_union u ALIGN32_END;  \
    u.imm = imm_;                  \
    xmm0_ = u.xmm[0];                            \
    xmm1_ = u.xmm[1];                            \
}

#define COPY_XMM_TO_IMM(xmm0_, xmm1_, imm_) {                       \
    imm_xmm_union u ALIGN32_END; \
    u.xmm[0]=xmm0_; u.xmm[1]=xmm1_; imm_ = u.imm; \
  }


#define AVX2_BITOP_USING_SSE2(fn) \
static inline v8si avx2_mm256_##fn(v8si x, int a) \
{ \
  /* use SSE2 instruction to perform the bitop AVX2 */ \
  v4si x1, x2; \
  v8si ret; \
  COPY_IMM_TO_XMM(x, x1, x2); \
  x1 = _mm_##fn(x1,a); \
  x2 = _mm_##fn(x2,a); \
  COPY_XMM_TO_IMM(x1, x2, ret); \
  return(ret); \
}

//#warning "Using SSE2 to perform AVX2 bitshift ops"
AVX2_BITOP_USING_SSE2(slli_epi32)
AVX2_BITOP_USING_SSE2(srli_epi32)

#define AVX2_INTOP_USING_SSE2(fn) \
static inline v8si avx2_mm256_##fn(v8si x, v8si y) \
{ \
  /* use SSE2 instructions to perform the AVX2 integer operation */ \
  v4si x1, x2; \
  v4si y1, y2; \
  v8si ret; \
  COPY_IMM_TO_XMM(x, x1, x2); \
  COPY_IMM_TO_XMM(y, y1, y2); \
  x1 = _mm_##fn(x1,y1); \
  x2 = _mm_##fn(x2,y2); \
  COPY_XMM_TO_IMM(x1, x2, ret); \
  return(ret); \
}

//#warning "Using SSE2 to perform AVX2 integer ops"
AVX2_INTOP_USING_SSE2(and_si128)
AVX2_INTOP_USING_SSE2(andnot_si128)
AVX2_INTOP_USING_SSE2(cmpeq_epi32)
AVX2_INTOP_USING_SSE2(sub_epi32)
AVX2_INTOP_USING_SSE2(add_epi32)
#define avx2_mm256_and_si256 avx2_mm256_and_si128
#define avx2_mm256_andnot_si256 avx2_mm256_andnot_si128
#else
#define avx2_mm256_slli_epi32 _mm256_slli_epi32
#define avx2_mm256_srli_epi32 _mm256_srli_epi32
#define avx2_mm256_and_si256 _mm256_and_si256
#define avx2_mm256_andnot_si256 _mm256_andnot_si256
#define avx2_mm256_cmpeq_epi32 _mm256_cmpeq_epi32
#define avx2_mm256_sub_epi32 _mm256_sub_epi32
#define avx2_mm256_add_epi32 _mm256_add_epi32
#endif /* __AVX2__ */


/* natural logarithm computed for 8 simultaneous float
   return NaN for x <= 0
*/
//wgvisots - added inline
inline v8sf log256_ps(v8sf x) {
  v8si imm0;
  v8sf one = *(v8sf*)_ps256_1;

  //v8sf invalid_mask = _mm256_cmple_ps(x, _mm256_setzero_ps());
  v8sf invalid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OS);

  x = _mm256_max_ps(x, *(v8sf*)_ps256_min_norm_pos);  /* cut off denormalized stuff */

  // can be done with AVX2
  imm0 = avx2_mm256_srli_epi32(_mm256_castps_si256(x), 23);

  /* keep only the fractional part */
  x = _mm256_and_ps(x, *(v8sf*)_ps256_inv_mant_mask);
  x = _mm256_or_ps(x, *(v8sf*)_ps256_0p5);

  // this is again another AVX2 instruction
  imm0 = avx2_mm256_sub_epi32(imm0, *(v8si*)_pi32_256_0x7f);
  v8sf e = _mm256_cvtepi32_ps(imm0);

  e = _mm256_add_ps(e, one);

  /* part2:
     if( x < SQRTHF ) {
       e -= 1;
       x = x + x - 1.0;
     } else { x = x - 1.0; }
  */
  //v8sf mask = _mm256_cmplt_ps(x, *(v8sf*)_ps256_cephes_SQRTHF);
  v8sf mask = _mm256_cmp_ps(x, *(v8sf*)_ps256_cephes_SQRTHF, _CMP_LT_OS);
  v8sf tmp = _mm256_and_ps(x, mask);
  x = _mm256_sub_ps(x, one);
  e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
  x = _mm256_add_ps(x, tmp);

  v8sf z = _mm256_mul_ps(x,x);

  v8sf y = *(v8sf*)_ps256_cephes_log_p0;
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_log_p1);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_log_p2);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_log_p3);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_log_p4);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_log_p5);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_log_p6);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_log_p7);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_log_p8);
  y = _mm256_mul_ps(y, x);

  y = _mm256_mul_ps(y, z);

  tmp = _mm256_mul_ps(e, *(v8sf*)_ps256_cephes_log_q1);
  y = _mm256_add_ps(y, tmp);


  tmp = _mm256_mul_ps(z, *(v8sf*)_ps256_0p5);
  y = _mm256_sub_ps(y, tmp);

  tmp = _mm256_mul_ps(e, *(v8sf*)_ps256_cephes_log_q2);
  x = _mm256_add_ps(x, y);
  x = _mm256_add_ps(x, tmp);
  x = _mm256_or_ps(x, invalid_mask); // negative arg will be NAN
  return x;
}

_PS256_CONST(exp_hi,    88.3762626647949f);
_PS256_CONST(exp_lo,    -88.3762626647949f);

_PS256_CONST(cephes_LOG2EF, 1.44269504088896341F);
_PS256_CONST(cephes_exp_C1, 0.693359375F);
_PS256_CONST(cephes_exp_C2, -2.12194440e-4F);

_PS256_CONST(cephes_exp_p0, 1.9875691500E-4F);
_PS256_CONST(cephes_exp_p1, 1.3981999507E-3F);
_PS256_CONST(cephes_exp_p2, 8.3334519073E-3F);
_PS256_CONST(cephes_exp_p3, 4.1665795894E-2F);
_PS256_CONST(cephes_exp_p4, 1.6666665459E-1F);
_PS256_CONST(cephes_exp_p5, 5.0000001201E-1F);


//wgvisots - added inline
inline v8sf exp256_ps(v8sf x) {
  v8sf tmp = _mm256_setzero_ps(), fx;
  v8si imm0;
  v8sf one = *(v8sf*)_ps256_1;

  x = _mm256_min_ps(x, *(v8sf*)_ps256_exp_hi);
  x = _mm256_max_ps(x, *(v8sf*)_ps256_exp_lo);

  /* express exp(x) as exp(g + n*log(2)) */
  fx = _mm256_mul_ps(x, *(v8sf*)_ps256_cephes_LOG2EF);
  fx = _mm256_add_ps(fx, *(v8sf*)_ps256_0p5);

  /* how to perform a floorf with SSE: just below */
  //imm0 = _mm256_cvttps_epi32(fx);
  //tmp  = _mm256_cvtepi32_ps(imm0);

  tmp = _mm256_floor_ps(fx);

  /* if greater, substract 1 */
  //v8sf mask = _mm256_cmpgt_ps(tmp, fx);
  v8sf mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
  mask = _mm256_and_ps(mask, one);
  fx = _mm256_sub_ps(tmp, mask);

  tmp = _mm256_mul_ps(fx, *(v8sf*)_ps256_cephes_exp_C1);
  v8sf z = _mm256_mul_ps(fx, *(v8sf*)_ps256_cephes_exp_C2);
  x = _mm256_sub_ps(x, tmp);
  x = _mm256_sub_ps(x, z);

  z = _mm256_mul_ps(x,x);

  v8sf y = *(v8sf*)_ps256_cephes_exp_p0;
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p1);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p2);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p3);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p4);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p5);
  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, x);
  y = _mm256_add_ps(y, one);

  /* build 2^n */
  imm0 = _mm256_cvttps_epi32(fx);
  // another two AVX2 instructions
  imm0 = avx2_mm256_add_epi32(imm0, *(v8si*)_pi32_256_0x7f);
  imm0 = avx2_mm256_slli_epi32(imm0, 23);
  v8sf pow2n = _mm256_castsi256_ps(imm0);
  y = _mm256_mul_ps(y, pow2n);
  return y;
}

_PS256_CONST(minus_cephes_DP1, -0.78515625F);
_PS256_CONST(minus_cephes_DP2, -2.4187564849853515625e-4F);
_PS256_CONST(minus_cephes_DP3, -3.77489497744594108e-8F);
_PS256_CONST(sincof_p0, -1.9515295891E-4F);
_PS256_CONST(sincof_p1,  8.3321608736E-3F);
_PS256_CONST(sincof_p2, -1.6666654611E-1F);
_PS256_CONST(coscof_p0,  2.443315711809948E-005F);
_PS256_CONST(coscof_p1, -1.388731625493765E-003F);
_PS256_CONST(coscof_p2,  4.166664568298827E-002F);
_PS256_CONST(cephes_FOPI, 1.27323954473516F); // 4 / M_PI


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
inline v8sf sin256_ps(v8sf x) { // any x
  v8sf xmm1, xmm2 = _mm256_setzero_ps(), xmm3, sign_bit, y;
  v8si imm0, imm2;

#ifndef __AVX2__
  v4si imm0_1, imm0_2;
  v4si imm2_1, imm2_2;
#endif

  sign_bit = x;
  /* take the absolute value */
  x = _mm256_and_ps(x, *(v8sf*)_ps256_inv_sign_mask);
  /* extract the sign bit (upper one) */
  sign_bit = _mm256_and_ps(sign_bit, *(v8sf*)_ps256_sign_mask);

  /* scale by 4/Pi */
  y = _mm256_mul_ps(x, *(v8sf*)_ps256_cephes_FOPI);

  /*
    Here we start a series of integer operations, which are in the
    realm of AVX2.
    If we don't have AVX, let's perform them using SSE2 directives
  */

#ifdef __AVX2__
  /* store the integer part of y in mm0 */
  imm2 = _mm256_cvttps_epi32(y);
  /* j=(j+1) & (~1) (see the cephes sources) */
  // another two AVX2 instruction
  imm2 = avx2_mm256_add_epi32(imm2, *(v8si*)_pi32_256_1);
  imm2 = avx2_mm256_and_si256(imm2, *(v8si*)_pi32_256_inv1);
  y = _mm256_cvtepi32_ps(imm2);

  /* get the swap sign flag */
  imm0 = avx2_mm256_and_si256(imm2, *(v8si*)_pi32_256_4);
  imm0 = avx2_mm256_slli_epi32(imm0, 29);
  /* get the polynom selection mask
     there is one polynom for 0 <= x <= Pi/4
     and another one for Pi/4<x<=Pi/2

     Both branches will be computed.
  */
  imm2 = avx2_mm256_and_si256(imm2, *(v8si*)_pi32_256_2);
  imm2 = avx2_mm256_cmpeq_epi32(imm2,*(v8si*)_pi32_256_0);
#else
  /* we use SSE2 routines to perform the integer ops */
  COPY_IMM_TO_XMM(_mm256_cvttps_epi32(y),imm2_1,imm2_2);

  imm2_1 = _mm_add_epi32(imm2_1, *(v4si*)_pi32avx_1);
  imm2_2 = _mm_add_epi32(imm2_2, *(v4si*)_pi32avx_1);

  imm2_1 = _mm_and_si128(imm2_1, *(v4si*)_pi32avx_inv1);
  imm2_2 = _mm_and_si128(imm2_2, *(v4si*)_pi32avx_inv1);

  COPY_XMM_TO_IMM(imm2_1,imm2_2,imm2);
  y = _mm256_cvtepi32_ps(imm2);

  imm0_1 = _mm_and_si128(imm2_1, *(v4si*)_pi32avx_4);
  imm0_2 = _mm_and_si128(imm2_2, *(v4si*)_pi32avx_4);

  imm0_1 = _mm_slli_epi32(imm0_1, 29);
  imm0_2 = _mm_slli_epi32(imm0_2, 29);

  COPY_XMM_TO_IMM(imm0_1, imm0_2, imm0);

  imm2_1 = _mm_and_si128(imm2_1, *(v4si*)_pi32avx_2);
  imm2_2 = _mm_and_si128(imm2_2, *(v4si*)_pi32avx_2);

  imm2_1 = _mm_cmpeq_epi32(imm2_1, _mm_setzero_si128());
  imm2_2 = _mm_cmpeq_epi32(imm2_2, _mm_setzero_si128());

  COPY_XMM_TO_IMM(imm2_1, imm2_2, imm2);
#endif

  v8sf swap_sign_bit = _mm256_castsi256_ps(imm0);
  v8sf poly_mask = _mm256_castsi256_ps(imm2);
  sign_bit = _mm256_xor_ps(sign_bit, swap_sign_bit);

  /* The magic pass: "Extended precision modular arithmetic"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
  xmm1 = *(v8sf*)_ps256_minus_cephes_DP1;
  xmm2 = *(v8sf*)_ps256_minus_cephes_DP2;
  xmm3 = *(v8sf*)_ps256_minus_cephes_DP3;
  xmm1 = _mm256_mul_ps(y, xmm1);
  xmm2 = _mm256_mul_ps(y, xmm2);
  xmm3 = _mm256_mul_ps(y, xmm3);
  x = _mm256_add_ps(x, xmm1);
  x = _mm256_add_ps(x, xmm2);
  x = _mm256_add_ps(x, xmm3);

  /* Evaluate the first polynom  (0 <= x <= Pi/4) */
  y = *(v8sf*)_ps256_coscof_p0;
  v8sf z = _mm256_mul_ps(x,x);

  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_coscof_p1);
  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_coscof_p2);
  y = _mm256_mul_ps(y, z);
  y = _mm256_mul_ps(y, z);
  v8sf tmp = _mm256_mul_ps(z, *(v8sf*)_ps256_0p5);
  y = _mm256_sub_ps(y, tmp);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_1);

  /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

  v8sf y2 = *(v8sf*)_ps256_sincof_p0;
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_add_ps(y2, *(v8sf*)_ps256_sincof_p1);
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_add_ps(y2, *(v8sf*)_ps256_sincof_p2);
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_mul_ps(y2, x);
  y2 = _mm256_add_ps(y2, x);

  /* select the correct result from the two polynoms */
  xmm3 = poly_mask;
  y2 = _mm256_and_ps(xmm3, y2); //, xmm3);
  y = _mm256_andnot_ps(xmm3, y);
  y = _mm256_add_ps(y,y2);
  /* update the sign */
  y = _mm256_xor_ps(y, sign_bit);

  return y;
}

/* almost the same as sin_ps */
//wgvisots - added inline
inline v8sf cos256_ps(v8sf x) { // any x
  v8sf xmm1, xmm2 = _mm256_setzero_ps(), xmm3, y;
  v8si imm0, imm2;

#ifndef __AVX2__
  v4si imm0_1, imm0_2;
  v4si imm2_1, imm2_2;
#endif

  /* take the absolute value */
  x = _mm256_and_ps(x, *(v8sf*)_ps256_inv_sign_mask);

  /* scale by 4/Pi */
  y = _mm256_mul_ps(x, *(v8sf*)_ps256_cephes_FOPI);

#ifdef __AVX2__
  /* store the integer part of y in mm0 */
  imm2 = _mm256_cvttps_epi32(y);
  /* j=(j+1) & (~1) (see the cephes sources) */
  imm2 = avx2_mm256_add_epi32(imm2, *(v8si*)_pi32_256_1);
  imm2 = avx2_mm256_and_si256(imm2, *(v8si*)_pi32_256_inv1);
  y = _mm256_cvtepi32_ps(imm2);
  imm2 = avx2_mm256_sub_epi32(imm2, *(v8si*)_pi32_256_2);

  /* get the swap sign flag */
  imm0 = avx2_mm256_andnot_si256(imm2, *(v8si*)_pi32_256_4);
  imm0 = avx2_mm256_slli_epi32(imm0, 29);
  /* get the polynom selection mask */
  imm2 = avx2_mm256_and_si256(imm2, *(v8si*)_pi32_256_2);
  imm2 = avx2_mm256_cmpeq_epi32(imm2, *(v8si*)_pi32_256_0);
#else

  /* we use SSE2 routines to perform the integer ops */
  COPY_IMM_TO_XMM(_mm256_cvttps_epi32(y),imm2_1,imm2_2);

  imm2_1 = _mm_add_epi32(imm2_1, *(v4si*)_pi32avx_1);
  imm2_2 = _mm_add_epi32(imm2_2, *(v4si*)_pi32avx_1);

  imm2_1 = _mm_and_si128(imm2_1, *(v4si*)_pi32avx_inv1);
  imm2_2 = _mm_and_si128(imm2_2, *(v4si*)_pi32avx_inv1);

  COPY_XMM_TO_IMM(imm2_1,imm2_2,imm2);
  y = _mm256_cvtepi32_ps(imm2);

  imm2_1 = _mm_sub_epi32(imm2_1, *(v4si*)_pi32avx_2);
  imm2_2 = _mm_sub_epi32(imm2_2, *(v4si*)_pi32avx_2);

  imm0_1 = _mm_andnot_si128(imm2_1, *(v4si*)_pi32avx_4);
  imm0_2 = _mm_andnot_si128(imm2_2, *(v4si*)_pi32avx_4);

  imm0_1 = _mm_slli_epi32(imm0_1, 29);
  imm0_2 = _mm_slli_epi32(imm0_2, 29);

  COPY_XMM_TO_IMM(imm0_1, imm0_2, imm0);

  imm2_1 = _mm_and_si128(imm2_1, *(v4si*)_pi32avx_2);
  imm2_2 = _mm_and_si128(imm2_2, *(v4si*)_pi32avx_2);

  imm2_1 = _mm_cmpeq_epi32(imm2_1, _mm_setzero_si128());
  imm2_2 = _mm_cmpeq_epi32(imm2_2, _mm_setzero_si128());

  COPY_XMM_TO_IMM(imm2_1, imm2_2, imm2);
#endif

  v8sf sign_bit = _mm256_castsi256_ps(imm0);
  v8sf poly_mask = _mm256_castsi256_ps(imm2);

  /* The magic pass: "Extended precision modular arithmetic"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
  xmm1 = *(v8sf*)_ps256_minus_cephes_DP1;
  xmm2 = *(v8sf*)_ps256_minus_cephes_DP2;
  xmm3 = *(v8sf*)_ps256_minus_cephes_DP3;
  xmm1 = _mm256_mul_ps(y, xmm1);
  xmm2 = _mm256_mul_ps(y, xmm2);
  xmm3 = _mm256_mul_ps(y, xmm3);
  x = _mm256_add_ps(x, xmm1);
  x = _mm256_add_ps(x, xmm2);
  x = _mm256_add_ps(x, xmm3);

  /* Evaluate the first polynom  (0 <= x <= Pi/4) */
  y = *(v8sf*)_ps256_coscof_p0;
  v8sf z = _mm256_mul_ps(x,x);

  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_coscof_p1);
  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_coscof_p2);
  y = _mm256_mul_ps(y, z);
  y = _mm256_mul_ps(y, z);
  v8sf tmp = _mm256_mul_ps(z, *(v8sf*)_ps256_0p5);
  y = _mm256_sub_ps(y, tmp);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_1);

  /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

  v8sf y2 = *(v8sf*)_ps256_sincof_p0;
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_add_ps(y2, *(v8sf*)_ps256_sincof_p1);
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_add_ps(y2, *(v8sf*)_ps256_sincof_p2);
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_mul_ps(y2, x);
  y2 = _mm256_add_ps(y2, x);

  /* select the correct result from the two polynoms */
  xmm3 = poly_mask;
  y2 = _mm256_and_ps(xmm3, y2); //, xmm3);
  y = _mm256_andnot_ps(xmm3, y);
  y = _mm256_add_ps(y,y2);
  /* update the sign */
  y = _mm256_xor_ps(y, sign_bit);

  return y;
}

/* since sin256_ps and cos256_ps are almost identical, sincos256_ps could replace both of them..
   it is almost as fast, and gives you a free cosine with your sine */
//wgvisots - added inline
inline void sincos256_ps(v8sf x, v8sf *s, v8sf *c) {

  v8sf xmm1, xmm2, xmm3 = _mm256_setzero_ps(), sign_bit_sin, y;
  v8si imm0, imm2, imm4;

#ifndef __AVX2__
  v4si imm0_1, imm0_2;
  v4si imm2_1, imm2_2;
  v4si imm4_1, imm4_2;
#endif

  sign_bit_sin = x;
  /* take the absolute value */
  x = _mm256_and_ps(x, *(v8sf*)_ps256_inv_sign_mask);
  /* extract the sign bit (upper one) */
  sign_bit_sin = _mm256_and_ps(sign_bit_sin, *(v8sf*)_ps256_sign_mask);

  /* scale by 4/Pi */
  y = _mm256_mul_ps(x, *(v8sf*)_ps256_cephes_FOPI);

#ifdef __AVX2__
  /* store the integer part of y in imm2 */
  imm2 = _mm256_cvttps_epi32(y);

  /* j=(j+1) & (~1) (see the cephes sources) */
  imm2 = avx2_mm256_add_epi32(imm2, *(v8si*)_pi32_256_1);
  imm2 = avx2_mm256_and_si256(imm2, *(v8si*)_pi32_256_inv1);

  y = _mm256_cvtepi32_ps(imm2);
  imm4 = imm2;

  /* get the swap sign flag for the sine */
  imm0 = avx2_mm256_and_si256(imm2, *(v8si*)_pi32_256_4);
  imm0 = avx2_mm256_slli_epi32(imm0, 29);
  //v8sf swap_sign_bit_sin = _mm256_castsi256_ps(imm0);

  /* get the polynom selection mask for the sine*/
  imm2 = avx2_mm256_and_si256(imm2, *(v8si*)_pi32_256_2);
  imm2 = avx2_mm256_cmpeq_epi32(imm2, *(v8si*)_pi32_256_0);
  //v8sf poly_mask = _mm256_castsi256_ps(imm2);
#else
  /* we use SSE2 routines to perform the integer ops */
  COPY_IMM_TO_XMM(_mm256_cvttps_epi32(y),imm2_1,imm2_2);

  imm2_1 = _mm_add_epi32(imm2_1, *(v4si*)_pi32avx_1);
  imm2_2 = _mm_add_epi32(imm2_2, *(v4si*)_pi32avx_1);

  imm2_1 = _mm_and_si128(imm2_1, *(v4si*)_pi32avx_inv1);
  imm2_2 = _mm_and_si128(imm2_2, *(v4si*)_pi32avx_inv1);

  COPY_XMM_TO_IMM(imm2_1,imm2_2,imm2);
  y = _mm256_cvtepi32_ps(imm2);

  imm4_1 = imm2_1;
  imm4_2 = imm2_2;

  imm0_1 = _mm_and_si128(imm2_1, *(v4si*)_pi32avx_4);
  imm0_2 = _mm_and_si128(imm2_2, *(v4si*)_pi32avx_4);

  imm0_1 = _mm_slli_epi32(imm0_1, 29);
  imm0_2 = _mm_slli_epi32(imm0_2, 29);

  COPY_XMM_TO_IMM(imm0_1, imm0_2, imm0);

  imm2_1 = _mm_and_si128(imm2_1, *(v4si*)_pi32avx_2);
  imm2_2 = _mm_and_si128(imm2_2, *(v4si*)_pi32avx_2);

  imm2_1 = _mm_cmpeq_epi32(imm2_1, _mm_setzero_si128());
  imm2_2 = _mm_cmpeq_epi32(imm2_2, _mm_setzero_si128());

  COPY_XMM_TO_IMM(imm2_1, imm2_2, imm2);
#endif
  v8sf swap_sign_bit_sin = _mm256_castsi256_ps(imm0);
  v8sf poly_mask = _mm256_castsi256_ps(imm2);

  /* The magic pass: "Extended precision modular arithmetic"
     x = ((x - y * DP1) - y * DP2) - y * DP3; */
  xmm1 = *(v8sf*)_ps256_minus_cephes_DP1;
  xmm2 = *(v8sf*)_ps256_minus_cephes_DP2;
  xmm3 = *(v8sf*)_ps256_minus_cephes_DP3;
  xmm1 = _mm256_mul_ps(y, xmm1);
  xmm2 = _mm256_mul_ps(y, xmm2);
  xmm3 = _mm256_mul_ps(y, xmm3);
  x = _mm256_add_ps(x, xmm1);
  x = _mm256_add_ps(x, xmm2);
  x = _mm256_add_ps(x, xmm3);

#ifdef __AVX2__
  imm4 = avx2_mm256_sub_epi32(imm4, *(v8si*)_pi32_256_2);
  imm4 = avx2_mm256_andnot_si256(imm4, *(v8si*)_pi32_256_4);
  imm4 = avx2_mm256_slli_epi32(imm4, 29);
#else
  imm4_1 = _mm_sub_epi32(imm4_1, *(v4si*)_pi32avx_2);
  imm4_2 = _mm_sub_epi32(imm4_2, *(v4si*)_pi32avx_2);

  imm4_1 = _mm_andnot_si128(imm4_1, *(v4si*)_pi32avx_4);
  imm4_2 = _mm_andnot_si128(imm4_2, *(v4si*)_pi32avx_4);

  imm4_1 = _mm_slli_epi32(imm4_1, 29);
  imm4_2 = _mm_slli_epi32(imm4_2, 29);

  COPY_XMM_TO_IMM(imm4_1, imm4_2, imm4);
#endif

  v8sf sign_bit_cos = _mm256_castsi256_ps(imm4);

  sign_bit_sin = _mm256_xor_ps(sign_bit_sin, swap_sign_bit_sin);

  /* Evaluate the first polynom  (0 <= x <= Pi/4) */
  v8sf z = _mm256_mul_ps(x,x);
  y = *(v8sf*)_ps256_coscof_p0;

  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_coscof_p1);
  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_coscof_p2);
  y = _mm256_mul_ps(y, z);
  y = _mm256_mul_ps(y, z);
  v8sf tmp = _mm256_mul_ps(z, *(v8sf*)_ps256_0p5);
  y = _mm256_sub_ps(y, tmp);
  y = _mm256_add_ps(y, *(v8sf*)_ps256_1);

  /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

  v8sf y2 = *(v8sf*)_ps256_sincof_p0;
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_add_ps(y2, *(v8sf*)_ps256_sincof_p1);
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_add_ps(y2, *(v8sf*)_ps256_sincof_p2);
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_mul_ps(y2, x);
  y2 = _mm256_add_ps(y2, x);

  /* select the correct result from the two polynoms */
  xmm3 = poly_mask;
  v8sf ysin2 = _mm256_and_ps(xmm3, y2);
  v8sf ysin1 = _mm256_andnot_ps(xmm3, y);
  y2 = _mm256_sub_ps(y2,ysin2);
  y = _mm256_sub_ps(y, ysin1);

  xmm1 = _mm256_add_ps(ysin1,ysin2);
  xmm2 = _mm256_add_ps(y,y2);

  /* update the sign */
  *s = _mm256_xor_ps(xmm1, sign_bit_sin);
  *c = _mm256_xor_ps(xmm2, sign_bit_cos);
}

//TODO -> make this faster!!!
//wgvisots
//return -pi < z <= pi (ANSI C like)
inline void atan2(const __m256 &a, const __m256 &b, __m256 &c){
    float* __restrict__ aArray = (float*)&a;
    float* __restrict__ bArray = (float*)&b;
    float* __restrict__ cArray = (float*)&c;

    cArray[0] = std::atan2(aArray[0], aArray[1]);
    cArray[1] = std::atan2(aArray[2], aArray[3]);
    cArray[2] = std::atan2(bArray[0], bArray[1]);
    cArray[3] = std::atan2(bArray[2], bArray[3]);
    cArray[4] = std::atan2(aArray[4], aArray[5]);
    cArray[5] = std::atan2(aArray[6], aArray[7]);
    cArray[6] = std::atan2(bArray[4], bArray[5]);
    cArray[7] = std::atan2(bArray[6], bArray[7]);
}
inline void atan(const __m256 &a, __m256 &c){
    float *aArray = (float*)&a;
    float cArray[8] ALIGN32_END;

    cArray[0] = std::atan(aArray[0]);
    cArray[1] = std::atan(aArray[2]);
    cArray[2] = std::atan(aArray[0]);
    cArray[3] = std::atan(aArray[2]);
    cArray[4] = std::atan(aArray[4]);
    cArray[5] = std::atan(aArray[6]);
    cArray[6] = std::atan(aArray[4]);
    cArray[7] = std::atan(aArray[6]);

    memcpy(&c, cArray, sizeof(cArray));
}

inline void printVec(const v8sf& v) {
    float* vArray = (float*)&v;
    std::cout << "[ ";
    for (std::size_t i = 0; i < 8; ++i) {
        std::cout << vArray[i] << " ";
    }
    std::cout << "]" << std::endl;
}

/**
 * Vectorized version of polevl from Cephes.
 * Original copyright:
 * Cephes Math Library Release 2.1:  December, 1988
 * Copyright 1984, 1987, 1988 by Stephen L. Moshier
 * Direct inquiries to 30 Frost Street, Cambridge, MA 02140
 *
 * @param x
 * @param coef
 * @param N
 * @return
 */
inline v8sf polevl(const v8sf& x, const float* coef, const std::size_t& N) {
    const float* p = coef;
    v8sf ans = _mm256_broadcast_ss(p++);
    std::size_t i = N;
    do {
        v8sf pVec = _mm256_broadcast_ss(p++);
        ans = _mm256_fmadd_ps(ans, x, pVec);
    }while (--i);
    return ans;
}

/**
 * TODO -> IN WORK - DO NOT USE THIS, IT WILL LIE TO YOU!
 *
 * This is an atan implementation intneded for use by atan2 below.  As such, it doesn't test for
 * inf/-inf, as it assumes atan2 will take care of those.
 *
 * @param ratio y/x
 * @return
 */
inline v8sf atan(v8sf ratio) {
    //NOTE: I know flag should really be an integer, but keeping it as a float makes the code slightly cleaner.
    //      And '1' and '2' are exactly representable as floats - so it is safe... probably.

    //make argument positive and save the sign
    __m256 sign = _mm256_cmp_ps(ratio, *(__m256*)_ps256_zero, _CMP_LT_OQ);
    ratio = _mm256_andnot_ps(_mm256_castsi256_ps(*(__m256i*)_pi32_256_fp_sign_mask), ratio);

    //range reduction
    // if( x > T3P8 ){
    //     y = PIO2;
    //     flag = 1;
    //     x = -( 1.0/x );/*It seems odd that the - is outside the parenthesis - this might be important)
    // }
    const __m256 mask1 = _mm256_cmp_ps(ratio, *(__m256*)_ps256_T3P8, _CMP_GT_OQ);
    __m256 y = _mm256_and_ps(mask1, *(__m256*)_ps256_PIO2);
    ratio = _mm256_or_ps(_mm256_andnot_ps(mask1, ratio),
                         _mm256_and_ps(mask1,_mm256_div_ps(_mm256_set1_ps(-1.0f),
                                                          ratio)));

    //else if( x <= 0.66 ){
	//  y = 0.0;
	//}
    //if(x <= 0.66) y = 0.0 /*we get the assignment for free*/
    const __m256 mask2 = _mm256_cmp_ps(ratio, *(__m256*)_ps256_0_66, _CMP_LE_OQ);

    //else{ /* x > 0.66 */
    //  y = PIO4;
	//  flag = 2;
	//  x = (x-1.0)/(x+1.0);
    //}

    y = _mm256_or_ps(y, _mm256_andnot_ps(mask2, *(__m256*)_ps256_PIO4));
    //x = (x-1.0)/(x+1.0);
    __m256 newx = _mm256_div_ps(_mm256_sub_ps(ratio, *(__m256*)_ps256_1),
                                _mm256_add_ps(ratio, *(__m256*)_ps256_1));
    __m256 mask = _mm256_or_ps(mask1, mask2);

    ratio = _mm256_or_ps(_mm256_and_ps(ratio, mask),
                         _mm256_andnot_ps(mask, newx));

    //z = x * x;
    __m256 z = _mm256_mul_ps(ratio, ratio);
    //z = z * polevl( z, P, 4 ) / p1evl( z, Q, 5 );
    z = _mm256_div_ps(_mm256_mul_ps(z,
                                    polevl(z, atan_P, 4)),
                      polevl(z, atan_Q, 5));

    //z = x * z + x
    z = _mm256_fmadd_ps(ratio, z, ratio);

    z = _mm256_add_ps(z, _mm256_and_ps(mask2, *(__m256*)_ps256_HALF_MOREBITS));
    z = _mm256_add_ps(z, _mm256_and_ps(mask1, *(__m256*)_ps256_MOREBITS));

    y = _mm256_add_ps(y, z);

    //restore sign
    __m256 ss = _mm256_castsi256_ps(*(__m256i*)_pi32_256_fp_sign_mask);
    ss = _mm256_and_ps(ss, sign);
    y = _mm256_or_ps(ss, y);
    return y;
}

/**
 * TODO -> IN WORK - DO NOT USE THIS, IT WILL LIE TO YOU!
 *
 * NOTE: Does not curently test for NAN.  NANs will produce errors.
 * NOTE: Does not currently test for INF/-INF.  These will produce undefined results
 *
 * @param y
 * @param x
 * @param z
 */
inline v8sf atan2(__m256 y, __m256 x) {

    //mask NaNs here

    const __m256 y_zero       = _mm256_cmp_ps(y, *(__m256*)_ps256_zero,         _CMP_EQ_UQ);
    const __m256 y_negzero    = _mm256_cmp_ps(y, *(__m256*)_ps256_negativeZero, _CMP_EQ_UQ);
    const __m256 y_gt_zero    = _mm256_cmp_ps(y, *(__m256*)_ps256_zero,         _CMP_GT_OQ);
    const __m256 x_ge_zero    = _mm256_cmp_ps(x, *(__m256*)_ps256_zero,         _CMP_GE_OQ);
    const __m256 x_le_negzero = _mm256_cmp_ps(x, *(__m256*)_ps256_negativeZero, _CMP_LE_OQ);
    const __m256 x_eq_zero    = _mm256_cmp_ps(x, *(__m256*)_ps256_zero,         _CMP_EQ_UQ);


    //if((y == -0.0) && (x >  0.0)) z = y
    //if((y == -0.0) && (x == 0.0)) z = y
    __m256 z = _mm256_and_ps(y, _mm256_and_ps(y_negzero, x_ge_zero));

    //if((y == -0.0) && (x <   0.0)) z = -PI
    //if((y == -0.0) && (x == -0.0)) z = -PI
    z = _mm256_or_ps(z, _mm256_and_ps(_mm256_and_ps(y_negzero, x_le_negzero), *(__m256*)_ps256_negPI));

    //if((y == 0.0) && (x == -0.0)) z = PI
    //if((y == 0.0) && (x <   0.0)) z = PI
    z = _mm256_or_ps(z, _mm256_and_ps(_mm256_and_ps(y_zero, x_le_negzero), *(__m256*)_ps256_PI));

    //if((y == 0.0) && (x == 0.0)) z = 0.0
    //if((y == 0.0) && (z >  0.0)) z = 0.0
    //already set to zero

    //if((x == 0.0) && (y > 0.0)) z = PI/2
    z = _mm256_or_ps(z, _mm256_and_ps(_mm256_and_ps(x_eq_zero, y_gt_zero), *(__m256*)_ps256_PIO2));

    //if((x == 0.0) && (y <= 0.0) z = -PI/2
    z = _mm256_or_ps(z, _mm256_and_ps(_mm256_andnot_ps(y_gt_zero, x_eq_zero), *(__m256*)_ps256_negPIO2));


    //TODO -> finish filling out tests for infinity
    //TODO -> NOTE: Don't forget to put the test for 0.0/-0.0 in the doneMask below
    //if((x ==  INFINITY) && (y ==  INFINITY)) z = 0.25 * PI
    //if((x ==  INFINITY) && (y == -INFINITY)) z = -0.25 * PI
    //if((x ==  INFINITY) && (y <   0.0))      z = -0.0
    //if((x ==  INFINITY) && (y >=  0.0)       z = 0.0
    //if((y ==  INFINITY) && (z == 0.0))       z = PI/2
    //if((y == -INFINITY) && (z == 0.0))       z = -PI/2

    //now we need to mask off all the values we already set, so we don't overwrite them below
    const __m256 doneMask = _mm256_and_ps(_mm256_and_ps(y_zero, x_ge_zero),
                                          _mm256_cmp_ps(z, *(__m256*)_ps256_zero, _CMP_NEQ_UQ));


    //if(x < 0.0) code = 2
    __m256i code;
    code = _mm256_and_si256(_mm256_castps_si256(x_le_negzero),
                            *(__m256i*)_pi32_256_2);

    //if(y < 0.0) code |= 1
    code = _mm256_or_si256(code,
                           _mm256_andnot_si256(_mm256_castps_si256(y_gt_zero),
                                               *(__m256i*)_pi32_256_1));

    const __m256 w2 = _mm256_and_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(code, *(__m256i*)_pi32_256_2)), *(__m256*)_ps256_PI);
    const __m256 w3 = _mm256_and_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(code, *(__m256i*)_pi32_256_3)), *(__m256*)_ps256_PI);
    const __m256 w  = _mm256_or_ps(w2,w3);

    const __m256 ratio = _mm256_div_ps(y, x);

    const __m256 atanRatio = _mm256_add_ps(w, atan(ratio));

    const __m256 z1 = _mm256_and_ps(doneMask, z);
    const __m256 z2 = _mm256_andnot_ps(doneMask, atanRatio);
    z = _mm256_or_ps(z1, z2);

    //if((z == 0.0) && (y < 0.0)) z = NEGZERO
    const __m256 negzeroMask = _mm256_and_ps(_mm256_cmp_ps(z, *(__m256*)_ps256_zero, _CMP_EQ_UQ),
                               _mm256_cmp_ps(y, *(__m256*)_ps256_zero, _CMP_LT_OQ));

    z = _mm256_andnot_ps(negzeroMask, z);
    z = _mm256_or_ps(z, _mm256_and_ps(negzeroMask, *(__m256*)_ps256_negativeZero));
    return z;
}

}


