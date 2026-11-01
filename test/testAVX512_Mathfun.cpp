/*
  Copyright (C) 2026 W. Greg Visotski

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

/*
  Tests for avx512_mathfun.h.

  This file must be compiled with AVX-512F enabled (see test/CMakeLists.txt)
  and run on a CPU that supports it, or under Intel SDE.

  Three kinds of checks:
    * accuracy against the C library evaluated in double precision, measured
      in ulps (or absolute error where the result can be ~0),
    * C99 Annex F special values (signed zeros, infinities, NaN),
    * agreement with the AVX2 versions in avx_mathfun.h over their valid
      input ranges.
*/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numbers>
#include <random>
#include <vector>

#include "gtest/gtest.h"

#include "../src/avx_mathfun.h"
#include "../src/avx512_mathfun.h"
#include "mathfun_test_helpers.h"

namespace {

/* ---- helpers ------------------------------------------------------------ */

// Apply a 16-wide function to an arbitrary-length input, padding the tail.
template <typename F>
FloatVec apply512(F f, const FloatVec& in) {
    FloatVec out(in.size());
    alignas(64) float buf[16];
    for (std::size_t i = 0; i < in.size(); i += 16) {
        const std::size_t n = std::min<std::size_t>(16, in.size() - i);
        std::fill(buf, buf + 16, 1.0F);
        std::copy(in.begin() + i, in.begin() + i + n, buf);
        _mm512_store_ps(buf, f(_mm512_load_ps(buf)));
        std::copy(buf, buf + n, out.begin() + i);
    }
    return out;
}

template <typename F>
FloatVec apply512(F f, const FloatVec& a, const FloatVec& b) {
    FloatVec out(a.size());
    alignas(64) float ba[16];
    alignas(64) float bb[16];
    for (std::size_t i = 0; i < a.size(); i += 16) {
        const std::size_t n = std::min<std::size_t>(16, a.size() - i);
        std::fill(ba, ba + 16, 1.0F);
        std::fill(bb, bb + 16, 1.0F);
        std::copy(a.begin() + i, a.begin() + i + n, ba);
        std::copy(b.begin() + i, b.begin() + i + n, bb);
        _mm512_store_ps(ba, f(_mm512_load_ps(ba), _mm512_load_ps(bb)));
        std::copy(ba, ba + n, out.begin() + i);
    }
    return out;
}


__m512 sin512(__m512 x) { return avx512_mathfun::sin512_ps(x); }
__m512 cos512(__m512 x) { return avx512_mathfun::cos512_ps(x); }
__m512 log512(__m512 x) { return avx512_mathfun::log512_ps(x); }
__m512 exp512(__m512 x) { return avx512_mathfun::exp512_ps(x); }
__m512 atan512(__m512 x) { return avx512_mathfun::atan(x); }
__m512 atan2_512(__m512 y, __m512 x) { return avx512_mathfun::atan2(y, x); }

} // namespace

/* ---- log ----------------------------------------------------------------- */

TEST(AVX512MathFunTest, LogAccuracyNormalRange) {
    const FloatVec in = bitSweep(std::numeric_limits<float>::min(), std::numeric_limits<float>::max(), 97);
    const ErrorStats s = measure(in, apply512(log512, in), refLog);
    EXPECT_LE(s.maxUlp, 2) << "worst input " << s.maxUlpAt;
}

TEST(AVX512MathFunTest, LogAccuracyNearOne) {
    // log is hardest near 1, where the result goes to zero.
    const FloatVec in = bitSweep(0.5F, 2.0F, 7);
    const ErrorStats s = measure(in, apply512(log512, in), refLog);
    EXPECT_LE(s.maxUlp, 2) << "worst input " << s.maxUlpAt;
}

TEST(AVX512MathFunTest, LogAccuracySubnormal) {
    const FloatVec in = bitSweep(std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::min(), 13);
    const ErrorStats s = measure(in, apply512(log512, in), refLog);
    EXPECT_LE(s.maxUlp, 2) << "worst input " << s.maxUlpAt;
}

TEST(AVX512MathFunTest, LogSpecialValues) {
    const FloatVec in{0.0F, -0.0F, 1.0F, -1.0F, kInf, -kInf, kNaN, -kNaN,
                      2.0F, 0.5F, std::numeric_limits<float>::max(), -std::numeric_limits<float>::denorm_min()};
    const FloatVec out = apply512(log512, in);
    EXPECT_TRUE(sameFloat(out[0], -kInf));
    EXPECT_TRUE(sameFloat(out[1], -kInf));
    EXPECT_TRUE(sameFloat(out[2], 0.0F));
    EXPECT_TRUE(std::isnan(out[3]));
    EXPECT_TRUE(sameFloat(out[4], kInf));
    EXPECT_TRUE(std::isnan(out[5]));
    EXPECT_TRUE(std::isnan(out[6]));
    EXPECT_TRUE(std::isnan(out[7]));
    EXPECT_LE(ulpDiff(out[8], std::log(2.0F)), 1);
    EXPECT_LE(ulpDiff(out[9], std::log(0.5F)), 1);
    EXPECT_LE(ulpDiff(out[10], std::log(std::numeric_limits<float>::max())), 1);
    EXPECT_TRUE(std::isnan(out[11]));
}

TEST(AVX512MathFunTest, LogMatchesAVX2) {
    const FloatVec in = bitSweep(std::numeric_limits<float>::min(), std::numeric_limits<float>::max(), 1009);
    float worst = 0.0F;
    const std::int64_t d = maxUlpBetween(apply512(log512, in),
                                         apply256([](__m256 x) { return avx_mathfun::log256_ps(x); }, in),
                                         &worst, &in);
    EXPECT_LE(d, 2) << "worst input " << worst;
}

/* ---- exp ----------------------------------------------------------------- */

TEST(AVX512MathFunTest, ExpAccuracyNormalRange) {
    // exp(x) is a normal float for x in about [-87.33, 88.72]
    const FloatVec in = concat(linspace(-87.3F, 88.7F, 1000003), linspace(-1.0F, 1.0F, 100003));
    const ErrorStats s = measure(in, apply512(exp512, in), refExp);
    EXPECT_LE(s.maxUlp, 2) << "worst input " << s.maxUlpAt;
}

TEST(AVX512MathFunTest, ExpAccuracySubnormalResults) {
    // Results are subnormal here.
    const FloatVec in = linspace(-103.9F, -87.4F, 100003);
    const ErrorStats s = measure(in, apply512(exp512, in), refExp);
    EXPECT_LE(s.maxUlp, 1) << "worst input " << s.maxUlpAt;
}

TEST(AVX512MathFunTest, ExpSpecialValues) {
    const FloatVec in{0.0F, -0.0F, kInf, -kInf, kNaN, 89.0F, 1000.0F, -105.0F, -1000.0F, 1.0F, 88.72F};
    const FloatVec out = apply512(exp512, in);
    EXPECT_TRUE(sameFloat(out[0], 1.0F));
    EXPECT_TRUE(sameFloat(out[1], 1.0F));
    EXPECT_TRUE(sameFloat(out[2], kInf));
    EXPECT_TRUE(sameFloat(out[3], 0.0F));
    EXPECT_TRUE(std::isnan(out[4]));
    EXPECT_TRUE(sameFloat(out[5], kInf));
    EXPECT_TRUE(sameFloat(out[6], kInf));
    EXPECT_TRUE(sameFloat(out[7], 0.0F));
    EXPECT_TRUE(sameFloat(out[8], 0.0F));
    EXPECT_LE(ulpDiff(out[9], std::numbers::e_v<float>), 1);
    EXPECT_LE(ulpDiff(out[10], static_cast<float>(std::exp(static_cast<double>(88.72F)))), 2);
}

TEST(AVX512MathFunTest, ExpMatchesAVX2) {
    const FloatVec in = linspace(-87.3F, 88.3F, 200003);
    float worst = 0.0F;
    const std::int64_t d = maxUlpBetween(apply512(exp512, in),
                                         apply256([](__m256 x) { return avx_mathfun::exp256_ps(x); }, in),
                                         &worst, &in);
    EXPECT_LE(d, 2) << "worst input " << worst;
}

/* ---- sin / cos / sincos -------------------------------------------------- */

TEST(AVX512MathFunTest, SinCosAccuracyPrimaryRange) {
    const FloatVec pos = linspace(0.0F, 2.0F * std::numbers::pi_v<float>, 1000003);
    const FloatVec in = concat(pos, negated(pos));
    const ErrorStats s = measure(in, apply512(sin512, in), refSin);
    const ErrorStats c = measure(in, apply512(cos512, in), refCos);
    EXPECT_LE(s.maxAbs, 1.2e-7) << "sin worst input " << s.maxAbsAt;
    EXPECT_LE(c.maxAbs, 1.2e-7) << "cos worst input " << c.maxAbsAt;
}

TEST(AVX512MathFunTest, SinCosAccuracySmallArguments) {
    // Near zero sin(x) ~ x, so check relative accuracy (ulps) too.
    const FloatVec pos = bitSweep(1e-30F, std::numbers::pi_v<float> / 4.0F, 101);
    const FloatVec in = concat(pos, negated(pos));
    const ErrorStats s = measure(in, apply512(sin512, in), refSin);
    const ErrorStats c = measure(in, apply512(cos512, in), refCos);
    EXPECT_LE(s.maxUlp, 2) << "sin worst input " << s.maxUlpAt;
    EXPECT_LE(c.maxUlp, 2) << "cos worst input " << c.maxUlpAt;
}

TEST(AVX512MathFunTest, SinCosAccuracyUpTo8192) {
    // Cephes documents excellent accuracy for |x| < 8192.
    const FloatVec pos = linspace(0.0F, 8192.0F, 2000003);
    const FloatVec in = concat(pos, negated(pos));
    const ErrorStats s = measure(in, apply512(sin512, in), refSin);
    const ErrorStats c = measure(in, apply512(cos512, in), refCos);
    EXPECT_LE(s.maxAbs, 1.2e-7) << "sin worst input " << s.maxAbsAt;
    EXPECT_LE(c.maxAbs, 1.2e-7) << "cos worst input " << c.maxAbsAt;
}

TEST(AVX512MathFunTest, SinCosSpecialValues) {
    const FloatVec in{0.0F, -0.0F, kNaN, kInf, -kInf};
    const FloatVec s = apply512(sin512, in);
    const FloatVec c = apply512(cos512, in);
    EXPECT_TRUE(sameFloat(s[0], 0.0F));
    EXPECT_TRUE(sameFloat(s[1], -0.0F));
    EXPECT_TRUE(sameFloat(c[0], 1.0F));
    EXPECT_TRUE(sameFloat(c[1], 1.0F));
    for (std::size_t i = 2; i < in.size(); ++i) {
        EXPECT_TRUE(std::isnan(s[i])) << "sin(" << in[i] << ") = " << s[i];
        EXPECT_TRUE(std::isnan(c[i])) << "cos(" << in[i] << ") = " << c[i];
    }
}

TEST(AVX512MathFunTest, SinCosMatchesSinAndCos) {
    const FloatVec pos = linspace(0.0F, 8192.0F, 500009);
    const FloatVec in = concat(pos, negated(pos));
    FloatVec s(in.size());
    FloatVec c(in.size());
    alignas(64) float buf[16];
    alignas(64) float sb[16];
    alignas(64) float cb[16];
    for (std::size_t i = 0; i + 16 <= in.size(); i += 16) {
        std::copy(in.begin() + i, in.begin() + i + 16, buf);
        __m512 vs;
        __m512 vc;
        avx512_mathfun::sincos512_ps(_mm512_load_ps(buf), &vs, &vc);
        _mm512_store_ps(sb, vs);
        _mm512_store_ps(cb, vc);
        std::copy(sb, sb + 16, s.begin() + i);
        std::copy(cb, cb + 16, c.begin() + i);
    }
    const std::size_t n = (in.size() / 16) * 16;
    const FloatVec s1 = apply512(sin512, in);
    const FloatVec c1 = apply512(cos512, in);
    for (std::size_t i = 0; i < n; ++i) {
        ASSERT_TRUE(sameFloat(s[i], s1[i])) << "sincos sin, x = " << in[i];
        ASSERT_TRUE(sameFloat(c[i], c1[i])) << "sincos cos, x = " << in[i];
    }
}

TEST(AVX512MathFunTest, SinCosMatchesAVX2) {
    const FloatVec pos = linspace(0.0F, 8192.0F, 500009);
    const FloatVec in = concat(pos, negated(pos));
    // FMA in the AVX-512 range reduction changes rounding slightly, so compare
    // absolute differences rather than ulps (results near zero have tiny ulps).
    const FloatVec s512 = apply512(sin512, in);
    const FloatVec c512 = apply512(cos512, in);
    const FloatVec s256 = apply256([](__m256 x) { return avx_mathfun::sin256_ps(x); }, in);
    const FloatVec c256 = apply256([](__m256 x) { return avx_mathfun::cos256_ps(x); }, in);
    float maxS = 0.0F;
    float maxC = 0.0F;
    for (std::size_t i = 0; i < in.size(); ++i) {
        maxS = std::max(maxS, std::fabs(s512[i] - s256[i]));
        maxC = std::max(maxC, std::fabs(c512[i] - c256[i]));
    }
    EXPECT_LE(maxS, 1.2e-7F);
    EXPECT_LE(maxC, 1.2e-7F);
}

/* ---- atan ---------------------------------------------------------------- */

TEST(AVX512MathFunTest, AtanAccuracy) {
    const FloatVec pos = bitSweep(1e-30F, 1e30F, 101);
    const FloatVec in = concat(pos, negated(pos));
    const ErrorStats s = measure(in, apply512(atan512, in), refAtan);
    EXPECT_LE(s.maxUlp, 2) << "worst input " << s.maxUlpAt;
}

TEST(AVX512MathFunTest, AtanAccuracyAroundBranchPoints) {
    // The range reduction switches at 0.66 and tan(3*pi/8) ~ 2.414
    const FloatVec pos = concat(linspace(0.6F, 0.7F, 100003), linspace(2.3F, 2.5F, 100003));
    const FloatVec in = concat(pos, negated(pos));
    const ErrorStats s = measure(in, apply512(atan512, in), refAtan);
    EXPECT_LE(s.maxUlp, 2) << "worst input " << s.maxUlpAt;
}

TEST(AVX512MathFunTest, AtanSpecialValues) {
    const FloatVec in{0.0F, -0.0F, kInf, -kInf, kNaN, 1.0F, -1.0F};
    const FloatVec out = apply512(atan512, in);
    EXPECT_TRUE(sameFloat(out[0], 0.0F));
    EXPECT_TRUE(sameFloat(out[1], -0.0F));
    EXPECT_TRUE(sameFloat(out[2], std::numbers::pi_v<float> / 2.0F));
    EXPECT_TRUE(sameFloat(out[3], -std::numbers::pi_v<float> / 2.0F));
    EXPECT_TRUE(std::isnan(out[4]));
    EXPECT_LE(ulpDiff(out[5], std::numbers::pi_v<float> / 4.0F), 1);
    EXPECT_LE(ulpDiff(out[6], -std::numbers::pi_v<float> / 4.0F), 1);
}

TEST(AVX512MathFunTest, AtanMatchesAVX2) {
    const FloatVec pos = bitSweep(1e-20F, 1e20F, 1009);
    const FloatVec in = concat(pos, negated(pos));
    float worst = 0.0F;
    const std::int64_t d = maxUlpBetween(apply512(atan512, in),
                                         apply256([](__m256 x) { return avx_mathfun::atan(x); }, in),
                                         &worst, &in);
    EXPECT_LE(d, 2) << "worst input " << worst;
}

/* ---- atan2 --------------------------------------------------------------- */

TEST(AVX512MathFunTest, Atan2AllQuadrants) {
    // Grid covering all four quadrants, including both x and y negative.
    FloatVec ys;
    FloatVec xs;
    const FloatVec axis = linspace(-12.0F, 12.0F, 1201); // step 0.02, includes 0
    for (float y : axis) {
        for (float x : axis) {
            ys.push_back(y);
            xs.push_back(x);
        }
    }
    const FloatVec out = apply512(atan2_512, ys, xs);
    std::int64_t worst = 0;
    std::size_t worstIdx = 0;
    for (std::size_t i = 0; i < out.size(); ++i) {
        const float ref = static_cast<float>(std::atan2(static_cast<double>(ys[i]), static_cast<double>(xs[i])));
        const std::int64_t u = ulpDiff(out[i], ref);
        if (u > worst) {
            worst = u;
            worstIdx = i;
        }
    }
    EXPECT_LE(worst, 3) << "worst at atan2(" << ys[worstIdx] << ", " << xs[worstIdx] << ") = " << out[worstIdx]
                        << ", expected " << std::atan2(ys[worstIdx], xs[worstIdx]);
}

TEST(AVX512MathFunTest, Atan2RandomWideRange) {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> mant(-1.0F, 1.0F);
    std::uniform_int_distribution<int> expo(-40, 40);
    FloatVec ys(1000000);
    FloatVec xs(ys.size());
    for (std::size_t i = 0; i < ys.size(); ++i) {
        ys[i] = std::ldexp(mant(rng), expo(rng));
        xs[i] = std::ldexp(mant(rng), expo(rng));
    }
    const FloatVec out = apply512(atan2_512, ys, xs);
    std::int64_t worst = 0;
    std::size_t worstIdx = 0;
    for (std::size_t i = 0; i < out.size(); ++i) {
        const float ref = static_cast<float>(std::atan2(static_cast<double>(ys[i]), static_cast<double>(xs[i])));
        const std::int64_t u = ulpDiff(out[i], ref);
        if (u > worst) {
            worst = u;
            worstIdx = i;
        }
    }
    EXPECT_LE(worst, 3) << "worst at atan2(" << ys[worstIdx] << ", " << xs[worstIdx] << ")";
}

TEST(AVX512MathFunTest, Atan2SpecialValues) {
    // Every combination of these, compared exactly (incl. sign of zero) to std::atan2.
    const FloatVec vals{0.0F, -0.0F, 1.0F, -1.0F, 2.5F, -2.5F, 1e-40F, -1e-40F, kInf, -kInf, kNaN};
    FloatVec ys;
    FloatVec xs;
    for (float y : vals) {
        for (float x : vals) {
            ys.push_back(y);
            xs.push_back(x);
        }
    }
    const FloatVec out = apply512(atan2_512, ys, xs);
    for (std::size_t i = 0; i < out.size(); ++i) {
        const float ref = std::atan2(ys[i], xs[i]);
        if (std::isnan(ref) || ref == 0.0F || std::isinf(ys[i]) || std::isinf(xs[i])) {
            EXPECT_TRUE(sameFloat(out[i], ref)) << "atan2(" << ys[i] << ", " << xs[i] << ")";
        } else {
            EXPECT_LE(ulpDiff(out[i], ref), 1) << "atan2(" << ys[i] << ", " << xs[i] << ") = " << out[i]
                                               << ", expected " << ref;
        }
    }
}

TEST(AVX512MathFunTest, Atan2MatchesAtanInRightHalfPlane) {
    // For x > 0, atan2(y, x) == atan(y / x)
    const FloatVec ys = linspace(-100.0F, 100.0F, 100003);
    const FloatVec xs = linspace(0.01F, 50.0F, 100003);
    FloatVec ratio(ys.size());
    for (std::size_t i = 0; i < ys.size(); ++i) {
        ratio[i] = ys[i] / xs[i];
    }
    const FloatVec a2 = apply512(atan2_512, ys, xs);
    const FloatVec a1 = apply512(atan512, ratio);
    for (std::size_t i = 0; i < a2.size(); ++i) {
        ASSERT_TRUE(sameFloat(a2[i], a1[i])) << "y=" << ys[i] << " x=" << xs[i];
    }
}

TEST(AVX512MathFunTest, Atan2MatchesAVX2) {
    const FloatVec ys = linspace(-50.0F, 50.0F, 100003);
    const FloatVec xs = linspace(30.0F, -30.0F, 100003);
    const FloatVec a512 = apply512(atan2_512, ys, xs);
    FloatVec a256(ys.size());
    for (std::size_t i = 0; i + 8 <= ys.size(); i += 8) {
        _mm256_storeu_ps(&a256[i], avx_mathfun::atan2(_mm256_loadu_ps(&ys[i]), _mm256_loadu_ps(&xs[i])));
    }
    for (std::size_t i = 0; i < (ys.size() / 8) * 8; ++i) {
        ASSERT_LE(ulpDiff(a512[i], a256[i]), 2) << "atan2(" << ys[i] << ", " << xs[i] << ")";
    }
}

/* ---- throughput (informational) ------------------------------------------ */
// Disabled by default: it is slow under emulation and timings are only
// meaningful on real hardware.  Run with --gtest_also_run_disabled_tests.

namespace {
template <typename F>
double nsPerElement(F f, std::size_t elementsPerCall, std::size_t calls) {
    const auto t0 = std::chrono::steady_clock::now();
    f(calls);
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::nano>(t1 - t0).count() / static_cast<double>(calls * elementsPerCall);
}
} // namespace

TEST(AVX512MathFunTest, DISABLED_ThroughputVsAVX2) {
    constexpr std::size_t N = 20000000;
    const __m256 x8 = _mm256_set_ps(0.1F, 0.7F, 1.3F, 2.9F, 3.3F, 4.1F, 5.7F, 6.2F);
    const __m512 x16 = _mm512_set_ps(0.1F, 0.7F, 1.3F, 2.9F, 3.3F, 4.1F, 5.7F, 6.2F,
                                     0.2F, 0.8F, 1.4F, 2.1F, 3.4F, 4.2F, 5.8F, 6.1F);
    // tiny per-iteration nudge so the compiler can't hoist the call out of the loop
    const __m256 step8 = _mm256_set1_ps(1e-7F);
    const __m512 step16 = _mm512_set1_ps(1e-7F);
    volatile float sink = 0.0F;

#define BENCH(NAME, EXPR256, EXPR512)                                                                       \
    {                                                                                                       \
        const double t256 = nsPerElement([&](std::size_t n) {                                               \
            __m256 acc = _mm256_setzero_ps(); __m256 v = x8;                                                \
            for (std::size_t i = 0; i < n; ++i) { acc = _mm256_add_ps(acc, EXPR256); v = _mm256_add_ps(v, step8); } \
            sink = sink + _mm256_cvtss_f32(acc);                                                            \
        }, 8, N);                                                                                           \
        const double t512 = nsPerElement([&](std::size_t n) {                                               \
            __m512 acc = _mm512_setzero_ps(); __m512 v = x16;                                               \
            for (std::size_t i = 0; i < n; ++i) { acc = _mm512_add_ps(acc, EXPR512); v = _mm512_add_ps(v, step16); } \
            sink = sink + _mm512_cvtss_f32(acc);                                                            \
        }, 16, N);                                                                                          \
        std::cout << NAME << ": AVX2 " << t256 << " ns/elem, AVX-512 " << t512 << " ns/elem, speedup "    \
                  << t256 / t512 << "x" << std::endl;                                                       \
    }

    BENCH("log  ", avx_mathfun::log256_ps(v), avx512_mathfun::log512_ps(v))
    BENCH("exp  ", avx_mathfun::exp256_ps(v), avx512_mathfun::exp512_ps(v))
    BENCH("sin  ", avx_mathfun::sin256_ps(v), avx512_mathfun::sin512_ps(v))
    BENCH("cos  ", avx_mathfun::cos256_ps(v), avx512_mathfun::cos512_ps(v))
    BENCH("atan ", avx_mathfun::atan(v), avx512_mathfun::atan(v))
    BENCH("atan2", avx_mathfun::atan2(v, x8), avx512_mathfun::atan2(v, x16))
#undef BENCH
    (void)sink;
}
