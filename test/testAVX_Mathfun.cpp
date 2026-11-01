/*
  Copyright (C) 2024 - 2026 W. Greg Visotski

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

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numbers>
#include <random>
#include <vector>

#include "gtest/gtest.h"

#include "../src/avx_mathfun.h"

namespace {
constexpr std::size_t NUM_ITERS{100000000};
constexpr std::size_t PRIME{2251};

// PRIME vectors of 8 random angles in [0, 2*pi), stored as plain floats and
// loaded with _mm256_loadu_ps.  (A std::vector<__m256> drops __m256's
// may_alias attribute, and reading its lanes through a float* is not valid C++.)
const std::vector<float>& benchData() {
    static const std::vector<float> data = [] {
        std::mt19937 rng(20240101U);  // fixed seed: reproducible runs
        std::uniform_real_distribution<float> dist(0.0F, 2.0F * std::numbers::pi_v<float>);
        std::vector<float> d(PRIME * 8);
        for (float& f : d) {
            f = dist(rng);
        }
        return d;
    }();
    return data;
}

__m256 benchVec(std::size_t i) {
    return _mm256_loadu_ps(&benchData()[(i % PRIME) * 8]);
}

float sumLanes(__m256 v) {
    alignas(32) float f[8];
    _mm256_store_ps(f, v);
    float r = 0.0F;
    for (float x : f) {
        r += x;
    }
    return r;
}
}

TEST(MathFunTest, TestATan) {
    alignas(32) float v[8];
    for (std::size_t i = 0; i < 8; ++i) {
        v[i] = std::numbers::pi_v<float> / static_cast<float>(i + 1);
    }

    __m256 vv = _mm256_load_ps(v);
    __m256 result = avx_mathfun::atan(vv);

    alignas(32) float res[8];
    _mm256_store_ps(res, result);

    for (std::size_t i = 0; i < 8; ++i) {
        //TODO -> this is some huge error...
        EXPECT_NEAR(res[i], std::atan(v[i]), 1e-5);
    }
}

TEST(MathFunTest, TestATan_no2) {
    alignas(32) float x[8];
    for (std::int32_t i = 0; i < 8; ++i) {
        float xx = (float)(i-4);
        float yy = (float)((7-i)-4);
        x[i] = xx / yy;
    }

    __m256 xx = _mm256_load_ps(x);

    __m256 zz = avx_mathfun::atan(xx);

    alignas(32) float result[8];
    _mm256_store_ps(result, zz);
    for (std::size_t i = 0; i < 8; ++i) {
        float truth = std::atan(x[i]);
        EXPECT_NEAR(result[i], truth, 1e-5);
    }
}

TEST(MathFunTest, TestATan2) {
    alignas(32) float x[8];
    alignas(32) float y[8];
    alignas(32) float truth[8];
    for (std::int32_t i = 0; i < 8; ++i) {
        const float xx = static_cast<float>(i - 4);
        const float yy = static_cast<float>((7 - i) - 4);
        x[i] = xx;
        y[i] = yy;
        truth[i] = std::atan2(yy,xx);
    }

    const __m256 xx = _mm256_load_ps(x);
    const __m256 yy = _mm256_load_ps(y);

    const __m256 zz = avx_mathfun::atan2(yy, xx);

    alignas(32) float result[8];
    _mm256_store_ps(result, zz);
    for (std::size_t i = 0; i < 8; ++i) {
        EXPECT_NEAR(result[i], truth[i], 1e-5);
    }
}

TEST(MathFunTest, TestATan2_no3) {
    // Dense grid over [-12, 12) x [-12, 12) with step 0.001 (~576M points).
    // Points are streamed through an 8-float buffer instead of being stored, so
    // memory use is constant, and errors are summarised in one assertion so a
    // regression doesn't print millions of lines.
    float ys[8];
    float xs[8];
    float rf[8];
    std::size_t n = 0;
    std::size_t count = 0;
    std::size_t failures = 0;
    float worstErr = 0.0F;
    float worstY = 0.0F;
    float worstX = 0.0F;

    const auto flush = [&]() {
        _mm256_storeu_ps(rf, avx_mathfun::atan2(_mm256_loadu_ps(ys), _mm256_loadu_ps(xs)));
        for (std::size_t j = 0; j < n; ++j) {
            const float err = std::fabs(rf[j] - std::atan2(ys[j], xs[j]));
            if (!(err <= 1e-5F)) {  // also catches NaN
                ++failures;
            }
            if (!(err <= worstErr)) {
                worstErr = err;
                worstY = ys[j];
                worstX = xs[j];
            }
        }
        count += n;
        n = 0;
    };

    for (float f1 = -12.0F; f1 < 12.0F; f1 += 0.001F) {
        for (float f2 = -12.0F; f2 < 12.0F; f2 += 0.001F) {
            ys[n] = f1;
            xs[n] = f2;
            if (++n == 8) {
                flush();
            }
        }
    }
    if (n != 0) {
        for (std::size_t j = n; j < 8; ++j) {  // pad the last partial vector
            ys[j] = 1.0F;
            xs[j] = 1.0F;
        }
        flush();
    }

    EXPECT_EQ(failures, 0U) << failures << " of " << count << " points off by more than 1e-5; worst error "
                            << worstErr << " at atan2(" << worstY << ", " << worstX << ")";
}

TEST(MathFunTest, TestATan2_no2) {
    alignas(32) float x[8];
    alignas(32) float y[8];
    alignas(32) float truth[8];

    for (float fx = -1.0; fx <= 1.0; fx += 0.01) {
        for (std::size_t i = 0; i < 8; ++i) {
            x[i] = fx + (float)i * 0.002f;
        }
        for (float fy = -1.0; fy <= 1.0; fy += 0.01) {
            for (std::size_t j = 0; j < 8; ++j) {
                y[j] = fy + (float)j * 0.002f;
                const float s = x[j] + y[j];
                truth[j] = std::atan2(y[j]/s,x[j]/s);
            }

            __m256 xx = _mm256_load_ps(x);
            __m256 yy = _mm256_load_ps(y);
            const __m256 sum = _mm256_add_ps(xx,yy);
            xx = _mm256_div_ps(xx, sum);
            yy = _mm256_div_ps(yy, sum);

            const __m256 zz = avx_mathfun::atan2(yy, xx);

            alignas(32) float result[8];
            _mm256_store_ps(result, zz);
            for (std::size_t i = 0; i < 8; ++i) {
                EXPECT_NEAR(result[i], truth[i], 1e-5);
            }
        }

    }
}

// Regression test: atan2 used to return the wrong quadrant when x < 0 and y < 0,
// and NaN for atan2(0, 0).  Covers all four quadrants, the axes, signed zeros,
// infinities and NaN, compared against std::atan2.
TEST(MathFunTest, TestATan2QuadrantsAndSpecialValues) {
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float> vals{0.0F, -0.0F, 1.0F, -1.0F, 0.3F, -0.3F, 2.5F, -2.5F,
                                  1e-40F, -1e-40F, 1e30F, -1e30F, inf, -inf, nan};
    std::vector<float> ys;
    std::vector<float> xs;
    for (float yv : vals) {
        for (float xv : vals) {
            ys.push_back(yv);
            xs.push_back(xv);
        }
    }
    while (ys.size() % 8 != 0) {
        ys.push_back(1.0F);
        xs.push_back(1.0F);
    }

    alignas(32) float res[8];
    for (std::size_t i = 0; i < ys.size(); i += 8) {
        _mm256_store_ps(res, avx_mathfun::atan2(_mm256_loadu_ps(&ys[i]), _mm256_loadu_ps(&xs[i])));
        for (std::size_t j = 0; j < 8; ++j) {
            const float y = ys[i + j];
            const float x = xs[i + j];
            const float truth = std::atan2(y, x);
            if (std::isnan(truth)) {
                EXPECT_TRUE(std::isnan(res[j])) << "atan2(" << y << ", " << x << ") = " << res[j];
            } else {
                EXPECT_NEAR(res[j], truth, 1e-6) << "atan2(" << y << ", " << x << ")";
                EXPECT_EQ(std::signbit(res[j]), std::signbit(truth)) << "sign of atan2(" << y << ", " << x << ")";
            }
        }
    }
}

TEST(MathFunTest, TestATanVecSpeed) {
    float r{0.0};
    for (std::size_t i = 0; i < NUM_ITERS; i++) {
        r += sumLanes(avx_mathfun::atan(benchVec(i)));
    }

    EXPECT_GT(r, 1.0F);
}

TEST(MathFunTest, TestATanSpeed) {
    __m256 c;
    float r{0.0};
    for (std::size_t i = 0; i < NUM_ITERS; i++) {
        avx_mathfun::atan(benchVec(i), c);
        r += sumLanes(c);
    }

    EXPECT_GT(r, 1.0F);
}

// The atan2 speed tests take their inputs from benchData(), which changes every
// iteration, so the compiler can't evaluate the call once and hoist it out of the
// loop.  Shifting the angles by -pi gives values in [-pi, pi), so all four
// quadrants are exercised; y and x come from different offsets in the table.
namespace {
__m256 benchY(std::size_t i) {
    return _mm256_sub_ps(benchVec(i), _mm256_set1_ps(std::numbers::pi_v<float>));
}
__m256 benchX(std::size_t i) {
    return _mm256_sub_ps(benchVec(i + 1000), _mm256_set1_ps(std::numbers::pi_v<float>));
}
}

TEST(MathFunTest, TestATan2VecSpeed) {
    float r{0.0};
    for (std::size_t i = 0; i < NUM_ITERS; i++) {
        r += sumLanes(avx_mathfun::atan2(benchY(i), benchX(i)));
    }
    EXPECT_NE(r, 0.0F);
}

TEST(MathFunTest, TestATan2Speed_v2) {
    __m256 zz;
    float r{0.0};
    for (std::size_t i = 0; i < NUM_ITERS; i++) {
        avx_mathfun::atan2(benchY(i), benchX(i), zz);
        r += sumLanes(zz);
    }

    EXPECT_NE(r, 0.0F);
}