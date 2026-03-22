/*
  Copyright (C) 2025 - 2026 W. Greg Visotski

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

#include <iostream>
#include <cmath>
#include <chrono>
#include <limits>

#include "gtest/gtest.h"

#include "../src/avx_mathfun.h"

namespace {
std::size_t NUM_ITERS{100000000};
const std::size_t PRIME{2251};
constexpr float scale = (1.0F / static_cast<float>(std::numeric_limits<std::int32_t>::max() - 1)) * (2.0F*static_cast<float>(M_PI));
std::vector<__m256> data;

__m256 makeData() {
    __m256 result;
    float* rr = reinterpret_cast<float*>(&result);
    //Yes, I'm calling 'random' and not 'rand' or c++11+ RNG.  My BSD-isms coming out...
    rr[0] = static_cast<float>(random()) * scale;
    rr[1] = static_cast<float>(random()) * scale;
    rr[2] = static_cast<float>(random()) * scale;
    rr[3] = static_cast<float>(random()) * scale;
    rr[4] = static_cast<float>(random()) * scale;
    rr[5] = static_cast<float>(random()) * scale;
    rr[6] = static_cast<float>(random()) * scale;
    rr[7] = static_cast<float>(random()) * scale;
    return result;
}

void makeAllData() {
    if (data.empty()) {
        srandom((unsigned int)std::chrono::system_clock::now().time_since_epoch().count());
        data.reserve(PRIME);
        for (std::size_t i = 0; i < PRIME; i++) {
            data.push_back(makeData());
        }
    }
}


// std::size_t NUM_ITERS{1000000};
}
TEST(MathFunTest, TestATan) {
    float v[8] ALIGN32_END;
    for (std::size_t i = 0; i < 8; ++i) {
        v[i]= M_PI / (float) (i+1);
    }

    __m256 vv = _mm256_load_ps(v);
    __m256 result = avx_mathfun::atan(vv);

    float res[8] ALIGN32_END;
    _mm256_store_ps(res, result);

    for (std::size_t i = 0; i < 8; ++i) {
        //TODO -> this is some huge error...
        EXPECT_NEAR(res[i], std::atan(v[i]), 1e-5);
    }
}

TEST(MathFunTest, TestATan_no2) {
    float x[8];
    for (std::int32_t i = 0; i < 8; ++i) {
        float xx = (float)(i-4);
        float yy = (float)((7-i)-4);
        x[i] = xx / yy;
    }

    __m256 xx = _mm256_load_ps(x);

    __m256 zz = avx_mathfun::atan(xx);

    float result[8];
    _mm256_store_ps(result, zz);
    for (std::size_t i = 0; i < 8; ++i) {
        float truth = std::atan(x[i]);
        EXPECT_NEAR(result[i], truth, 1e-5);
    }
}

TEST(MathFunTest, TestATan2) {
    float x[8];
    float y[8];
    float truth[8];
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

    float result[8];
    _mm256_store_ps(result, zz);
    for (std::size_t i = 0; i < 8; ++i) {
        EXPECT_NEAR(result[i], truth[i], 1e-5);
    }
}

TEST(MathFunTest, TestATan2_no2) {
    float x[8];
    float y[8];
    float truth[8];

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

            float result[8];
            _mm256_store_ps(result, zz);
            for (std::size_t i = 0; i < 8; ++i) {
                EXPECT_NEAR(result[i], truth[i], 1e-5);
            }
        }

    }
}

TEST(MathFunTest, TestATanVecSpeed) {
    makeAllData();
    const __m256* d = data.data();
    float r{0.0};
    for (std::size_t i = 0; i < NUM_ITERS; i++) {
        __m256 result = avx_mathfun::atan(d[i%PRIME]);
        float* rr = reinterpret_cast<float*>(&result);
        r += rr[0];
        r += rr[1];
        r += rr[2];
        r += rr[3];
        r += rr[4];
        r += rr[5];
        r += rr[6];
        r += rr[7];
    }

    EXPECT_GT(r, 1.0F);
    // EXPECT_GE(avx_mathfun::counter, NUM_ITERS);
}

TEST(MathFunTest, TestATanSpeed) {
    __m256 c;
    const __m256* d = data.data();
    float r{0.0};
    for (std::size_t i = 0; i < NUM_ITERS; i++) {
        avx_mathfun::atan(d[i%PRIME], c);
        float* rr = reinterpret_cast<float*>(&c);
        r += rr[0];
        r += rr[1];
        r += rr[2];
        r += rr[3];
        r += rr[4];
        r += rr[5];
        r += rr[6];
        r += rr[7];
    }

    EXPECT_GT(r, 1.0F);
}

TEST(MathFunTest, TestATan2VecSpeed) {
    float x[8] ALIGN32_END;
    float y[8] ALIGN32_END;
    for (std::int32_t i = 0; i < 8; ++i) {
        float xx = (float)(i-4);
        float yy = (float)((7-i)-4);
        x[i] = xx;
        y[i] = yy;
    }

    __m256 xx = _mm256_load_ps(x);
    __m256 yy = _mm256_load_ps(y);

    float r{0.0};

    for (std::size_t i = 0; i < NUM_ITERS; i++) {
        const __m256 result = avx_mathfun::atan2(xx, yy);
        const float* rr = reinterpret_cast<const float*>(&result);
        for (std::size_t j = 0; j < 8; ++j) {
            r += rr[j];
        }
    }
    EXPECT_NE(r, 0.0F);
}
TEST(MathFunTest, TestATan2Speed_v2) {
    float x[8] ALIGN32_END;
    float y[8] ALIGN32_END;
    for (std::int32_t i = 0; i < 8; ++i) {
        float xx = (float)(i-4);
        float yy = (float)((7-i)-4);
        x[i] = xx;
        y[i] = yy;
    }

    __m256 xx ALIGN32_END = _mm256_load_ps(x);
    __m256 yy ALIGN32_END = _mm256_load_ps(y);
    __m256 zz ALIGN32_END;
    float r{0.0};
    for (std::size_t i = 0; i < NUM_ITERS; i++) {
        avx_mathfun::atan2(xx, yy, zz);
        const float* rr = reinterpret_cast<float*>(&zz);
        for (std::size_t j = 0; j < 8; ++j) {
            r += rr[j];
        }
    }

    EXPECT_NE(r, 0.0F);
}