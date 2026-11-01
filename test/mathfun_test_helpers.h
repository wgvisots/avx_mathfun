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
  Test helpers shared by testAVX_Mathfun.cpp and testAVX512_Mathfun.cpp: ulp distances, accuracy
  statistics against a double-precision reference, input generators, and apply256(), which runs
  an 8-wide function over an arbitrary-length input.

  Each test file is its own executable and includes this header once, so the helpers live in an
  anonymous namespace.
*/

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <immintrin.h>

#include "gtest/gtest.h"

namespace {

using FloatVec = std::vector<float>;

constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

// Apply an 8-wide AVX2 function to an arbitrary-length input, padding the tail with 1.0.
template <typename F>
FloatVec apply256(F f, const FloatVec& in) {
    FloatVec out(in.size());
    alignas(32) float buf[8];
    for (std::size_t i = 0; i < in.size(); i += 8) {
        const std::size_t n = std::min<std::size_t>(8, in.size() - i);
        std::fill(buf, buf + 8, 1.0F);
        std::copy(in.begin() + i, in.begin() + i + n, buf);
        _mm256_store_ps(buf, f(_mm256_load_ps(buf)));
        std::copy(buf, buf + n, out.begin() + i);
    }
    return out;
}

// Map a float to an integer such that adjacent floats map to adjacent integers.
std::int64_t orderedBits(float f) {
    std::int32_t i;
    std::memcpy(&i, &f, sizeof(i));
    return (i < 0) ? -static_cast<std::int64_t>(i & 0x7FFFFFFF) : static_cast<std::int64_t>(i);
}

// Distance in ulps.  NaN vs NaN and identical infinities are 0; NaN vs number is "infinite".
std::int64_t ulpDiff(float a, float b) {
    if (std::isnan(a) || std::isnan(b)) {
        return (std::isnan(a) && std::isnan(b)) ? 0 : std::numeric_limits<std::int64_t>::max();
    }
    if (a == b) {
        return 0; // also covers +0 == -0 and equal infinities
    }
    if (std::isinf(a) || std::isinf(b)) {
        return std::numeric_limits<std::int64_t>::max();
    }
    const std::int64_t d = orderedBits(a) - orderedBits(b);
    return d < 0 ? -d : d;
}

// Exact comparison that also distinguishes +0/-0 and treats all NaNs as equal.
::testing::AssertionResult sameFloat(float actual, float expected) {
    if (std::isnan(expected) && std::isnan(actual)) {
        return ::testing::AssertionSuccess();
    }
    if (actual == expected && std::signbit(actual) == std::signbit(expected)) {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << "got " << actual << " expected " << expected;
}

struct ErrorStats {
    std::int64_t maxUlp{0};
    float maxUlpAt{0.0F};
    double maxAbs{0.0};
    float maxAbsAt{0.0F};
};

// Compare results against a double-precision reference.
ErrorStats measure(const FloatVec& in, const FloatVec& out, double (*ref)(double)) {
    ErrorStats s;
    for (std::size_t i = 0; i < in.size(); ++i) {
        const double r = ref(static_cast<double>(in[i]));
        const std::int64_t u = ulpDiff(out[i], static_cast<float>(r));
        if (u > s.maxUlp) {
            s.maxUlp = u;
            s.maxUlpAt = in[i];
        }
        const double a = std::fabs(static_cast<double>(out[i]) - r);
        if (a > s.maxAbs) {
            s.maxAbs = a;
            s.maxAbsAt = in[i];
        }
    }
    return s;
}

std::int64_t maxUlpBetween(const FloatVec& a, const FloatVec& b, float* worstIdxValue = nullptr,
                           const FloatVec* in = nullptr) {
    std::int64_t m = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const std::int64_t u = ulpDiff(a[i], b[i]);
        if (u > m) {
            m = u;
            if (worstIdxValue != nullptr && in != nullptr) {
                *worstIdxValue = (*in)[i];
            }
        }
    }
    return m;
}

FloatVec linspace(float lo, float hi, std::size_t n) {
    FloatVec v(n);
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = lo + (hi - lo) * static_cast<float>(static_cast<double>(i) / static_cast<double>(n - 1));
    }
    return v;
}

// Every float whose bit pattern is a multiple of 'stride' in [lo, hi] (lo, hi > 0).
FloatVec bitSweep(float lo, float hi, std::uint32_t stride) {
    std::uint32_t a;
    std::uint32_t b;
    std::memcpy(&a, &lo, 4);
    std::memcpy(&b, &hi, 4);
    FloatVec v;
    v.reserve((b - a) / stride + 1);
    for (std::uint64_t i = a; i <= b; i += stride) {
        const std::uint32_t u = static_cast<std::uint32_t>(i);
        float f;
        std::memcpy(&f, &u, 4);
        v.push_back(f);
    }
    return v;
}

FloatVec negated(const FloatVec& v) {
    FloatVec r(v.size());
    std::transform(v.begin(), v.end(), r.begin(), [](float f) { return -f; });
    return r;
}

FloatVec concat(FloatVec a, const FloatVec& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

double refLog(double x) { return std::log(x); }
double refExp(double x) { return std::exp(x); }
double refSin(double x) { return std::sin(x); }
double refCos(double x) { return std::cos(x); }
[[maybe_unused]] double refAtan(double x) { return std::atan(x); }

} // namespace
