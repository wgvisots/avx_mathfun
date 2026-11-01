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
#include <iostream>
#include <numbers>
#include <random>
#include <vector>

#include "../src/avx_mathfun.h"

namespace {
    constexpr std::size_t NUM_ITERS{1000000};
    constexpr std::size_t PRIME{2251};

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

// Vectors are stored as plain floats (8 per vector) and loaded with _mm256_loadu_ps.
std::vector<float> makeData(std::size_t numVecs) {
    std::mt19937 rng(static_cast<std::mt19937::result_type>(
        std::chrono::system_clock::now().time_since_epoch().count()));
    std::uniform_real_distribution<float> dist(0.0F, 2.0F * std::numbers::pi_v<float>);
    std::vector<float> data(numVecs * 8);
    for (float& f : data) {
        f = dist(rng);
    }
    // first vector: fixed, easy to eyeball values
    const float fixed[8] = {0.0F, 0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 0.8F, 0.9F};
    for (std::size_t i = 0; i < 8; ++i) {
        data[i] = fixed[i];
    }
    return data;
}

float testVec(const std::vector<float>& data) {
    const std::size_t size = data.size() / 8;
    float r{0.0};
    for (std::size_t i = 0; i < NUM_ITERS; i++) {
        r += sumLanes(avx_mathfun::atan(_mm256_loadu_ps(&data[(i % size) * 8])));
    }
    return r;
}

float testNonVec(const std::vector<float>& data) {
    const std::size_t size = data.size() / 8;
    __m256 c;
    float r{0.0};
    for (std::size_t i = 0; i < NUM_ITERS; i++) {
        avx_mathfun::atan(_mm256_loadu_ps(&data[(i % size) * 8]), c);
        r += sumLanes(c);
    }
    return r;
}

void compare(const __m256& data) {
    alignas(32) float d[8];
    alignas(32) float dd[8];
    _mm256_store_ps(d, data);
    _mm256_store_ps(dd, avx_mathfun::atan(data));

    std::cout << "################################" << std::endl;
    for (std::size_t i = 0; i < 8; i++) {
        std::cout << d[i] << ",\t " << dd[i] << ",\t " << std::atan(d[i]) << std::endl;
    }
    std::cout << "################################" << std::endl;
}

int main() {
    const std::vector<float> vec = makeData(PRIME);

    avx_mathfun::printVec(_mm256_loadu_ps(vec.data()));
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    std::cout << testVec(vec) << std::endl;
    std::chrono::system_clock::time_point mid = std::chrono::system_clock::now();
    std::cout << testNonVec(vec) << std::endl;
    std::chrono::system_clock::time_point stop = std::chrono::system_clock::now();

    std::chrono::system_clock::duration nonVecTime = stop - mid;
    std::chrono::system_clock::duration vecTime    = mid - start;

    std::cout << "Non vec time: "    << std::chrono::duration_cast<std::chrono::milliseconds>(nonVecTime).count() << " milliseconds" <<  std::endl;
    std::cout << "Vectorized time: " << std::chrono::duration_cast<std::chrono::milliseconds>(vecTime).count()    << " milliseconds" <<  std::endl;

    // compare(_mm256_loadu_ps(vec.data()));

    return 0;
}
