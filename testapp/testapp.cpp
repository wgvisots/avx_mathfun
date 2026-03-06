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
#include <cstdint>
#include <limits>
#include <chrono>

#include "../src/avx_mathfun.h"

namespace {
    std::size_t NUM_ITERS{1000000};
    const std::size_t PRIME{2251};
    constexpr float scale = (1.0F / static_cast<float>(std::numeric_limits<std::int32_t>::max() - 1)) * (2.0F*static_cast<float>(M_PI));
}

/**
 * This fills the return with random numbers from 0 -> 2PI.
 * The point of doing all this weird work is to confuse compiler optimizer as much as possible.
 * @return
 */
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
float testVec(const __m256* data, const std::size_t size) {
    float r{0.0};
    for (std::size_t i = 0; i < NUM_ITERS; i++) {
        __m256 result = avx_mathfun::atan(data[i%size]);
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

    return r;
}

float testNonVec(const __m256* data, const std::size_t size) {
    __m256 c;
    float r{0.0};
    for (std::size_t i = 0; i < NUM_ITERS; i++) {
        avx_mathfun::atan(data[i%size], c);
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

    return r;
}

/**
 * The whole point of this application is to make something that you can run in valgrind and get good profiling information.
 * Some profiling information is lost (or at least harder to parse out) when running inside gtest.
 *
 * The application doesn't do anything but spin some cycles.
 *
 * @return 0
 */
int main() {
    srandom((unsigned int)std::chrono::system_clock::now().time_since_epoch().count());
    std::vector<__m256> vec;
    vec.reserve(PRIME);
    for (std::size_t i = 0; i < PRIME; i++) {
        vec.push_back(makeData());
    }

    avx_mathfun::printVec(vec[0]);
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    std::cout << testVec(vec.data(), vec.size()) << std::endl;
    std::chrono::system_clock::time_point mid = std::chrono::system_clock::now();
    std::cout << testNonVec(vec.data(), vec.size()) << std::endl;
    std::chrono::system_clock::time_point stop = std::chrono::system_clock::now();

    std::chrono::system_clock::duration nonVecTime = stop - mid;
    std::chrono::system_clock::duration vecTime    = mid - start;

    std::cout << "Non vec time: "    << std::chrono::duration_cast<std::chrono::milliseconds>(nonVecTime) << " milliseconds" <<  std::endl;
    std::cout << "Vectorized time: " << std::chrono::duration_cast<std::chrono::milliseconds>(vecTime)    << " milliseconds" <<  std::endl;

    return 0;
}