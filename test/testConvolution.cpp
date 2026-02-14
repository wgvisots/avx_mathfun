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
#include <numeric>

#include "gtest/gtest.h"

#include "../src/Aligned.h"
#include "../src/Convolution.h"

TEST(ConvolutionTest, TestCreate) {
    //16 samples
    aligned::alignedFloatVector kernel(32, 1.0F);
    aligned::alignedFloatVector data(200, 0.0F);
    data[100] = 1.0f;

    Convolution con(kernel.data(), kernel.size());
}

/**
 * What we're going to do here is make a kernel where all values are equal.  The Convolution will normalize this to
 * a total of 1.  This'll have the effect of outputting, for each sample under the kernel, the average of all samples
 * under the filter.
 *
 * We are then going to make our data all zeros, with a single 1 value.  This means that after we filter, we can sum
 * the result and get a value of 1 (with floating point rounding error).
 */
TEST(ConvolutionTest, TestConvolve) {
    //16 samples
    aligned::alignedFloatVector kernel(32, 1.0F);
    aligned::alignedFloatVector data(200, 0.0F);
    data[100] = 1.0f;

    Convolution con(kernel.data(), kernel.size());

    aligned::alignedFloatVector result(data.size(), 0.0);

    con.convolve(data.data(), result.data(), data.size());

    float v = std::accumulate(result.begin(), result.end(), 0.0F);

    // std::for_each(result.begin(), result.end(), [](const float& f){std::cout << f << ", ";});
    // std::cout << std::endl;
    EXPECT_NEAR(v, 1.0F, 0.1F);
}
