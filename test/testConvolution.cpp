/**********************************************************************
 * testConvolution.cpp
 *
 * Copyright 2024, 2025 W. Greg Visotski
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 **********************************************************************/

#include <iostream>
#include <numeric>

#include "gtest/gtest.h"

#include "Aligned.h"
#include "Convolution.h"

TEST(ConvolutionTest, TestCreate) {
    //16 samples
    aligned::alignedFloatVector kernel(32, 1.0F);
    aligned::alignedFloatVector data(200, 0.0F);
    data[100] = 1.0f;

    Convolution con(kernel.data(), kernel.size());
}

TEST(ConvolutionTest, TestConvolve) {
    //16 samples
    aligned::alignedFloatVector kernel(32, 1.0F);
    aligned::alignedFloatVector data(200, 0.0F);
    data[100] = 1.0f;

    Convolution con(kernel.data(), kernel.size());

    aligned::alignedFloatVector result(data.size(), 0.0);

    con.convolve(data.data(), result.data(), data.size());

    float v = std::accumulate(result.begin(), result.end(), 0.0F);

    std::for_each(result.begin(), result.end(), [](const float& f){std::cout << f << ", ";});
    std::cout << std::endl;
    EXPECT_NEAR(v, 1.0F, 0.1F);
}