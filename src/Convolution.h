/*
Copyright (C) 2025 W. Greg Visotski

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

#pragma once

#include <memory>

#include "Aligned.h"

/**
 * This is a tool for performing complex convolution.  The intent is for this to be used on digital RF data (AKA: IQ data).
 * Each "sample" consists of two floats: an in-phase and quadrature (IQ).  They can be considered a std::complex<float>,
 * but I use a flat array of floats for performance reasons.
 *
 * This class SIMD vectorizes the convolution, which gives some pretty big performance gains, but comes with some
 * caveats:
 * 1) All your inputs/outputs shall be 32-byte aligned.  ('kernel', 'in', and 'out')
 * 2) Short kernels get less performance boosts.
 * 3) Kernels should, but don't need to be, a multiple of 4 samples long.
 * 4) Kernels less than 4 samples shall not be used.
 * 4) Data ('in'/'out') should be much longer than the kernel.
 * 5) Note the use of RESTRICT keyword below - it is your responsibility to obey it.
 *
 * Sizes are given in the number of numbers in each array, not the number samples.  Sizes must, therefore, be an even
 * number.
 */
class Convolution {
public:
    /**
     * Create a convolution object with a given kernel.
     *
     * @param kernel Your kernel, composed of std::complex<float>, but provided as a vector of floats (I,Q,I,Q...)
     * @param kernelSize Number of floats in the kernel (2X number of samples)
     */
    Convolution (const float* kernel, std::size_t kernelSize);
    ~Convolution() = default;

    /**
     *
     * @param in array of std::complex<float> (I,Q,I,Q...), as an array of 'size' floats
     * @param out the complex convolution result, as an array of floats - truncated to the same length as the input
     * @param size number of floats (not complexes) in 'in' (2X number of samples)
     */
    void convolve(const float* __restrict__ in, float* __restrict__ out, std::size_t size);

private:
    const std::size_t kernelLength; // in floats, not complexes
    const aligned::alignedFloatVector theKernel;

    //NOTE: Using raw pointers here, because some compiler versions keep ignoring attributes on __m256 when I try to
    //      use them in an aligned::alignedVector.  And other unclear warnings when using a std::unique_ptr.
    aligned::alignedVecVector kernelReverseUV;
    aligned::alignedVecVector kernelReverseVU;

    //I'm not sure about this.  This is used for continuation between calls to convolve.
    aligned::alignedFloatVector leftover;

    aligned::alignedFloatVector dataAligned[4];
};