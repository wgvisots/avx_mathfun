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
#include <immintrin.h>

#include "Aligned.h"

class Convolution {
public:
    /**
     *
     * @param kernel Your kernel, composed of std::complex<float>, but provided as a vector of floats (I,Q,I,Q...)
     * @param kernelSize Number of floats in the kernel
     */
    Convolution (const float* kernel, std::size_t kernelSize);
    ~Convolution() = default;

    /**
     *
     * @param in array of std::complex<float> (I,Q,I,Q...), as an array of 'size' floats
     * @param out the complex convolution result, as an array of floats - truncated to the same length as the input
     * @param size number of floats (not complexes) in 'in'
     */
    void convolve(const float* RESTRICT in, float* RESTRICT out, std::size_t size);

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