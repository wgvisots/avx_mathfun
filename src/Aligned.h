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

#include <cstdlib>
#include <complex>
#include <vector>
#include <immintrin.h>

#include "AlignedAllocator.h"

namespace aligned
{

    /*
     * A bit on alignment:
     * All __m256 variables need to be aligned on 32 byte boundaries.  This
     * *should* be done with 'alignof(__m256) varName;', but if
     * 'alignof(std::max_align_t) < alignof(__m256)', then the behavior
     * of the alignof is undefined, and it might not throw a warning.
     *
     * On GCC 4.9 and VC, max_align_t gives you 16 and not 32, so you need to
     * use the compiler specific '__attribute__ ((aligned (32)))' (trailing).
     * This gives you a maximum alignment of '__BIGGEST_ALIGNMENT__' (32).
     *
     * Same holds true of 64, for __m512
     */
    #ifndef __GNUG__
    #error "Building on Windows is not currently supported.  This library uses multiple GCC compiler extensions."
    #endif

    #define ALIGN64_END   __attribute__ ((aligned (64)))
    #define ALIGN32_END   __attribute__ ((aligned (32)))
    #define ALIGN16_END   __attribute__ ((aligned (16)))
    #define ALWAYS_INLINE __attribute__ ((always_inline))
    #define RESTRICT      __restrict__

    template<typename T>
    T *allocateAligned(const aligned::Alignment align, const std::size_t numElements){
        std::size_t len = numElements * sizeof(T);
        len = len + (static_cast<std::size_t>(align) - (len % static_cast<std::size_t>(align)));
        return static_cast<T*>(std::aligned_alloc(static_cast<std::size_t>(align), len));
    }

    template<typename T>
    void freeAligned(T *t)
    {
        free(t);
    }

    //provided mostly as examples:
    template< class T > using alignedVector = std::vector< T, aligned::AlignedAllocator<T, Alignment::AVX> >;
    using alignedComplexVector = alignedVector<std::complex<float> >;
    using alignedFloatVector   = alignedVector<float>;

    //NOTE: If you are getting warning: ignoring attributes on template argument ‘__m256’, then you have to carefully handle
    //      going between __m256 and float[8] using this structure.  Intel tried to help us with this by using the
    //      __may_alias__ attribute, but your compiler can't handle that as a template argument.  Therefore, strict
    //      aliasing rules apply to __m256s that are in this container.
    //
    //      More to the point: the attribute that is getting ignored is __may_alias__.
    using alignedVecVector     = alignedVector<__m256>;
};


