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
 */
#ifdef __GNUG__
#define ALIGN32_BEG
#define ALIGN32_END __attribute__ ((aligned (32)))
#define ALIGN16_BEG
#define ALIGN16_END __attribute__ ((aligned (16)))
#define RESTRICT __restrict__
#define ALWAYS_INLINE __attribute__ ((always_inline))
    template<typename T>
    inline T *allocateAligned(const aligned::Alignment align, const std::size_t numElements){
        std::size_t len = numElements * sizeof(T);
        len = len + ((std::size_t)align - (len % (std::size_t)align));
        return (T *) std::aligned_alloc((std::size_t)align, len);
    }

    template<typename T>
    inline void freeAligned(T *t)
    {
        free(t);
    }

#else
#define ALIGN32_BEG __declspec(align(32))
#define ALIGN32_END
#define ALIGN16_BEG __declspec(align(16))
#define ALIGN16_END
#define RESTRICT __restrict
#define ALWAYS_INLINE
    template<typename T>
    inline T *allocateAligned(const unsigned int align, const unsigned int numElements)
    {
        uint32_t len = numElements * sizeof(T);
        len = len + (align - (len % align));
        return (T *)_aligned_malloc(len, align);
    }
    template<typename T>
    inline void freeAligned(T *t)
    {
        _aligned_free(t);
    }
#endif

    //provided mostly as examples:
    template< class T > using alignedVector = std::vector< T, aligned::AlignedAllocator<T, Alignment::AVX> >;
    using alignedComplexVector = alignedVector<std::complex<float> >;
    using alignedFloatVector   = alignedVector<float>;

    //NOTE: If you are getting warning: ignoring attributes on template argument ‘__m256’, then you have to carefully handle
    //      going between __m256 and float[8] using this structure.  Intel tried to help us with this by using the
    //      __may_alias__ attribute, but your compiler can't handle that as a template argument.  Therefore, strict
    //      aliasing rules apply to __m256s that are in this container.
    using alignedVecVector     = alignedVector<__m256>;
};


