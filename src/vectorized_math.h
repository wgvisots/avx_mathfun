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

#include <cstdint>

#include "Aligned.h"

/**
 * This is a set of NON-PORTABLE X64 vectorized math/conversion functions.  Don't try to use this
 * on old X64 (pre-AVX2), ARM, Power, or other ISA hardware.
 */
namespace vectorized{
    /**
     * Convert an array of std::int16_ts to an array of floats
     *
     * @param in - a pointer to an array of std::uint16_t of length 'length' to be read from
     * @param out - a pointer to an array of float of length 'length' to be written to
     * @param length - the length of in and out
     */
    void shortToFloat(const std::int16_t * RESTRICT in, float * RESTRICT out,  std::size_t length);

    /**
     * Convert an array of floats to an array of std::uint16_t
     *
     * The values of 'in' should be in the valid range of a short, and are rounded/saturated otherwise.
     *
     * @param in - a pointer to an array of float of length 'length' to be read from
     * @param out - a pointer to an array of std::uint16_t of length 'length' to be written to
     * @param length - the length of in and out
     */
    void floatToShort(const float * RESTRICT in, std::int16_t * RESTRICT out,  std::size_t length);

    /**
     * Rewrite the array in the opposite order
     * NOTE: Assumes aligned memory!
     * @param in - an array
     * @param out - will be filled with in, only backwards
     * @param size
     */
    void reverse(const float * RESTRICT in, float * RESTRICT out,  std::size_t length);

    /**
     * Multiply every value in 'io' by 'num', and save back into 'io'
     * NOTE: Assumes aligned memory!
     * @param io
     * @param num
     * @param size
     */
    void multiply(  float* RESTRICT io,  float num,   std::size_t size);

    /**
     * This assumes that 'io' is an array containing IQ data, arranged as I,Q,I,Q,...I,Q
     * It will multiply every I by 'ival', and every Q by 'qval', and write back to 'io'.
     * NOTE: Assumes aligned memory!
     * @param io
     * @param ival
     * @param qval
     * @param size
     */
    void multiplyIQ(float* RESTRICT io,  float ival,  float qval,  std::size_t size);

    /**
     * Add the elements, element by element, of the two arrays ('io' and 'secon'), and write back to 'io'.
     * NOTE: Assumes aligned memory!
     * @param io
     * @param second
     * @param size
     */
    void add(   std::int16_t* RESTRICT io, const std::int16_t* RESTRICT second,  std::size_t size); /*32 byte aligned*/

    /**
     * Add the elements, element by element, of the two arrays ('io' and 'secon'), and write back to 'io'.
     * NOTE: Assumes un-aligned memory!
     * @param io
     * @param second
     * @param size
     */
    void addUNA(std::int16_t* RESTRICT io, const std::int16_t* RESTRICT second,  std::size_t size); /*32 byte un-aligned*/

    /**
     * Add the elements, element by element, of the two arrays ('io' and 'secon'), and write back to 'io'.
     * NOTE: Assumes aligned memory!
     * @param io
     * @param second
     * @param size
     */
    void add(   float*        RESTRICT io, const float*        RESTRICT second,  std::size_t size); /*32 byte aligned*/

    /**
     * Add the elements, element by element, of the two arrays ('io' and 'secon'), and write back to 'io'.
     * NOTE: Assumes un-aligned memory!
     * @param io
     * @param second
     * @param size
     */
    void addUNA(float*        RESTRICT io, const float*        RESTRICT second,  std::size_t size); /*32 byte un-aligned*/

    /**
     * Take in an array of std::complex<float> (as a float*), and output an array of amplitude/phase.
     *
     * @param in I,Q,I,Q...
     * @param out Amp,Phase,Amp,Phase...
     * @param size the number of floats in 'in' (2X number of samples)
     */
    void iqToAmpAndPhase(const float * RESTRICT in, float * RESTRICT out, std::size_t size);

    /**
     * Simple helper to give you a constant __m256 (array of 8 floats) that are all set to 0.0F.
     * @return
     */
    const __m256& zeroVec();

}
