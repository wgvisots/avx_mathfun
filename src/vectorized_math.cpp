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

#include <cassert>
#include <mutex>

#include <immintrin.h>

#include "avx_mathfun.h"

#include "Aligned.h"
#include "vectorized_math.h"

namespace{
    inline std::size_t doVectorizedMultiply(float* __restrict__ io, const __m256& mul , const std::size_t size){
        assert(size >= 64);
        //    __m256 mul ALIGN32_END;
        __m256 f0  ALIGN32_END;
        __m256 f1  ALIGN32_END;
        __m256 f2  ALIGN32_END;
        __m256 f3  ALIGN32_END;
        __m256 f4  ALIGN32_END;
        __m256 f5  ALIGN32_END;
        __m256 f6  ALIGN32_END;
        __m256 f7  ALIGN32_END;

        const std::size_t stop = size - 64;
        std::size_t i = 0;
        for (; i < stop; i += 64)
        {
            f0 = _mm256_load_ps(io + i + 0);
            f1 = _mm256_load_ps(io + i + 8);
            f2 = _mm256_load_ps(io + i + 16);
            f3 = _mm256_load_ps(io + i + 24);
            f4 = _mm256_load_ps(io + i + 32);
            f5 = _mm256_load_ps(io + i + 40);
            f6 = _mm256_load_ps(io + i + 48);
            f7 = _mm256_load_ps(io + i + 56);

            f0 = _mm256_mul_ps(f0, mul);
            f1 = _mm256_mul_ps(f1, mul);
            f2 = _mm256_mul_ps(f2, mul);
            f3 = _mm256_mul_ps(f3, mul);
            f4 = _mm256_mul_ps(f4, mul);
            f5 = _mm256_mul_ps(f5, mul);
            f6 = _mm256_mul_ps(f6, mul);
            f7 = _mm256_mul_ps(f7, mul);

            _mm256_store_ps(io + i + 0, f0);
            _mm256_store_ps(io + i + 8, f1);
            _mm256_store_ps(io + i + 16, f2);
            _mm256_store_ps(io + i + 24, f3);
            _mm256_store_ps(io + i + 32, f4);
            _mm256_store_ps(io + i + 40, f5);
            _mm256_store_ps(io + i + 48, f6);
            _mm256_store_ps(io + i + 56, f7);
        }
        return i;
    }

}
void vectorized::shortToFloat(const std::int16_t * __restrict__ in, float * __restrict__ out, const std::size_t length){
    std::size_t i{0};
    if(length > 64){
        const std::size_t stop{length - 64};
        for(i = 0; i < stop; i += 64)
        {
            //done like this so the compiler will unroll the inner loop for us
            for(int j = 0; j < 64; j += 16)
            {
                const __m256i in0 = _mm256_load_si256((const __m256i *)(in + i + j));
                const __m128i in1 = _mm256_extracti128_si256(in0, 0);
                const __m128i in2 = _mm256_extracti128_si256(in0, 1);
                const __m256i tmp1 = _mm256_cvtepi16_epi32(in1);
                const __m256i tmp2 = _mm256_cvtepi16_epi32(in2);
                const __m256  out1 = _mm256_cvtepi32_ps(tmp1);
                const __m256  out2 = _mm256_cvtepi32_ps(tmp2);
                _mm256_store_ps(out + i + j + 0, out1);
                _mm256_store_ps(out + i + j + 8, out2);
            }
        }
        _mm256_zeroupper();
    }
    for(; i < length; i++)
    {
        out[i] = (float)in[i];
    }

}

void vectorized::floatToShort(const float * __restrict__ in, std::int16_t * __restrict__ out, const std::size_t length){
    std::size_t i{0};
    if(length > 16){
        __m256  in0  ALIGN32_END;
        __m256  in1  ALIGN32_END;
        __m256i out0 ALIGN32_END;
        __m256i out1 ALIGN32_END;

        const std::size_t stop{length - 16};

        for(i = 0; i < stop; i += 16)
        {
            in0 = _mm256_load_ps(in + i + 0);              //read float 0-7
            in1 = _mm256_load_ps(in + i + 8);              //read float 8-15
            out0 = _mm256_cvttps_epi32(in0);               //convert to int
            out1 = _mm256_cvttps_epi32(in1);               //convert to int

            out0 = _mm256_packs_epi32(out0, out1);         //pack integers as saturated shorts (packed in this order: 0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15)
            out0 = _mm256_permute4x64_epi64(out0, 0XD8);   //shuffle to 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

            _mm256_store_si256((__m256i *)(out + i), out0); //store
        }
    }
    for(; i < length; i++)
    {
        out[i] = (short)in[i];
    }

}

void vectorized::reverse(const float * __restrict__ in, float * __restrict__ out,  std::size_t length) {
    std::size_t i;
    const std::size_t s2 = length - 32;
    for(i = 0; i < s2; i += 32)
    {
        __m256 v0 = _mm256_load_ps(in + i +   0);
        __m256 v1 = _mm256_load_ps(in + i +   8);
        __m256 v2 = _mm256_load_ps(in + i +  16);
        __m256 v3 = _mm256_load_ps(in + i +  24);

        v0 = _mm256_permute2f128_ps(v0, v0, 1);
        v1 = _mm256_permute2f128_ps(v1, v1, 1);
        v2 = _mm256_permute2f128_ps(v2, v2, 1);
        v3 = _mm256_permute2f128_ps(v3, v3, 1);
        v0 = _mm256_permute_ps(v0, 0x1B);
        v1 = _mm256_permute_ps(v1, 0x1B);
        v2 = _mm256_permute_ps(v2, 0x1B);
        v3 = _mm256_permute_ps(v3, 0x1B);

        _mm256_storeu_ps((out + length) - (i +  8), v0);
        _mm256_storeu_ps((out + length) - (i + 16), v1);
        _mm256_storeu_ps((out + length) - (i + 24), v2);
        _mm256_storeu_ps((out + length) - (i + 32), v3);
    }
    _mm256_zeroupper();
    for(; i < length; i++)
    {
        out[(length - i) - 1] = in[i];
    }
}

void vectorized::multiply(  float* __restrict__ io, const float num,  const std::size_t size){
    __m256 mul ALIGN32_END;

    mul = _mm256_broadcast_ss(&num);
    std::size_t i = doVectorizedMultiply(io, mul, size);
    _mm256_zeroupper();

    for (; i < size; i++){
        io[i] = io[i] * num;
    }
}

void vectorized::multiplyIQ(float* __restrict__ io,  float ival,  float qval,  std::size_t size){
    float mulArray[8] ALIGN32_END;
    mulArray[0] = ival;
    mulArray[1] = qval;
    mulArray[2] = ival;
    mulArray[3] = qval;
    mulArray[4] = ival;
    mulArray[5] = qval;
    mulArray[6] = ival;
    mulArray[7] = qval;

    std::size_t i = doVectorizedMultiply(io, *((__m256*)(&mulArray[0])), size);
    _mm256_zeroupper();
    for(; i < size; i+= 2){
        io[i]   *= ival;
        io[i+1] *= qval;
    }

}

void vectorized::add(std::int16_t* __restrict__ io, const std::int16_t* __restrict__ second,  std::size_t size){
    std::size_t i = 0;
    const std::size_t stop = size - 32;
    for (; i <= stop; i += 32)
    {
        __m256i ini0 = _mm256_load_si256((__m256i *)(io + i + 0));
        __m256i ini1 = _mm256_load_si256((__m256i *)(io + i + 16));
        __m256i ins0 = _mm256_load_si256((const __m256i *)(second + i + 0));
        __m256i ins1 = _mm256_load_si256((const __m256i *)(second + i + 16));

        ini0 = _mm256_adds_epi16(ini0, ins0);
        ini1 = _mm256_adds_epi16(ini1, ins1);

        _mm256_store_si256((__m256i *)(io + i + 0), ini0);
        _mm256_store_si256((__m256i *)(io + i + 16), ini1);
    }
    _mm256_zeroupper();
    for (; i < size; i++)
    {
        io[i] = (int16_t)std::min(io[i] + second[i], (int)std::numeric_limits<int16_t>::max());
    }

}

void vectorized::addUNA(std::int16_t* __restrict__ io, const std::int16_t* __restrict__ second,  std::size_t size){
    std::size_t i = 0;
    const std::size_t stop = size - 32;
    for (; i <= stop; i += 32)
    {
        __m256i ini0 = _mm256_loadu_si256((__m256i *)(io + i + 0));
        __m256i ini1 = _mm256_loadu_si256((__m256i *)(io + i + 16));
        __m256i ins0 = _mm256_loadu_si256((const __m256i *)(second + i + 0));
        __m256i ins1 = _mm256_loadu_si256((const __m256i *)(second + i + 16));

        ini0 = _mm256_adds_epi16(ini0, ins0);
        ini1 = _mm256_adds_epi16(ini1, ins1);

        _mm256_storeu_si256((__m256i *)(io + i + 0), ini0);
        _mm256_storeu_si256((__m256i *)(io + i + 16), ini1);
    }
    _mm256_zeroupper();
    for (; i < size; i++)
    {
        io[i] = (int16_t)std::min(io[i] + second[i], (int)std::numeric_limits<int16_t>::max());
    }
}

void vectorized::add(float* __restrict__ io, const float* __restrict__ second,  std::size_t size){
    std::size_t i = 0;
    const std::size_t stop = size - 32;
    for(; i <= stop; i += 32)
    {
        __m256 ini0 = _mm256_load_ps(io     + i +  0);
        __m256 ini1 = _mm256_load_ps(io     + i +  8);
        __m256 ini2 = _mm256_load_ps(io     + i + 16);
        __m256 ini3 = _mm256_load_ps(io     + i + 24);
        __m256 ins0 = _mm256_load_ps(second + i +  0);
        __m256 ins1 = _mm256_load_ps(second + i +  8);
        __m256 ins2 = _mm256_load_ps(second + i + 16);
        __m256 ins3 = _mm256_load_ps(second + i + 24);

        ini0 = _mm256_add_ps(ini0, ins0);
        ini1 = _mm256_add_ps(ini1, ins1);
        ini2 = _mm256_add_ps(ini2, ins2);
        ini3 = _mm256_add_ps(ini3, ins3);

        _mm256_store_ps(io + i +  0, ini0);
        _mm256_store_ps(io + i +  8, ini1);
        _mm256_store_ps(io + i + 16, ini2);
        _mm256_store_ps(io + i + 24, ini3);
    }
    _mm256_zeroupper();
    for(; i < size; i++)
    {
        io[i] = io[i] + second[i];
    }
}

void vectorized::addUNA(float* __restrict__ io, const float* __restrict__ second, std::size_t size){
    std::size_t i = 0;
    const std::size_t stop = size - 32;
    for(; i <= stop; i += 32)
    {
        __m256 ini0 = _mm256_loadu_ps(io     + i +  0);
        __m256 ini1 = _mm256_loadu_ps(io     + i +  8);
        __m256 ini2 = _mm256_loadu_ps(io     + i + 16);
        __m256 ini3 = _mm256_loadu_ps(io     + i + 24);
        __m256 ins0 = _mm256_loadu_ps(second + i +  0);
        __m256 ins1 = _mm256_loadu_ps(second + i +  8);
        __m256 ins2 = _mm256_loadu_ps(second + i + 16);
        __m256 ins3 = _mm256_loadu_ps(second + i + 24);

        ini0 = _mm256_add_ps(ini0, ins0);
        ini1 = _mm256_add_ps(ini1, ins1);
        ini2 = _mm256_add_ps(ini2, ins2);
        ini3 = _mm256_add_ps(ini3, ins3);

        _mm256_storeu_ps(io + i +  0, ini0);
        _mm256_storeu_ps(io + i +  8, ini1);
        _mm256_storeu_ps(io + i + 16, ini2);
        _mm256_storeu_ps(io + i + 24, ini3);
    }
    _mm256_zeroupper();
    for(; i < size; i++)
    {
        io[i] = io[i] + second[i];
    }
}

void vectorized::iqToAmpAndPhase(const float * __restrict__ in, float * __restrict__ out, std::size_t size) {
    static constexpr float piFloat{static_cast<float>(M_PI)};
    __m256 val0 ALIGN32_END;
    __m256 val1 ALIGN32_END;
    __m256 tmp0 ALIGN32_END;
    __m256 tmp1 ALIGN32_END;
    __m256 tmp2 ALIGN32_END;
    __m256 tmp3 ALIGN32_END;

    __m256 tp ALIGN32_END;
    tp = _mm256_broadcast_ss(&piFloat);

    const std::size_t stop = size - 16;
    std::size_t i = 0;
    for(; i < stop; i+=16){
        val0 = _mm256_load_ps(in + i);
        val1 = _mm256_load_ps(in + i + 8);

        tmp0 = _mm256_mul_ps(val0, val0);
        tmp1 = _mm256_mul_ps(val1, val1);

        tmp2 = _mm256_hadd_ps(tmp0, tmp1);
        //TODO -> find a way to make rsqrt work!!!
        tmp2 = _mm256_sqrt_ps(tmp2);          // 0,1,4,5,2,3,6,7

        avx_mathfun::atan2(val0, val1, tmp3); // 0,1,4,5,2,3,6,7

        // convert from -pi,+pi to 0,2pi
        tmp3 = _mm256_add_ps(tmp3, tp);

        tmp0 = _mm256_unpackhi_ps(tmp2, tmp3);
        tmp1 = _mm256_unpacklo_ps(tmp2, tmp3);

        _mm256_store_ps(out + i + 0, tmp1);
        _mm256_store_ps(out + i + 8, tmp0);
    }
    _mm256_zeroupper();
    for(; i < size; i+=2){
        out[i+0] = std::hypot(in[i], in[i+1]);
        out[i+1] = std::atan2(in[i], in[i+1]) + piFloat;
    }
}

const __m256& vectorized::zeroVec() {
    static __m256 ret ALIGN32_END;
    static bool set{false};
    if (!set) {
        std::mutex lock;
        std::lock_guard<std::mutex> lk(lock);
        if (!set) {
            constexpr float zero{0.0F};
            set = true;
            ret = _mm256_broadcast_ss(&zero);
        }
    }
    return ret;
}