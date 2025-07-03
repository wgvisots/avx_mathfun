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

#include <cstdlib>
#include <memory>

namespace aligned{

    enum class Alignment : std::size_t
    {
        Normal = sizeof(void*),
        SSE    = 16,
        AVX    = 32,
        AVX512 = 64
    };

    /**
     * Just a stupid simple allocator to get aligned memory for a std::vector
     *
     * Future -> maybe do something cool like described here:
     * https://indiegamedev.net/2022/03/27/custom-c20-memory-allocators-for-stl-containers/
     *
     * @tparam T
     * @tparam ALIGNMENT_TYPE
     */
    template<typename T, aligned::Alignment ALIGNMENT_TYPE = aligned::Alignment::AVX>
    class AlignedAllocator{
    public:
        static_assert((std::size_t)ALIGNMENT_TYPE >= alignof(T), "You may be violating minimum alignment requirements.");
        using value_type = T;
        static std::align_val_t constexpr ALIGNMENT{(std::align_val_t)ALIGNMENT_TYPE};

        /**
         * @see https://stackoverflow.com/a/48062758/2191065
         */
        template<class OTHER>
        struct rebind{
            using other = AlignedAllocator<OTHER, ALIGNMENT_TYPE>;
        };

        constexpr AlignedAllocator() noexcept = default;
        constexpr AlignedAllocator( const AlignedAllocator& ) noexcept = default;

        template<typename U>
        constexpr explicit AlignedAllocator( AlignedAllocator<U, ALIGNMENT_TYPE> const& ) noexcept{}

        [[nodiscard]] constexpr T* allocate(std::size_t n){
            if ( n > (std::numeric_limits<std::size_t>::max() / sizeof(T))){
                throw std::bad_array_new_length();
            }
            return static_cast<T*>(::operator new[](n * sizeof(T), ALIGNMENT));
        }

        void deallocate(T* allocatedPointer, [[maybe_unused]] std::size_t n)
        {
            ::operator delete[]( allocatedPointer, ALIGNMENT );
        }
    };
}