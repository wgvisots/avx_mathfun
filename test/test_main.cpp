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

#include <cstdint>
#include <iostream>

#include "Aligned.h"

int main(){
    std::size_t* thing = aligned::allocateAligned<std::size_t>(aligned::Alignment::AVX, 10);
    std::cout << (0 == ((std::uint64_t)thing % 32)) << std::endl;

    using allocator = aligned::AlignedAllocator<std::size_t, aligned::Alignment::AVX>;
    std::vector<std::size_t, allocator> vec;
    vec.resize(1);

    std::size_t* vptr = vec.data();
    std::cout << (0 == ((std::uint64_t)vptr % 32)) << std::endl;
    return 0;
}