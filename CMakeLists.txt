##############################################################################
# Copyright (C) 2025 W. Greg Visotski
#
# This software is provided 'as-is', without any express or implied
# warranty.  In no event will the authors be held liable for any damages
# arising from the use of this software.
#
# Permission is granted to anyone to use this software for any purpose,
# including commercial applications, and to alter it and redistribute it
# freely, subject to the following restrictions:
#
# 1. The origin of this software must not be misrepresented; you must not
# claim that you wrote the original software. If you use this software
# in a product, an acknowledgment in the product documentation would be
# appreciated but is not required.
# 2. Altered source versions must be plainly marked as such, and must not be
# misrepresented as being the original software.
# 3. This notice may not be removed or altered from any source distribution.
#
#(this is the zlib license)
##############################################################################

cmake_minimum_required(VERSION 3.20)

set(PROJ mathfun)
project(${PROJ})

set(CMAKE_CXX_STANDARD 20)

set(O_FLAG "-O0")
set(MATH_FLAG "-mavx2")

set(CMAKE_CXX_FLAGS "-FPIC -fopenmp ${MATH_FLAG} -Wall -Wextra -Wpedantic -g -pg ${O_FLAG}")

set(INC_DIRS "./src" )
file(GLOB LIB_SRCS "./src/*.cpp")

set(LIBS "pthread" )

add_library(${PROJ} ${LIB_SRCS})
target_include_directories(${PROJ} PUBLIC ${INC_DIRS})
target_link_libraries(${PROJ} ${LIBS})

enable_testing()
add_subdirectory(test)
#
#set(APPS "test_main")
#
#foreach (APP ${APPS})
#    add_executable(${APP} "test/${APP}.cpp")
#    target_include_directories(${APP} PUBLIC ${INC_DIRS})
#    target_link_libraries(${APP} ${LIBS} ${PROJ})
#endforeach ()
