###########################################################################
# CMakeLists.txt
#
# Copyright 2024, 2025 W. Greg Visotski
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###########################################################################

cmake_minimum_required(VERSION 3.20)

set(PROJ avx_mathfun_tests)
project(${PROJ})

find_package(GTest REQUIRED)

set(INC_DIRS "./"
             ${gtest_INCLUDE_DIR})

set(LIBS mathfun)

################################
# GTest
################################
enable_testing()


################################
# Unit Tests (AVX/AVX2)
################################
add_executable( runUnitTests testAVX_Mathfun.cpp )

target_include_directories(runUnitTests PUBLIC ${INC_DIRS})
target_link_libraries(runUnitTests ${LIBS})

# Link test executable against gtest & gtest_main
target_link_libraries(runUnitTests gtest gtest_main)
add_test(NAME runUnitTests
         COMMAND runUnitTests
         WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})


################################
# Unit Tests (AVX-512)
################################
# This executable is always built with AVX-512F enabled, regardless of the
# host CPU, so it is kept separate from runUnitTests.  It is run:
#   * natively if the configuring machine supports AVX-512F,
#   * otherwise under Intel SDE if SDE_EXECUTABLE is set / sde64 is on PATH
#     (or SDE_PATH points at the SDE install directory),
#   * otherwise the test is registered but disabled.
add_executable( runUnitTests512 testAVX512_Mathfun.cpp )
target_compile_options(runUnitTests512 PRIVATE -mavx512f -mfma)
target_include_directories(runUnitTests512 PUBLIC ${INC_DIRS})
target_link_libraries(runUnitTests512 ${LIBS} gtest gtest_main)

include(CheckCXXSourceRuns)
check_cxx_source_runs("
int main() {
    __builtin_cpu_init();
    return __builtin_cpu_supports(\"avx512f\") ? 0 : 1;
}" HOST_HAS_AVX512F)

find_program(SDE_EXECUTABLE NAMES sde64 sde HINTS ENV SDE_PATH)
set(SDE_CPU "-spr" CACHE STRING "CPU for Intel SDE to emulate when running AVX-512 tests (e.g. -spr, -icl, -skx, -future)")

if(HOST_HAS_AVX512F)
    message(STATUS "AVX-512 tests: running natively")
    add_test(NAME runUnitTests512
             COMMAND runUnitTests512
             WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
elseif(SDE_EXECUTABLE)
    message(STATUS "AVX-512 tests: running under Intel SDE (${SDE_EXECUTABLE} ${SDE_CPU})")
    add_test(NAME runUnitTests512
             COMMAND ${SDE_EXECUTABLE} ${SDE_CPU} -- $<TARGET_FILE:runUnitTests512>
             WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
else()
    message(STATUS "AVX-512 tests: host lacks AVX-512F and Intel SDE was not found - test disabled. "
                   "Set SDE_EXECUTABLE or SDE_PATH to run them under emulation.")
    add_test(NAME runUnitTests512 COMMAND runUnitTests512)
    set_tests_properties(runUnitTests512 PROPERTIES DISABLED TRUE)
endif()
