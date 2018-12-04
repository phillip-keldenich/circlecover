#MIT License
#
#Copyright (c) 2018 TU Braunschweig, Algorithms Group
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

cmake_minimum_required(VERSION 3.5)
project(circlecover_prover CXX)

set(UTIL_BOOST_LIBRARIES program_options)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/util.cmake" NO_POLICY_SCOPE)
include("${CMAKE_CURRENT_SOURCE_DIR}/algcuda/UseCUDA.cmake")

add_executable(circlecover_prover prover.cpp interrupt.cpp)
util_setup_target(circlecover_prover)

set(CIRCLECOVER_COMMON_CUDA_SOURCES operations.cu)
set(CIRCLECOVER_SIZE_BOUND_SOURCES
	rectangle_size_bound/compute_efficient_rectangle_cover_weight_range.cu
	rectangle_size_bound/compute_even_split_distance_bound.cu
	rectangle_size_bound/disk_satisfies_size_bound.cu
	rectangle_size_bound/even_split_recursion.cu
	rectangle_size_bound/uneven_split_recursion.cu
	rectangle_size_bound/multi_disk_strip.cu
	rectangle_size_bound/vertical_wall_building_recursion.cu
	rectangle_size_bound/maximize_covered.cu
	rectangle_size_bound/seven_disk_strategies.cu
	rectangle_size_bound/six_disks_cover_width.cu
	rectangle_size_bound/r1_in_corner_explicit_recursion.cu
	rectangle_size_bound/r1_in_corner_wall_building_recursion.cu
	rectangle_size_bound/two_by_two_cover.cu
	rectangle_size_bound/two_by_two_cover_with_strip_and_recursion.cu
	rectangle_size_bound/l_shaped_recursion.cu
	rectangle_size_bound/five_disk_full_cover.cu
	rectangle_size_bound/six_disk_full_cover.cu
	rectangle_size_bound/r1_r2_opposite_corners_recursion.cu
	rectangle_size_bound/r1_r2_vertical_strip.cu
	rectangle_size_bound/can_recurse.cu
)

set(CIRCLECOVER_CUDA_SOURCES
	${CIRCLECOVER_COMMON_CUDA_SOURCES}

	${CIRCLECOVER_SIZE_BOUND_SOURCES}
	rectangle_size_bound/rectangle_size_bound.cu

	tight_rectangle/tight_rectangle.cu
	tight_rectangle/can_apply_size_bounded_covering.cu
	tight_rectangle/can_recurse.cu
	tight_rectangle/critical_ratio.cu
	tight_rectangle/r1_strategies.cu
	tight_rectangle/r1_r2_strategies.cu
	tight_rectangle/r1_r2_r3_strategies.cu
	tight_rectangle/two_by_two_strategies.cu
	tight_rectangle/r1_r2_large_r3_r4_gaps_strategy.cu
)

util_add_cuda_sources(circlecover_prover "${CIRCLECOVER_CUDA_SOURCES}")

