//MIT License
//
//Copyright (c) 2018 TU Braunschweig, Algorithms Group
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included in all
//copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.

#include "strategies.cuh"
#include "../subintervals.hpp"
#include "../output_case.hpp"
#include "../interrupt.hpp"
#include <algcuda/interval.cuh>
#include <algcuda/memory.hpp>
#include <algcuda/device.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <limits>
#include <algorithm>

constexpr int la_subintervals_init = 256;
constexpr int r1_subintervals_init = 128;
constexpr int r2_subintervals_init = 128;
constexpr int r3_subintervals_init = 64;
constexpr int r4_subintervals_init = 64;
constexpr int r5_subintervals_init = 64;
constexpr int r6_subintervals_init = 64;

constexpr int r3_parallel_init = 64;
constexpr int r4_parallel_init = 64;
constexpr int r5_parallel_init = 16;
constexpr int r6_parallel_init = 16;

constexpr int la_subintervals_subdiv = 64;
constexpr int r1_subintervals_subdiv = 32;
constexpr int r2_subintervals_subdiv = 32;
constexpr int r3_subintervals_subdiv = 16;
constexpr int r4_subintervals_subdiv = 16;
constexpr int r5_subintervals_subdiv = 16;
constexpr int r6_subintervals_subdiv = 32;

constexpr int r3_parallel_subdiv = 16;
constexpr int r4_parallel_subdiv = 16;
constexpr int r5_parallel_subdiv = 16;
constexpr int r6_parallel_subdiv = 16;

constexpr int num_generations = 1;

constexpr std::size_t default_output_buffer_size = 1 << 20;

namespace circlecover {
	namespace rectangle_size_bound {
		void __host__ find_critical_intervals();

		struct Interval_buffer {
			__host__ Interval_buffer() :
				buffer(algcuda::device::make_unique<Variables[]>(default_output_buffer_size)),
				buffer_size_current(algcuda::device::make_unique<std::size_t>()),
				buffer_size_total(default_output_buffer_size)
			{}

			void __host__ enlarge(int new_size) {
				buffer = algcuda::device::make_unique<Variables[]>(new_size);
				buffer_size_total = new_size;
			}

			algcuda::device::Memory<Variables[]> buffer;
			algcuda::device::Memory<std::size_t> buffer_size_current;
			std::size_t buffer_size_total;
		};

		static void __host__ find_critical_intervals_in(const Variables& intervals, Interval_buffer& buffers, std::vector<Variables>& append_to, bool interruptible, bool is_initialization, unsigned start_from_lambda = 0);
		static void __host__ find_critical_intervals_in_call_kernel(const Variables& intervals, Interval_buffer& buffers, std::vector<Variables>& append_to, bool is_initialization);
	}
}

using namespace circlecover;
using namespace circlecover::rectangle_size_bound;

static inline void __device__ push_interval_set_to_buffer(const Variables& vars, Variables* buffer, std::size_t buffer_size, std::size_t* current_size) {
	static_assert(sizeof(std::size_t) == sizeof(unsigned long long), "Size of unsigned long long is wrong!");
	std::size_t index = static_cast<std::size_t>(atomicAdd(reinterpret_cast<unsigned long long*>(current_size), 1));
	if(index < buffer_size) {
		buffer[index] = vars;
	}
}

template<int NumRadiiMerged> static __host__ std::vector<Variables> merge_critical_intervals(std::vector<Variables>& criticals) {
	static_assert(NumRadiiMerged <= 6, "Only 6 radii can be merged by");

	auto eq_comp = [] (const IV& i1, const IV& i2) noexcept -> bool {
		return i1.get_lb() == i2.get_lb() && i1.get_ub() == i2.get_ub();
	};

	if(criticals.empty()) {
		return {};
	}

	std::sort(criticals.begin(), criticals.end());
	std::vector<Variables> result;
	result.push_back(criticals.front());

	for(const Variables& v : criticals) {
		Variables& prev = result.back();
		if(v.la.get_lb() == prev.la.get_lb() && v.la.get_ub() == prev.la.get_ub() && (NumRadiiMerged <= 0 || std::equal(+prev.radii, prev.radii+NumRadiiMerged, +v.radii, eq_comp))) {
			for(int i = NumRadiiMerged; i < 6; ++i) {
				prev.radii[i].do_join(v.radii[i]);
			}
		} else {
			result.push_back(v);
		}
	}

	return result;
}

static inline IV __device__ r5_refine_range(const Variables& intervals, IV r3, IV r4) {
	IV r = intervals.radii[4];
	r.tighten_ub(r4.get_ub());
	
	double ub_remaining = __dmul_ru(intervals.la.get_ub(), critical_ratio);
	ub_remaining = __dadd_ru(ub_remaining, -intervals.radii[0].get_lb());
	ub_remaining = __dadd_ru(ub_remaining, -intervals.radii[1].get_lb());
	ub_remaining = __dadd_ru(ub_remaining, -r3.get_lb());
	ub_remaining = __dadd_ru(ub_remaining, -r4.get_lb());
	r.tighten_ub(ub_remaining);

	return r;
}

static inline IV __device__ r6_refine_range(const Variables& intervals, IV r3, IV r4, IV r5) {
	IV r = intervals.radii[5];
	r.tighten_ub(r5.get_ub());

	double ub_remaining = __dmul_ru(intervals.la.get_ub(), critical_ratio);
	ub_remaining = __dadd_ru(ub_remaining, -intervals.radii[0].get_lb());
	ub_remaining = __dadd_ru(ub_remaining, -intervals.radii[1].get_lb());
	ub_remaining = __dadd_ru(ub_remaining, -r3.get_lb());
	ub_remaining = __dadd_ru(ub_remaining, -r4.get_lb());
	ub_remaining = __dadd_ru(ub_remaining, -r5.get_lb());
	r.tighten_ub(ub_remaining);

	return r;
}

static __global__ void kernel_find_critical_intervals_in(Variables intervals, Variables* output, std::size_t *current_size, std::size_t buffer_size, bool is_initialization) {
	if(shortcut_even_split_recursion(intervals.la, intervals.radii[0], intervals.radii[1]) || shortcut_uneven_split_recursion(intervals.la, intervals.radii[0], intervals.radii[1])) {
		return;
	}

	const int r3_subintervals = is_initialization ? r3_subintervals_init : r3_subintervals_subdiv;
	const int r4_subintervals = is_initialization ? r4_subintervals_init : r4_subintervals_subdiv;
	const int r5_subintervals = is_initialization ? r5_subintervals_init : r5_subintervals_subdiv;
	const int r6_subintervals = is_initialization ? r6_subintervals_init : r6_subintervals_subdiv;

	for(int r3_offset = blockIdx.x; r3_offset < r3_subintervals; r3_offset += gridDim.x) {
		IV r3 = get_subinterval(intervals.radii[2], r3_offset, r3_subintervals);
		if(
			shortcut_uneven_split_recursion(intervals.la, intervals.radii[0], intervals.radii[1], r3) ||
			shortcut_r1_r2_opposite_corners_recursion(intervals.la, intervals.radii[0], intervals.radii[1], r3) ||
			shortcut_r1_r2_vertical_strip(intervals.la, intervals.radii[0], intervals.radii[1], r3)
		) {
			continue;
		}

		for(int r4_offset = blockIdx.y; r4_offset < r4_subintervals; r4_offset += gridDim.y) {
			IV r4 = get_subinterval(intervals.radii[3], r4_offset, r4_subintervals);
			for(int r5_offset = threadIdx.x; r5_offset < r5_subintervals; r5_offset += blockDim.x) {
				IV r5_range = r5_refine_range(intervals, r3, r4);
				IV r5 = get_subinterval(intervals.radii[4], r5_offset, r5_subintervals);
				for(int r6_offset = threadIdx.y; r6_offset < r6_subintervals; r6_offset += blockDim.y) {
					IV r6_range = r6_refine_range(intervals, r3, r4, r5);
					IV r6 = get_subinterval(intervals.radii[5], r6_offset, r6_subintervals);

					Variables vars{
						intervals.la, { intervals.radii[0], intervals.radii[1], r3, r4, r5, r6 }
					};
					vars = vars.tighten();

					Intermediate_values vals{vars};
					if(vars.empty() || vals.R.empty()) {
						continue;
					}

					// the simpler/faster strategies go here
					if(
						even_split_recursion(vars, vals) ||
						uneven_split_recursion(vars, vals) ||
						multi_disk_strip_vertical(vars, vals) ||
						vertical_wall_building_recursion(vars, vals) ||
						r1_in_corner_explicit_recursion(vars, vals) ||
						r1_r2_opposite_corners_recursion(vars, vals) ||
						r1_r2_vertical_strip(vars, vals)
					) {
						continue;
					}

					// compute a 2x2 cover
					vals.cover_2x2 = compute_two_by_two_cover(+vars.radii);
					if(vals.cover_2x2.width > 0) {
						if(vals.cover_2x2.width > vars.la.get_ub() || two_by_two_cover_with_strip_and_recursion(vars, vals)) {
							continue;
						}
					}

					// the more expensive strategies
					if(
						!l_shaped_recursion(vars, vals) &&
						!r1_in_corner_wall_building_recursion(vars, vals) &&
						!advanced_multi_disk_strip_vertical(vars, vals) &&
						!five_disk_full_cover(vars, vals) &&
						!six_disk_full_cover(vars, vals) &&
						!seven_disk_strategies(vars, vals)
					) {
						push_interval_set_to_buffer(vars, output, buffer_size, current_size);
					}
				}
			}
		}
	}
}

void __host__ circlecover::rectangle_size_bound::find_critical_intervals_in_call_kernel(const Variables& intervals, Interval_buffer& buffers, std::vector<Variables>& append_to, bool is_initialization) {
	const int r3_parallel = is_initialization ? r3_parallel_init : r3_parallel_subdiv;
	const int r4_parallel = is_initialization ? r4_parallel_init : r4_parallel_subdiv;
	const int r5_parallel = is_initialization ? r5_parallel_init : r5_parallel_subdiv;
	const int r6_parallel = is_initialization ? r6_parallel_init : r6_parallel_subdiv;

	const dim3 grid_dims(r3_parallel, r4_parallel);
	const dim3 block_dims(r5_parallel, r6_parallel);

	std::size_t dummy_0 = 0;
	std::size_t buffer_size_out = 0;

	algcuda::device::copy(&dummy_0, buffers.buffer_size_current);
	kernel_find_critical_intervals_in<<<grid_dims,block_dims>>>(intervals, buffers.buffer.get(), buffers.buffer_size_current.get(), buffers.buffer_size_total, is_initialization);
	algcuda::device::synchronize_check_errors();
	algcuda::device::copy(buffers.buffer_size_current, &buffer_size_out);

	if(buffer_size_out > 0) {
		if(buffer_size_out > buffers.buffer_size_total) {
			// the buffer was insufficient, we have to run again
			buffers.enlarge(buffer_size_out);

			algcuda::device::copy(&dummy_0, buffers.buffer_size_current);
			kernel_find_critical_intervals_in<<<grid_dims,block_dims>>>(intervals, buffers.buffer.get(), buffers.buffer_size_current.get(), buffers.buffer_size_total, is_initialization);
			algcuda::device::synchronize_check_errors();
			algcuda::device::copy(buffers.buffer_size_current, &buffer_size_out);

			if(buffer_size_out > buffers.buffer_size_total) {
				throw std::logic_error("Required buffer size grew unexpectedly!");
			}
		}

		std::size_t old_size = append_to.size();
		append_to.resize(old_size + buffer_size_out);
		algcuda::device::copy(buffers.buffer, buffer_size_out, append_to.data() + old_size);
	}
}

void __host__ circlecover::rectangle_size_bound::find_critical_intervals_in(const Variables& intervals, Interval_buffer& buffers, std::vector<Variables>& append_to, bool interruptible, bool is_initialization, unsigned start_from_lambda_offset) {
	const int la_subintervals = is_initialization ? la_subintervals_init : la_subintervals_subdiv;
	const int r1_subintervals = is_initialization ? r1_subintervals_init : r1_subintervals_subdiv;
	const int r2_subintervals = is_initialization ? r2_subintervals_init : r2_subintervals_subdiv;

	for(int la_offset = static_cast<int>(start_from_lambda_offset); la_offset < la_subintervals; ++la_offset) {
		IV la = get_subinterval(intervals.la, la_offset, la_subintervals);
		for(int r1_offset = 0; r1_offset < r1_subintervals; ++r1_offset) {
			IV r1 = get_subinterval(intervals.radii[0], r1_offset, r1_subintervals);
			for(int r2_offset = 0; r2_offset < r2_subintervals; ++r2_offset) {
				IV r2_range = intervals.radii[1];
				r2_range.tighten_ub(r1.get_ub());
				IV r2 = get_subinterval(r2_range, r2_offset, r2_subintervals);
				r1.tighten_lb(r2.get_lb());

				Variables subvar = Variables::from_la_r1_r2(intervals, la, r1, r2);
				if(subvar.empty()) {
					continue;
				}

				find_critical_intervals_in_call_kernel(subvar, buffers, append_to, is_initialization);
				if(r2_offset % 16 == 0) {
					std::cerr << "λ: " << std::setw(3) << la_offset << " of "
					          << std::setw(3) << la_subintervals << ", r_1: "
						  << std::setw(3) << r1_offset << " of " << r1_subintervals
						  << ", r_2: " << std::setw(3) << (r2_offset+1) << " of " << r2_subintervals << " done (" << append_to.size() << " critical intervals found)                \r" << std::flush;
				}

				if(interruptible && was_interrupted) {
					return;
				}
			}
		}
	}
}

void __host__ circlecover::rectangle_size_bound::find_critical_intervals() {
	output_begin_case("Critical intervals for rectangles with small disks");
	std::cerr << "Size bound: r <= " << ub_disk_radius << ", critical ratio: " << critical_ratio << std::endl;

	Interval_buffer buffer;
	Variables initial{
		IV{lb_proof_lambda, ub_proof_lambda},
		{ {0.0, ub_disk_weight}, {0.0, ub_disk_weight}, {0.0, ub_disk_weight}, {0.0, ub_disk_weight}, {0.0, ub_disk_weight}, {0.0, ub_disk_weight} }
	};

	std::vector<Variables> current_gen, next_gen;

	std::cerr << "Initializing..." << std::endl;
	find_critical_intervals_in(initial, buffer, next_gen, false, true);
	current_gen = merge_critical_intervals<0>(next_gen);
	next_gen.clear();

	if(current_gen.empty()) {
		std::cerr << "No critical regions after initialization - done!" << std::endl;
		output_end_case();
		return;
	}

	std::cerr << "Starting refinement - press Ctrl-C to interrupt and use best result achieved so far...                                               " << std::endl;
	start_interruptible_computation();

	for(int g = 1; g <= num_generations; ++g) {
		std::cerr << "Running generation " << g << " on " << current_gen.size() << " intervals...                                                                                                     " << std::endl;
		for(const Variables& s : current_gen) {
			find_critical_intervals_in(s, buffer, next_gen, true, false);
			if(was_interrupted) {
				break;
			}
		}

		if(was_interrupted) {
			break;
		}

		switch(g % 3) {
			case 0:
				current_gen = merge_critical_intervals<3>(next_gen);
			break;

			case 1:
				current_gen = merge_critical_intervals<2>(next_gen);
			break;

			case 2:
				current_gen = merge_critical_intervals<1>(next_gen);
			break;
		}

		next_gen.clear();
		if(current_gen.empty()) {
			break;
		}
	}

	stop_interruptible_computation();

	if(!current_gen.empty()) {
		//Interval_set result = merge_critical_intervals_all(current_gen).front();
		//std::cerr << "Critical intervals: la: " << result.la << ", r_1^2: " << result.r1 << ", r_2^2: " << result.r2 << ", r_3^2: " << result.r3 << std::endl;
		for(const auto& c : current_gen) {
			std::cout << "Critical: " << c << std::endl;
		}
	}

	output_end_case();
}

