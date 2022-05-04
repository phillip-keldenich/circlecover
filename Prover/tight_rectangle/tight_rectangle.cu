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
#include "../rectangle_size_bound/values.hpp"
#include <algcuda/memory.hpp>
#include <algcuda/device.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

namespace circlecover {
namespace tight_rectangle {
void find_critical_intervals();
}
}

using namespace circlecover;
using namespace circlecover::tight_rectangle;

static const int la_subintervals_init = 256;
static const int r1_subintervals_init = 128;
static const int r2_subintervals_init = 128;
static const int r3_subintervals_init = 128;
static const int r4_subintervals_init = 128;

static const int num_generations = 11;
static const int la_subintervals_subdiv[num_generations] = {
	1, 2, 2, 2, 2, 1, 2, 1, 2, 1, 2
};

static const int r1_subintervals_subdiv = 128;
static const int r2_subintervals_subdiv = 128;
static const int r3_subintervals_subdiv = 128;
static const int r4_subintervals_subdiv = 64;

static const int r1_parallel_init = 128;
static const int r2_parallel_init = 128;
static const int r3_parallel_init = 16;
static const int r4_parallel_init = 16;

static const int r1_parallel_subdiv = 64;
static const int r2_parallel_subdiv = 64;
static const int r3_parallel_subdiv = 16;
static const int r4_parallel_subdiv = 16;

static const std::size_t default_output_buffer_size = 1 << 23;

static __device__ IV r1_range(IV la) {
	double lb_increased_height_factor = __ddiv_rd(critical_ratio(la).get_lb(), rectangle_size_bound::critical_ratio);
	if(lb_increased_height_factor <= 1.0) {
		lb_increased_height_factor = 1.0;
	}

	double lb = __dmul_rd(rectangle_size_bound::ub_disk_weight, lb_increased_height_factor);
	double ub = __dmul_ru(0.25, __dadd_ru(1.0, __dmul_ru(la.get_ub(),la.get_ub())));
	return {lb,ub};
}

static __device__ IV r2_range(IV la, IV r1) {
	double lb = 0.0;
	double ub1 = r1.get_ub();
	double ub2 = (required_weight_for(la) - r1).get_ub();
	return {lb,ub1 < ub2 ? ub1 : ub2};
}

static __device__ IV r3_range(IV la, IV r1, IV r2) {
	double lb = 0.0;
	double ub1 = r2.get_ub();
	double ub2 = (required_weight_for(la) - r1 - r2).get_ub();
	return {lb, ub1 < ub2 ? ub1 : ub2};
}

static __device__ IV r4_range(IV la, IV r1, IV r2, IV r3) {
	double lb = 0.0;
	double ub1 = r3.get_ub();
	double ub2 = (required_weight_for(la) - r1 - r2 - r3).get_ub();
	return {lb, ub1 < ub2 ? ub1 : ub2};
}

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

static __host__ void sort_by_r1(std::vector<Variables>& vars) { 
	auto var_comp = [] (const Variables& v1, const Variables& v2) -> bool {
		return v1.radii[0].get_lb() < v2.radii[0].get_lb() ||
			(v1.radii[0].get_lb() == v2.radii[0].get_lb() && v1.radii[0].get_ub() < v2.radii[0].get_ub());
	};
	std::sort(vars.begin(), vars.end(), var_comp);
}

static __host__ std::vector<Variables> merge_adjacent_r1(std::vector<Variables>& vars) {
	sort_by_r1(vars);
	std::vector<Variables> result;
	result.push_back(vars.front());

	for(const Variables& v : vars) {
		Variables& prev = result.back();
		if(prev.radii[0].get_lb() <= v.radii[0].get_lb()) {
			if(prev.radii[0].get_ub() >= v.radii[0].get_lb()) {
				prev.radii[0].do_join(v.radii[0]);
				prev.radii[1].do_join(v.radii[1]);
				prev.radii[2].do_join(v.radii[2]);
				prev.radii[3].do_join(v.radii[3]);
				continue;
			}
		} else {
			if(v.radii[0].get_ub() >= prev.radii[0].get_lb()) {
				prev.radii[0].do_join(v.radii[0]);
				prev.radii[1].do_join(v.radii[1]);
				prev.radii[2].do_join(v.radii[2]);
				prev.radii[3].do_join(v.radii[3]);
				continue;
			}
		}

		// if we did not merge, the intervals are disjoint
		result.push_back(v);
	}

	return result;
}

static inline void __device__ push_interval_set_to_buffer(const Variables& vars, Variables* buffer, std::size_t buffer_size, std::size_t* current_size) {
	static_assert(sizeof(std::size_t) == sizeof(unsigned long long), "Size of unsigned long long is wrong!");
	std::size_t index = static_cast<std::size_t>(atomicAdd(reinterpret_cast<unsigned long long*>(current_size), 1));
	if(index < buffer_size) {
		buffer[index] = vars;
	}
}

static __global__ void kernel_find_critical_intervals_in(Variables intervals, Variables* output, std::size_t *current_size, std::size_t buffer_size, bool is_initialization) {
	const int r1_subintervals = is_initialization ? r1_subintervals_init : r1_subintervals_subdiv;
	const int r2_subintervals = is_initialization ? r2_subintervals_init : r2_subintervals_subdiv;
	const int r3_subintervals = is_initialization ? r3_subintervals_init : r3_subintervals_subdiv;
	const int r4_subintervals = is_initialization ? r4_subintervals_init : r4_subintervals_subdiv;

	IV weight = required_weight_for(intervals.la);

	IV r1r = r1_range(intervals.la).intersect(intervals.radii[0]);
	if(r1r.empty()) {
		return;
	}

	for(int r1_offset = blockIdx.x; r1_offset < r1_subintervals; r1_offset += gridDim.x) {
		IV r1 = get_subinterval(r1r, r1_offset, r1_subintervals);
		IV r2r = r2_range(intervals.la, r1).intersect(intervals.radii[1]);
		if(r2r.empty()) {
			continue;
		}

		if(r1_strategies(intervals.la, r1, r2r.get_ub())) {
			continue;
		}

		for(int r2_offset = blockIdx.y; r2_offset < r2_subintervals; r2_offset += gridDim.y) {
			IV r2 = get_subinterval(r2r, r2_offset, r2_subintervals);
			if(r1_strategies(intervals.la, r1, r2.get_ub())) {
				continue;
			}

			IV r3r = r3_range(intervals.la, r1, r2).intersect(intervals.radii[2]);
			if(r3r.empty()) {
				continue;
			}

			if(r1_r2_strategies(intervals.la, r1, r2, r3r.get_ub())) {
				continue;
			}

			for(int r3_offset = threadIdx.x; r3_offset < r3_subintervals; r3_offset += blockDim.x) {
				IV r3 = get_subinterval(r3r, r3_offset, r3_subintervals);
				if(r1_r2_strategies(intervals.la, r1, r2, r3.get_ub())) {
					continue;
				}

				IV r4r = r4_range(intervals.la, r1, r2, r3).intersect(intervals.radii[3]);
				if(r4r.empty()) {
					continue;
				}

				if(r1_r2_r3_strategies(intervals.la, r1, r2, r3, r4r.get_ub())) {
					continue;
				}

				IV R4 = weight - r1 - r2 - r3;
				R4.tighten_lb(0.0);
				if(R4.empty()) {
					continue;
				}

				for(int r4_offset = threadIdx.y; r4_offset < r4_subintervals; r4_offset += blockDim.y) {
					IV r4 = get_subinterval(r4r, r4_offset, r4_subintervals);
					if(r1_r2_r3_strategies(intervals.la, r1, r2, r3, r4.get_ub())) {
						continue;
					}
					
					Variables vars{intervals.la, {r1,r2,r3,r4}};
					vars.radii[2].tighten_lb(vars.radii[3].get_lb());
					vars.radii[1].tighten_lb(vars.radii[2].get_lb());
					vars.radii[0].tighten_lb(vars.radii[1].get_lb());

					IV R = weight - vars.radii[0] - vars.radii[1] - vars.radii[2] - vars.radii[3];
					R.tighten_lb(0.0);
					if(R.empty()) {
						continue;
					}

					if(!two_by_two_strategies(vars, R) && 
					   !r1_r2_r3_strategies(vars, R4) &&
					   !r1_r2_large_r3_r4_gaps_strategy(vars, R))
					{
//						printf("Critical interval: la = [%.17g,%.17g], r1 = [%.17g,%.17g], r2 = [%.17g,%.17g], r3 = [%.17g,%.17g], r4 = [%.17g, %.17g], R = [%.17g, %.17g]\n",
//							vars.la.get_lb(), vars.la.get_ub(), vars.radii[0].get_lb(), vars.radii[0].get_ub(), 
//							vars.radii[1].get_lb(), vars.radii[1].get_ub(), vars.radii[2].get_lb(), vars.radii[2].get_ub(),
//							vars.radii[3].get_lb(), vars.radii[3].get_ub(), R.get_lb(), R.get_ub()
//						);
						push_interval_set_to_buffer(vars, output, buffer_size, current_size);
//						algcuda::trap();
					}
				}
			}
		}
	}
}

static __host__ void find_critical_intervals_in_call_kernel(const Variables& intervals, Interval_buffer& buffers, std::vector<Variables>& append_to, bool is_initialization) {
	const int r1_parallel = is_initialization ? r1_parallel_init : r1_parallel_subdiv;
	const int r2_parallel = is_initialization ? r2_parallel_init : r2_parallel_subdiv;
	const int r3_parallel = is_initialization ? r3_parallel_init : r3_parallel_subdiv;
	const int r4_parallel = is_initialization ? r4_parallel_init : r4_parallel_subdiv;

	const dim3 grid_dims(r1_parallel, r2_parallel);
	const dim3 block_dims(r3_parallel, r4_parallel);

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

		std::vector<Variables> host_buffer(buffer_size_out);
		algcuda::device::copy(buffers.buffer, buffer_size_out, host_buffer.data());
		std::vector<Variables> merged = merge_adjacent_r1(host_buffer);
		append_to.insert(append_to.end(), merged.begin(), merged.end());
	}
}

static void __host__ find_critical_intervals_in(const Variables& intervals, Interval_buffer& buffers, std::vector<Variables>& append_to, bool interruptible, bool is_initialization, std::size_t idx = 0, std::size_t num_intervals = 0, int generation = -1) {
	const int la_subintervals = is_initialization ? la_subintervals_init : la_subintervals_subdiv[generation-1];
	for(int la_offset = 0; la_offset < la_subintervals; ++la_offset) {
		IV la = get_subinterval(intervals.la, la_offset, la_subintervals);
		Variables subvars{ la, { intervals.radii[0], intervals.radii[1], intervals.radii[2], intervals.radii[3] }};
		find_critical_intervals_in_call_kernel(subvars, buffers, append_to, is_initialization);

		if(!is_initialization) {
			std::cerr << "Interval " << idx << " of " << num_intervals << " - ";
		}

		std::cerr << "λ: " << std::setw(3) << la_offset << " of " << std::setw(3) << la_subintervals << " done (" << append_to.size() << " critical intervals found)                \r" << std::flush;

		if(interruptible && was_interrupted) {
			return;
		}
	}
}

struct Max_distances {
	bool have_range_2d, have_range_3d;
	IV la_range_2d, la_range_3d;
	IV dist_r1_2d;
	Variables dist_r1_2d_criticals[2];
	IV dist_r2_2d;
	Variables dist_r2_2d_criticals[2];
	double lb_h1_2d;
	double ub_la_by_S1;
	double ub_R3_2d;
	IV dist_r1_3d;
	Variables dist_r1_3d_criticals[2];
	IV dist_r2_3d;
	Variables dist_r2_3d_criticals[2];
	IV dist_r3_3d;
	Variables dist_r3_3d_criticals[2];
	double ub_R4_3d;
	IV S1, h2, S3;
	double pocket_width_3d;
};

static __global__ void kernel_compute_distance_from_theoretical_worst_cases(const Variables* criticals, std::size_t num_criticals, Max_distances* output) {
	output->have_range_2d = false;
	output->have_range_3d = false;
	output->la_range_3d = IV{1.0,1.0};
	output->la_range_2d = IV{2.0,2.0};
	output->dist_r1_2d = output->dist_r2_2d = IV{0.0,0.0};
	output->ub_R3_2d = output->ub_R4_3d = 0.0;
	output->dist_r1_3d = output->dist_r2_3d = output->dist_r3_3d = IV{0.0,0.0};
	output->pocket_width_3d = -1.0;
	output->lb_h1_2d = 1.0;
	output->ub_la_by_S1 = 1.0;

	for(std::size_t i = 0; i < num_criticals; ++i) {
		Variables vars = criticals[i];
		if(vars.radii[2].get_ub() < 0.1) {
			// classify this as 2-disk critical
			// r_1 is the circumcircle; r_2^2 is 0.25
			output->have_range_2d = true;
			output->la_range_2d.do_join(vars.la);
			IV r1_theoretical = 0.25 * (vars.la.square() + 1.0);
			IV r2_theoretical{0.25,0.25};

			IV dist_r1_2d = vars.radii[0] - r1_theoretical;
			if(dist_r1_2d.get_lb() < output->dist_r1_2d.get_lb()) {
				output->dist_r1_2d_criticals[0] = vars;
			}
			if(dist_r1_2d.get_ub() > output->dist_r1_2d.get_ub()) {
				output->dist_r1_2d_criticals[1] = vars;
			}
			output->dist_r1_2d.do_join(dist_r1_2d);

			IV dist_r2_2d = vars.radii[1] - r2_theoretical;
			if(dist_r2_2d.get_lb() < output->dist_r2_2d.get_lb()) {
				output->dist_r2_2d_criticals[0] = vars;
			}
			if(dist_r2_2d.get_ub() > output->dist_r2_2d.get_ub()) {
				output->dist_r2_2d_criticals[1] = vars;
			}
			output->dist_r2_2d.do_join(dist_r2_2d);
			
			IV R3 = required_weight_for(vars.la) - vars.radii[0] - vars.radii[1];
			if(R3.get_ub() > output->ub_R3_2d) {
				output->ub_R3_2d = R3.get_ub();
			}

			double lb_S1 = __dmul_rd(2.0, __dsqrt_rd(__dadd_rd(vars.radii[0].get_lb(), -0.25)));
			double ub_la_by_S1 = __ddiv_ru(vars.la.get_ub(), lb_S1);
			if(ub_la_by_S1 > output->ub_la_by_S1) {
				output->ub_la_by_S1 = ub_la_by_S1;
			}
			double ub_la_m_S1_half = __dadd_ru(vars.la.get_ub(), -__dmul_rd(0.5, lb_S1));
			double lb_h1 = __dmul_rd(2.0, __dsqrt_rd(__dadd_rd(vars.radii[0].get_lb(), -__dmul_ru(ub_la_m_S1_half,ub_la_m_S1_half))));
			if(lb_h1 < output->lb_h1_2d) {
				output->lb_h1_2d = lb_h1;
			}
		} else {
			// classify this as 3-disk critical
			output->have_range_3d = true;
			output->la_range_3d.do_join(vars.la);
			IV la_2 = vars.la.square();
			IV la_4 = la_2.square();
			IV theory_r = (16.0 * la_4 + 40.0 * la_2 + 9.0) / (256.0 * la_2);

			IV dist_r1_3d = vars.radii[0] - theory_r;
			if(dist_r1_3d.get_lb() < output->dist_r1_3d.get_lb()) {
				output->dist_r1_3d_criticals[0] = vars;
			}
			if(dist_r1_3d.get_ub() > output->dist_r1_3d.get_ub()) {
				output->dist_r1_3d_criticals[1] = vars;
			}
			IV dist_r2_3d = vars.radii[1] - theory_r;
			if(dist_r2_3d.get_lb() < output->dist_r2_3d.get_lb()) {
				output->dist_r2_3d_criticals[0] = vars;
			}
			if(dist_r2_3d.get_ub() > output->dist_r2_3d.get_ub()) {
				output->dist_r2_3d_criticals[1] = vars;
			}
			IV dist_r3_3d = vars.radii[2] - theory_r;
			if(dist_r3_3d.get_lb() < output->dist_r3_3d.get_lb()) {
				output->dist_r3_3d_criticals[0] = vars;
			}
			if(dist_r3_3d.get_ub() > output->dist_r3_3d.get_ub()) {
				output->dist_r3_3d_criticals[1] = vars;
			}
			output->dist_r1_3d.do_join(dist_r1_3d);
			output->dist_r2_3d.do_join(dist_r2_3d);
			output->dist_r3_3d.do_join(dist_r3_3d);
			
			IV R4 = required_weight_for(vars.la) - vars.radii[0] - vars.radii[1] - vars.radii[2];
			if(R4.get_ub() > output->ub_R4_3d) {
				output->ub_R4_3d = R4.get_ub();
			}

			IV S1 = sqrt(4.0 * vars.radii[0] - 1.0);
			IV h2 = sqrt(4.0 * vars.radii[1] - (vars.la - S1).square());
			IV S3 = sqrt(4.0 * vars.radii[2] - (1.0-h2).square());
			double ub_pocket_width = (vars.la - S1 - S3).get_ub();

			if(ub_pocket_width > output->pocket_width_3d) {
				if(output->pocket_width_3d < 0.0) {
					output->S1 = S1;
					output->h2 = h2;
					output->S3 = S3;
				}

				output->pocket_width_3d = ub_pocket_width;
			}

			output->S1.do_join(S1);
			output->h2.do_join(h2);
			output->S3.do_join(S3);
		}
	}
}

void circlecover::tight_rectangle::find_critical_intervals() {
	output_begin_case("Critical intervals for rectangles (tight)");
	Interval_buffer buffer;
	Variables initial{
		IV{1.0, 2.0898841580413818342},
		{ {0.0, DBL_MAX}, {0.0, DBL_MAX}, {0.0, DBL_MAX}, {0.0, DBL_MAX} }
	};

	std::vector<Variables> current_gen, next_gen;

	std::cerr << "Initializing..." << std::endl;
	find_critical_intervals_in(initial, buffer, next_gen, false, true);
	current_gen = next_gen;
	next_gen.clear();

	std::cerr << "Starting refinement - press Ctrl-C to interrupt and use best result achieved so far...                                               " << std::endl;
	start_interruptible_computation();

	for(int g = 1; g <= num_generations; ++g) {
		std::cerr << "Running generation " << g << " on " << current_gen.size() << " intervals...                                                                                                     " << std::endl;
		std::size_t idx = 1;
		for(const Variables& s : current_gen) {
			find_critical_intervals_in(s, buffer, next_gen, true, false, idx, current_gen.size(), g);
			if(was_interrupted) {
				break;
			}
			++idx;
		}

		if(was_interrupted) {
			break;
		}

		current_gen = next_gen;
		next_gen.clear();
		if(current_gen.empty()) {
			break;
		}
	}

	stop_interruptible_computation();

	if(!current_gen.empty()) {
		auto criticals_d = algcuda::device::make_unique<Variables[]>(current_gen.size());
		algcuda::device::copy(current_gen.data(), current_gen.size(), criticals_d);
		auto result_d = algcuda::device::make_unique<Max_distances>();
		kernel_compute_distance_from_theoretical_worst_cases<<<1,1>>>(criticals_d.get(), current_gen.size(), result_d.get());
		Max_distances result;
		algcuda::device::copy(result_d, &result);

		if(result.have_range_2d) {
			std::cout << "Two disks: Range for lambda that contains 2-disk critical intervals: " << result.la_range_2d << "; maximal distances from theoretical values: " << std::endl;
			std::cout << "\tr_1^2: " << result.dist_r1_2d << ", r_2^2: " << result.dist_r2_2d << ", R_3 <= "
				      << result.ub_R3_2d << ", h_1 >= " << result.lb_h1_2d << ", lambda / S_1 <= " << result.ub_la_by_S1 << std::endl;
			std::cout << "\tDominating criticals for r_1: " << std::endl;
			std::cout << "\t\t" << result.dist_r1_2d_criticals[0] << std::endl;
			std::cout << "\t\t" << result.dist_r1_2d_criticals[1] << std::endl;
			std::cout << "\tDominating criticals for r_2: " << std::endl;
			std::cout << "\t\t" << result.dist_r2_2d_criticals[0] << std::endl;
			std::cout << "\t\t" << result.dist_r2_2d_criticals[1] << std::endl;
		}

		if(result.have_range_3d) {
			std::cout << "Three disks: Range for lambda that contains 3-disk critical intervals: " << result.la_range_3d << "; maximal distances from theoretical values: " << std::endl;
			std::cout << "\tr_1^2: " << result.dist_r1_3d << ", r_2^2: " << result.dist_r2_3d << ", r_3^2: " << result.dist_r3_3d << ", R_4 <= " << result.ub_R4_3d << std::endl;
			std::cout << "\tS_1: " << result.S1 << ", h_2: " << result.h2 << ", S_3: " << result.S3 << ", pocket width <= " << result.pocket_width_3d << std::endl;
			std::cout << "\tDominating criticals for r_1: " << std::endl;
			std::cout << "\t\t" << result.dist_r1_3d_criticals[0] << std::endl;
			std::cout << "\t\t" << result.dist_r1_3d_criticals[1] << std::endl;
			std::cout << "\tDominating criticals for r_2: " << std::endl;
			std::cout << "\t\t" << result.dist_r2_3d_criticals[0] << std::endl;
			std::cout << "\t\t" << result.dist_r2_3d_criticals[1] << std::endl;
			std::cout << "\tDominating criticals for r_3: " << std::endl;
			std::cout << "\t\t" << result.dist_r3_3d_criticals[0] << std::endl;
			std::cout << "\t\t" << result.dist_r3_3d_criticals[1] << std::endl;
		}
	}

	output_end_case();
}

std::ostream& __host__ circlecover::tight_rectangle::operator<<(std::ostream& o, const Variables& v) {
	o << "λ = " << v.la;
	for(int i = 0; i < 4; ++i) {
		o << ", r²_" << (i+1) << " = " << v.radii[i];
	}
	return o;
}
