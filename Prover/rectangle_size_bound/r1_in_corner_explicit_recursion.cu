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

using namespace circlecover;
using namespace circlecover::rectangle_size_bound;

namespace {
struct Backtracking_stack_entry {
	double reached_height;
	int used_disks;
	int current_choice;
};
}

__device__ static bool r1_in_corner_explicit_recursion_above_nd(const Variables& vars, const Intermediate_values& vals, int nd) {
	double ub_weight = 0.0;
	for(int j = nd; j >= 0; --j) {
		ub_weight = __dadd_ru(ub_weight, vars.radii[j].get_ub());
	}

	// compute the minimum width we have to cover to be efficient enough
	double ub_min_width = __ddiv_ru(ub_weight, critical_ratio);
	// the maximum remaining weight in a single disk
	double ub_max_rem_weight = (nd == 5) ? (vars.radii[5].get_ub() > vals.R.get_ub() ? vals.R.get_ub() : vars.radii[5].get_ub()) : vars.radii[nd+1].get_ub();
	// compute an upper bound on the width that has to remain due to the size bound
	double ub_required_remaining_width = bound_required_height(ub_max_rem_weight);
	// and the maximum allowed width based on it
	double lb_max_allowed_width = __dadd_rd(vars.la.get_lb(), -ub_required_remaining_width);

	// check whether the size upper bound and the weight lower bound allow for a solution
	if(ub_min_width > lb_max_allowed_width) {
		return false;
	}

	// check whether there is a way to cover a strip of width lb_w1
	double w1 = ub_min_width;
	double w1sq = __dmul_ru(w1,w1);
	double lb_h1 = __dadd_rd(__dmul_rd(4.0, vars.radii[0].get_lb()), -w1sq);
	if(lb_h1 <= 0) {
		return false;
	}
	lb_h1 = __dsqrt_rd(lb_h1);

	Backtracking_stack_entry stack[6];
	int stack_height = 1;
	stack[0].reached_height = lb_h1;
	stack[0].used_disks = 0;
	stack[0].current_choice = 0;

	while(stack_height > 0) {
		int s = stack_height-1;
		int remaining_disks = nd - stack[s].used_disks;
		int next_choice = ++stack[s].current_choice;

		// no more viable choices? backtrack
		if(next_choice > remaining_disks) {
			--stack_height;
			continue;
		}

		int first_unused = 1+stack[s].used_disks;
		if(next_choice == remaining_disks) {
			if(nd_can_cover(w1, __dadd_ru(1.0, -stack[s].reached_height), &vars.radii[first_unused], next_choice)) {
				return true;
			} else {
				// no disks left after this; backtrack
				--stack_height;
				continue;
			}
		}

		double lb_h = nd_maximize_covered(w1, &vars.radii[first_unused], next_choice);
		if(lb_h == 0) {
			// width is not reachable; do not expand this choice further
			continue;
		}

		double new_height = __dadd_rd(stack[s].reached_height, lb_h);
		if(new_height >= 1.0) {
			return true;
		}

		stack[stack_height].reached_height = new_height;
		stack[stack_height].used_disks = stack[s].used_disks + next_choice;
		stack[stack_height].current_choice = 0;
		++stack_height;
	}

	return false;
}

__device__ static bool r1_in_corner_explicit_recursion_right_nd(const Variables& vars, const Intermediate_values& vals, int nd) {
	// weight we are using
	double ub_weight = 0.0;
	for(int j = nd; j >= 0; --j) {
		ub_weight = __dadd_ru(ub_weight, vars.radii[j].get_ub());
	}

	// compute the minimum height that we have to cover to be efficient enough
	double ub_min_area   = __ddiv_ru(ub_weight, critical_ratio);
	double ub_min_height = __ddiv_ru(ub_min_area, vars.la.get_lb());
	// the maximum remaining weight in a single disk
	double ub_max_rem_weight = (nd == 5) ? (vars.radii[5].get_ub() > vals.R.get_ub() ? vals.R.get_ub() : vars.radii[5].get_ub()) : vars.radii[nd+1].get_ub();
	// compute an upper bound on the width that has to remain due to the size bound
	double ub_required_remaining_height = bound_required_height(ub_max_rem_weight);
	double lb_max_allowed_height = __dadd_rd(1.0, -ub_required_remaining_height);
	// there is no solution when the height that has to remain is too high
	if(ub_min_height > lb_max_allowed_height) {
		return false;
	}
	
	double h1 = ub_min_height;
	double h1sq = __dmul_ru(h1,h1);
	double lb_w1 = __dadd_rd(__dmul_rd(4.0, vars.radii[0].get_lb()), -h1sq);
	if(lb_w1 <= 0) {
		return false;
	}
	lb_w1 = __dsqrt_rd(lb_w1);

	Backtracking_stack_entry stack[6];
	int stack_height = 1;
	stack[0].reached_height = lb_w1;
	stack[0].used_disks = 0;
	stack[0].current_choice = 0;

	while(stack_height > 0) {
		int s = stack_height-1;
		int remaining_disks = nd - stack[s].used_disks;
		int next_choice = ++stack[s].current_choice;

		// no more viable choices? backtrack
		if(next_choice > remaining_disks) {
			--stack_height;
			continue;
		}

		int first_unused = 1+stack[s].used_disks;
		if(next_choice == remaining_disks) {
			if(nd_can_cover(h1, __dadd_ru(vars.la.get_ub(), -stack[s].reached_height), &vars.radii[first_unused], next_choice)) {
				return true;
			} else {
				// no disks left after this; backtrack
				--stack_height;
				continue;
			}
		}

		double lb_w = nd_maximize_covered(h1, &vars.radii[first_unused], next_choice);
		if(lb_w == 0) {
			// width is not reachable; do not expand this choice further
			continue;
		}

		double new_width = __dadd_rd(stack[s].reached_height, lb_w);
		if(new_width >= vars.la.get_ub()) {
			return true;
		}

		stack[stack_height].reached_height = new_width;
		stack[stack_height].used_disks = stack[s].used_disks + next_choice;
		stack[stack_height].current_choice = 0;
		++stack_height;
	}

	return false;
}

static __device__ bool r1_in_corner_right_34_and_2_recursion(const Variables& vars, const Intermediate_values& vals) {
	IV w1 = sqrt(2.0 * vars.radii[0]);
	IV wrem = vars.la - w1;
	
	// r5 has to fit into A (above r1)
	double lb_h_A = __dadd_rd(1.0, -w1.get_ub());
	if(!disk_satisfies_size_bound(vars.radii[4].get_ub(), lb_h_A)) {
		return false;
	}

	// the height r3 and r4 can cover (of the remaining width)
	double lb_h34 = two_disks_maximize_height(vars.radii[2], vars.radii[3], wrem.get_ub());
	if(lb_h34 <= 0.0 || lb_h34 >= w1.get_lb()) {
		return false;
	}

	double ub_hrem = __dadd_ru(w1.get_ub(), -lb_h34);
	double ub_hrem_sq = __dmul_ru(ub_hrem, ub_hrem);
	double lb_w2 = __dadd_rd(__dmul_rd(4.0, vars.radii[1].get_lb()), -ub_hrem_sq);
	if(lb_w2 <= 0.0) {
		return false;
	}
	lb_w2 = __dsqrt_rd(lb_w2);
	
	bool no_B = false;
	if(lb_w2 >= wrem.get_ub()) {
		// there is no B
		no_B = true;
		lb_w2 = wrem.get_lb();
	}

	if(lb_w2 > wrem.get_lb()) {
		lb_w2 = wrem.get_lb();
	}

	// area covered by r2,r3,r4 at least
	double lb_a234 = __dadd_rd(__dmul_rd(lb_w2, w1.get_lb()), __dmul_rd(lb_h34, __dadd_rd(wrem.get_lb(), -lb_w2)));
	double ub_weight234 = __dadd_ru(vars.radii[1].get_ub(), __dadd_ru(vars.radii[2].get_ub(), vars.radii[3].get_ub()));

	double lb_extra_weight = __dadd_rd(__dmul_rd(__dadd_rd(__dmul_rd(2.0, critical_ratio), -1.0), vars.radii[0].get_lb()), __dadd_rd(__dmul_rd(critical_ratio, lb_a234), -ub_weight234));
	if(lb_extra_weight <= 0.0) {
		return false;
	} else if(no_B) {
		return true;
	}

	// does r6 fit into B?
	IV h_B = w1 - lb_h34;
	IV w_B = wrem - lb_w2;
	double lb_min_B = h_B.get_lb();
	if(lb_min_B > w_B.get_lb()) { lb_min_B = w_B.get_lb(); }

	// we have some extra cost due to splitting
	double ub_extra_cost = vars.radii[5].get_ub();
	if(!disk_satisfies_size_bound(vars.radii[5].get_ub(), lb_min_B)) {
		// we also have to pay more for covering B using worst-case recursion
		double ub_critical_ratio_B = bound_worst_case_ratio(w_B, h_B);
		double ub_area_B = __dmul_ru(h_B.get_ub(), w_B.get_ub());
		ub_extra_cost = __dadd_ru(ub_extra_cost, __dmul_ru(__dadd_ru(ub_critical_ratio_B, -critical_ratio), ub_area_B));
	}
	
	return ub_extra_cost <= lb_extra_weight;
}

__device__ bool circlecover::rectangle_size_bound::r1_in_corner_explicit_recursion(const Variables& vars, const Intermediate_values& vals) {
	for(int i = 2; i <= 5; ++i) {
		if(r1_in_corner_explicit_recursion_above_nd(vars, vals, i) || r1_in_corner_explicit_recursion_right_nd(vars, vals, i)) {
			return true;
		}
	}

	return r1_in_corner_right_34_and_2_recursion(vars, vals);
}

