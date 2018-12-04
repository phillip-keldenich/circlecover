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

bool __device__ circlecover::rectangle_size_bound::uneven_split_recursion(const Variables& vars, const Intermediate_values& vals) {
	// handle corner cases
	if(vars.radii[0].get_lb() <= 0.0) {
		return false;
	}

	// how much of the width is needed for a rectangle that is large enough to contain r1
	double ub_width_needed_r1 = bound_required_height(vars.radii[0].get_ub());

	// if lambda is not definitely greater than this, we have to fail
	if(ub_width_needed_r1 >= vars.la.get_lb()) {
		return false;
	}

	double ub_weight_needed = __dmul_ru(ub_width_needed_r1, critical_ratio);

	int possibly_exceeded_at = -1;
	int definitely_exceeded_at = -1;
	IV w_l{0.0,0.0};

	for(int i = 0; i < 6; ++i) {
		w_l += vars.radii[i];

		if(possibly_exceeded_at < 0 && w_l.get_ub() >= ub_weight_needed) {
			possibly_exceeded_at = i;
		}

		if(w_l.get_lb() >= ub_weight_needed) {
			definitely_exceeded_at = i;
			break;
		}
	}

	double ub_largest_rem = (possibly_exceeded_at < 0 || possibly_exceeded_at == 5) ? vars.radii[5].get_ub() : vars.radii[possibly_exceeded_at+1].get_ub();
	double ub_D1;

	if(definitely_exceeded_at < 0) {
		ub_D1 = __dadd_ru(ub_weight_needed, vars.radii[5].get_ub());
	} else {
		ub_D1 = w_l.get_ub();
	}

	if(ub_D1 >= __dmul_rd(vars.la.get_lb(), critical_ratio)) {
		// D2 could be empty
		return false;
	}

	double ub_w1 = __ddiv_ru(ub_D1, critical_ratio);
	double lb_wrem = __dadd_rd(vars.la.get_lb(), -ub_w1);

	return lb_wrem >= 1.0 || disk_satisfies_size_bound(ub_largest_rem, lb_wrem);
}

bool __device__ circlecover::rectangle_size_bound::shortcut_uneven_split_recursion(IV la, IV r1, IV r2) {
	// handle corner cases
	if(r1.get_lb() <= 0.0) {
		return false;
	}

	// how much of the width is needed for a rectangle that is large enough to contain r1
	double ub_width_needed_r1 = bound_required_height(r1.get_ub());

	// if lambda is not definitely greater than this, we have to fail
	if(ub_width_needed_r1 >= la.get_lb()) {
		return false;
	}

	double ub_weight_g1  = __dadd_ru(r2.get_ub(), __ddiv_ru(ub_width_needed_r1, critical_ratio));
	double lb_weight_rem = __dadd_rd(__dmul_rd(la.get_lb(), critical_ratio), -ub_weight_g1);
	if(lb_weight_rem <= 0.0) {
		return false;
	}

	double lb_width_rem = __ddiv_rd(lb_weight_rem, critical_ratio);
	return lb_width_rem >= 1.0 || disk_satisfies_size_bound(r2.get_ub(), lb_width_rem);
}

bool __device__ circlecover::rectangle_size_bound::shortcut_uneven_split_recursion(IV la, IV r1, IV r2, IV r3) {
	if(r1.get_lb() <= 0.0) {
		return false;
	}

	// how much of the width is needed for a rectangle that is large enough to contain r1
	double ub_width_needed_r1 = bound_required_height(r1.get_ub());

	// if lambda is not definitely greater than this, we have to fail
	if(ub_width_needed_r1 >= la.get_lb()) {
		return false;
	}

	// we do not need to check that r1+r2+r3 are too large - 3 disks are never enough to achieve
	// enough weight for a region they fit into according to the size bound
	double ub_weight_g1 = __dadd_ru(r3.get_ub(), __ddiv_ru(ub_width_needed_r1, critical_ratio));
	double lb_weight_rem = __dadd_rd(__dmul_rd(la.get_lb(), critical_ratio), -ub_weight_g1);
	if(lb_weight_rem <= 0.0) {
		return false;
	}

	double lb_width_rem = __ddiv_rd(lb_weight_rem, critical_ratio);
	return lb_width_rem >= 1.0 || disk_satisfies_size_bound(r3.get_ub(), lb_width_rem);
}

