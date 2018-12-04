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

static inline __device__ bool l_shaped_recursion_two_disks_vertical_one_horizontal(const Variables& vars, const Intermediate_values& vals, double ub_w, double lb_weight) {
	// the remaining weight
	double lb_R4 = __dadd_rd(lb_weight, -vars.radii[2].get_ub());
	
	// try placing r3 as square
	double lb_side_r3 = __dsqrt_rd(__dmul_rd(2.0, vars.radii[2].get_lb()));
	IV height_top = IV{1.0,1.0} - lb_side_r3;
	double lb_min_dim_top = height_top.get_lb();
	if(ub_w < lb_min_dim_top) {
		lb_min_dim_top = ub_w;
	}

	// check the top/right-bottom decomposition
	if(disk_satisfies_size_bound(vars.radii[3].get_ub(), lb_min_dim_top)) {
		// we use at most r4 + efficiency * area weight for the top
		double ub_weight_top = __dadd_ru(__dmul_ru(critical_ratio, __dmul_ru(height_top.get_ub(), ub_w)), vars.radii[3].get_ub());
		double lb_weight_rem_bottom_right = __dadd_rd(lb_R4, -ub_weight_top);
		if(lb_weight_rem_bottom_right > 0) {
			double width_bottom_right = __dadd_ru(ub_w, -lb_side_r3);
			double min_dim_br = width_bottom_right;
			if(lb_side_r3 < min_dim_br) {
				min_dim_br = lb_side_r3;
			}

			// cover the remaining width_bottom_right x lb_side_r3 rectangle in the bottom-right corner
			if(disk_satisfies_size_bound(vars.radii[4].get_ub(), min_dim_br)) {
				double weight_needed = __dmul_ru(critical_ratio, __dmul_ru(lb_side_r3, width_bottom_right));
				if(weight_needed <= lb_weight_rem_bottom_right) {
					return true;
				}
			} else {
				if(can_recurse(lb_weight_rem_bottom_right, vars.radii[4].get_ub(), width_bottom_right, lb_side_r3)) {
					return true;
				}
			}
		}
	}

	// check the top-left/right decomposition
	IV width_right = vars.la - lb_side_r3;
	double lb_min_dim_right = width_right.get_lb();
	if(lb_min_dim_right > 1.0) {
		lb_min_dim_right = 1.0;
	}
	if(lb_min_dim_right >= 1.0 || disk_satisfies_size_bound(vars.radii[3].get_ub(), lb_min_dim_right)) {
		double ub_weight_right = __dadd_ru(__dmul_ru(critical_ratio, width_right.get_ub()), vars.radii[3].get_ub());
		double lb_weight_rem_top_left = __dadd_rd(lb_R4, -ub_weight_right);
		if(lb_weight_rem_top_left > 0) {
			double height_top_left = __dadd_ru(1.0, -lb_side_r3);
			double min_dim_tl = height_top_left;
			if(lb_side_r3 < min_dim_tl) {
				min_dim_tl = lb_side_r3;
			}

			// cover the remaining height_top_left x lb_side_r3 rectangle in the top left corner
			if(disk_satisfies_size_bound(vars.radii[4].get_ub(), min_dim_tl)) {
				double weight_needed = __dmul_ru(critical_ratio, __dmul_ru(lb_side_r3, height_top_left));
				if(weight_needed <= lb_weight_rem_top_left) {
					return true;
				}
			} else {
				if(can_recurse(lb_weight_rem_top_left, vars.radii[4].get_ub(), lb_side_r3, height_top_left)) {
					return true;
				}
			}
		}
	}

	{
		// try to cover all remaining width using r3
		double ub_w_sq = __dmul_ru(ub_w, ub_w);
		double lb_h3 = __dadd_rd(__dmul_rd(4.0, vars.radii[2].get_lb()), -ub_w_sq);
		if(lb_h3 > 0.0) {
			lb_h3 = __dsqrt_rd(lb_h3);
			double ub_disk_weight = vars.radii[3].get_ub();
			double ub_hrem = __dadd_ru(1.0, -lb_h3);
			double rem_min = ub_hrem;
			double ub_area_rem = __dmul_ru(ub_w, ub_hrem);
			if(ub_w < rem_min) { rem_min = ub_w; }
			if(disk_satisfies_size_bound(ub_disk_weight, rem_min)) {
				if(__dmul_ru(ub_area_rem, critical_ratio) <= lb_R4) {
					return true;
				}
			} else {
				if(can_recurse(lb_R4, ub_disk_weight, ub_w, ub_hrem)) {
					return true;
				}
			}
		}
	}

	return false;
}

static inline __device__ bool l_shaped_recursion_two_disks_vertical(const Variables& vars, const Intermediate_values& vals) {
	double ub_weight_vertical  = __dadd_ru(vars.radii[0].get_ub(), vars.radii[1].get_ub());
	double lb_weight_remaining = __dadd_rd(__dmul_rd(vars.la.get_lb(), critical_ratio), -ub_weight_vertical);
	double lb_wmax = two_disks_maximize_height(vars.radii[0], vars.radii[1], 1.0);
	double ub_wrem = __dadd_ru(vars.la.get_ub(), -lb_wmax);

	if(__dmul_rd(ub_wrem, critical_ratio) > lb_weight_remaining) {
		return false;
	}

	return l_shaped_recursion_two_disks_vertical_one_horizontal(vars, vals, ub_wrem, lb_weight_remaining);
}

static inline __device__ bool l_shaped_recursion_two_disks_vertical_two_horizontal(const Variables& vars, const Intermediate_values& vals) {
	double lb_w12 = two_disks_maximize_height(vars.radii[0], vars.radii[1], 1.0);
	if(lb_w12 <= 0.0) {
		return false;
	}

	double ub_wrem = __dadd_ru(vars.la.get_ub(), -lb_w12);
	double lb_h34 = two_disks_maximize_height(vars.radii[2], vars.radii[3], ub_wrem);
	if(lb_h34 <= 0.0) {
		return false;
	}

	double hrem = __dadd_ru(1.0, -lb_h34);
	double lb_R5 = __dadd_rd(__dmul_rd(vars.la.get_lb(), critical_ratio), -__dadd_ru(vars.radii[0].get_ub(), __dadd_ru(vars.radii[1].get_ub(), __dadd_ru(vars.radii[2].get_ub(), vars.radii[3].get_ub()))));
	return can_recurse(lb_R5, vars.radii[5].get_ub(), ub_wrem, hrem);
}

__device__ bool circlecover::rectangle_size_bound::l_shaped_recursion(const Variables& vars, const Intermediate_values& vals) {
	return l_shaped_recursion_two_disks_vertical(vars, vals) || l_shaped_recursion_two_disks_vertical_two_horizontal(vars, vals);
}

