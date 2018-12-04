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
#include "../operations.cuh"

#include <cfloat>

using namespace circlecover;
using namespace circlecover::rectangle_size_bound;

static __device__ bool multi_disk_strip_vertical_nd(const Variables& vars, const Intermediate_values& vals, int strip_disks) {
	double ub_weight = 0.0;
	for(int j = 0; j < strip_disks; ++j) {
		ub_weight = __dadd_ru(ub_weight, vars.radii[j].get_ub());
	}
		
	// how much width is necessary to have good enough density
	double ub_width_necessary = __ddiv_ru(ub_weight, circlecover::rectangle_size_bound::critical_ratio);
	double ub_width_sq = __dmul_ru(ub_width_necessary, ub_width_necessary);

	double htotal = 0.0;
	for(int j = 0; j < strip_disks; ++j) {
		double hcur = __dadd_rd(__dmul_rd(4.0, vars.radii[j].get_lb()), -ub_width_sq);
		if(hcur <= 0.0) {
			break;
		}

		htotal = __dadd_rd(htotal, __dsqrt_rd(hcur));
	}

	if(htotal < 1.0) {
		return false;
	}

	// the largest remaining disk must fit the remainder
	double ub_largest_rem = (strip_disks == 6 ? vars.radii[5].get_ub() : vars.radii[strip_disks].get_ub());
	double lb_remaining_width = __dadd_rd(vars.la.get_lb(), -ub_width_necessary);
	return lb_remaining_width >= 1.0 || circlecover::rectangle_size_bound::disk_satisfies_size_bound(ub_largest_rem, lb_remaining_width);
}

static __device__ bool r1_r2_stacked_r3_r4_horizontal(const Variables& vars, const Intermediate_values& vals) {
	double ub_weight = 0.0;
	for(int j = 0; j < 4; ++j) {
		ub_weight = __dadd_ru(ub_weight, vars.radii[j].get_ub());
	}

	double ub_width_necessary = __ddiv_ru(ub_weight, circlecover::rectangle_size_bound::critical_ratio);
	double ub_width_sq = __dmul_ru(ub_width_necessary, ub_width_necessary);

	double h1 = __dadd_rd(__dmul_rd(4.0, vars.radii[0].get_lb()), -ub_width_sq);
	double h2 = __dadd_rd(__dmul_rd(4.0, vars.radii[1].get_lb()), -ub_width_sq);
	if(h1 <= 0.0 || h2 <= 0.0) {
		return false;
	}

	double h12  = __dadd_rd(__dsqrt_rd(h1), __dsqrt_rd(h2));
	double hrem = __dadd_ru(1.0, -h12);
	double hrem_sq = __dmul_ru(hrem, hrem);

	double w4 = __dadd_rd(__dmul_rd(4.0, vars.radii[3].get_lb()), -hrem_sq);
	double w3 = __dadd_rd(__dmul_rd(4.0, vars.radii[2].get_lb()), -hrem_sq);
	if(w4 <= 0.0 || w3 <= 0.0) {
		return false;
	}

	double w34 = __dadd_rd(__dsqrt_rd(w3), __dsqrt_rd(w4));
	if(w34 < ub_width_necessary) {
		return false;
	}

	double lb_remaining_width = __dadd_rd(vars.la.get_lb(), -ub_width_necessary);
	return lb_remaining_width >= 1.0 || circlecover::rectangle_size_bound::disk_satisfies_size_bound(vars.radii[4].get_ub(), lb_remaining_width);
}

static __device__ bool r1_r2_stacked_r3_r4_and_r5_r6_horizontal(const Variables& vars, const Intermediate_values& vals) {
	double ub_weight = 0.0;
	for(int j = 0; j < 6; ++j) {
		ub_weight = __dadd_ru(ub_weight, vars.radii[j].get_ub());
	}

	// we fix this width as the width we are trying to cover
	double ub_width_necessary = __ddiv_ru(ub_weight, circlecover::rectangle_size_bound::critical_ratio);
	double ub_width_sq = __dmul_ru(ub_width_necessary, ub_width_necessary);

	double h1 = __dadd_rd(__dmul_rd(4.0, vars.radii[0].get_lb()), -ub_width_sq);
	double h2 = __dadd_rd(__dmul_rd(4.0, vars.radii[1].get_lb()), -ub_width_sq);
	if(h1 <= 0.0 || h2 <= 0.0) {
		return false;
	}

	IV r34_diff = vars.radii[2] - vars.radii[3];
	r34_diff.tighten_lb(0.0);

	IV w3 = (2.0 * r34_diff) / ub_width_necessary + 0.5 * IV(ub_width_necessary,ub_width_necessary);
	IV w3_sq = w3 * w3;
	
	// check r3, r4 are large enough
	if(!definitely(4.0 * vars.radii[2] > w3_sq) || !definitely(4.0 * vars.radii[3] > (ub_width_necessary-w3) * (ub_width_necessary-w3))) {
		return false;
	}

	double lb_height_r34 = __dsqrt_rd(__dadd_rd(__dmul_rd(4.0, vars.radii[2].get_lb()), -w3_sq.get_ub()));
	
	IV r56_diff = vars.radii[4] - vars.radii[5];
	r56_diff.tighten_lb(0.0);

	IV w5 = (2.0 * r56_diff) / ub_width_necessary + 0.5 * IV(ub_width_necessary,ub_width_necessary);
	IV w5_sq = w5 * w5;

	// check r5, r6 are large enough
	if(!definitely(4.0 * vars.radii[4] > w5_sq) || !definitely(4.0 * vars.radii[5] > (ub_width_necessary-w5) * (ub_width_necessary-w5))) {
		return false;
	}

	double lb_height_r56 = __dsqrt_rd(__dadd_rd(__dmul_rd(4.0, vars.radii[4].get_lb()), -w5_sq.get_ub()));
	double lb_height_r12 = __dadd_rd(__dsqrt_rd(h1), __dsqrt_rd(h2));

	double lb_htotal = __dadd_rd(lb_height_r12, __dadd_rd(lb_height_r34, lb_height_r56));
	if(lb_htotal < 1.0) {
		return false;
	}

	double lb_remaining_width = __dadd_rd(vars.la.get_lb(), -ub_width_necessary);
	return lb_remaining_width >= 1.0 || circlecover::rectangle_size_bound::disk_satisfies_size_bound(vars.radii[5].get_ub(), lb_remaining_width);
}

__device__ static bool six_disk_strip_three_rows(const Variables& vars, const Intermediate_values& vals) {
	using namespace circlecover;
	using namespace circlecover::rectangle_size_bound;

	// the total width we use for the strip
	double ub_weight = 0.0;
	for(int i = 0; i < 6; ++i) {
		ub_weight = __dadd_ru(ub_weight, vars.radii[i].get_ub());
	}

	// how much width is necessary to have good enough density
	double ub_width_necessary = __ddiv_ru(ub_weight, critical_ratio);

	// bound on r7
	double ub_r7 = vars.radii[5].get_ub();
	if(vals.R.get_ub() < ub_r7) {
		ub_r7 = vals.R.get_ub();
	}

	// check that r7 still fits
	double lb_width_remaining = __dadd_rd(vars.la.get_lb(), -ub_width_necessary);
	if(lb_width_remaining < 1.0 && !disk_satisfies_size_bound(ub_r7, lb_width_remaining)) {
		return false;
	}

	// check that the first disks can cover the strip
	IV w{ub_width_necessary,ub_width_necessary};
	IV w_sq = w * ub_width_necessary;
	IV w1 = 2.0/w * (vars.radii[0] - vars.radii[3]) + 0.5*w;
	IV w2 = 2.0/w * (vars.radii[1] - vars.radii[2]) + 0.5*w;

	double ub_diff14 = __dadd_ru(vars.radii[0].get_ub(), -vars.radii[3].get_lb());
	double ub_diff14_sq = __dmul_ru(ub_diff14, ub_diff14);
	double lb_h14 = __dsqrt_rd(__dadd_rd(__dadd_rd(__dmul_rd(2.0, __dadd_rd(vars.radii[0].get_lb(), vars.radii[3].get_lb())), -__dmul_ru(0.25, w_sq.get_ub())), -__ddiv_ru(__dmul_ru(4.0, ub_diff14_sq), w_sq.get_lb())));
	if(lb_h14 < 0) {
		return false;
	}
	lb_h14 = __dsqrt_rd(lb_h14);

	double ub_diff23 = __dadd_ru(vars.radii[1].get_ub(), -vars.radii[2].get_lb());
	double ub_diff23_sq = __dmul_ru(ub_diff23, ub_diff23);
	double lb_h23 = __dsqrt_rd(__dadd_rd(__dadd_rd(__dmul_rd(2.0, __dadd_rd(vars.radii[1].get_lb(), vars.radii[2].get_lb())), -__dmul_ru(0.25, w_sq.get_ub())), -__ddiv_ru(__dmul_ru(4.0, ub_diff23_sq), w_sq.get_lb())));
	if(lb_h23 < 0) {
		return false;
	}
	lb_h23 = __dsqrt_rd(lb_h23);

	// place r5 covering the left side
	IV height_rem = 1.0 - IV{lb_h23,lb_h23} - lb_h14;
	if(height_rem.get_ub() <= 0.0) {
		return true;
	}

	double lb_x_r5 = __dadd_rd(vars.radii[4].get_lb(), -__dmul_ru(0.25, __dmul_ru(height_rem.get_ub(), height_rem.get_ub())));
	double lb_w_r6 = __dadd_rd(vars.radii[5].get_lb(), -__dmul_ru(0.25, __dmul_ru(height_rem.get_ub(), height_rem.get_ub())));
	if(lb_x_r5 < 0 || lb_w_r6 < 0) {
		return false;
	}
	lb_x_r5 = __dsqrt_rd(lb_x_r5);
	lb_w_r6 = __dsqrt_rd(lb_w_r6);
	double ub_x_r6 = __dadd_ru(ub_width_necessary, -lb_w_r6);
	IV y_r5 = 0.5 * height_rem + lb_h23;

	double y_r1 = __dadd_ru(1.0, -__dmul_rd(0.5, lb_h14));
	double y_r2 = __dmul_rd(0.5, lb_h23);
	Point c_r5{IV{lb_x_r5,lb_x_r5}, y_r5};
	Point c_r1{w-0.5*w1, IV{y_r1,y_r1}};
	Point c_r2{w-0.5*w2, IV{y_r2,y_r2}};
	Point c_r3{0.5*(w-w2), IV{y_r2,y_r2}};
	Point c_r4{0.5*(w-w1), IV{y_r1,y_r1}};
	Point c_r6{IV{ub_x_r6,ub_x_r6}, y_r5};
	Circle c1{c_r1, {vars.radii[0].get_lb(),vars.radii[0].get_lb()}};
	Circle c2{c_r2, {vars.radii[1].get_lb(),vars.radii[1].get_lb()}};
	Circle c5{c_r5, {vars.radii[4].get_lb(),vars.radii[4].get_lb()}};

	Intersection_points cr15 = intersection(c5, c1);
	Intersection_points cr25 = intersection(c5, c2);

	// if r5 does not definitely intersect the disks on the far side, fail
	if(!cr15.definitely_intersecting || !cr25.definitely_intersecting) {
		return false;
	}

	// check that the lexicographically smaller intersection points are in r4/r3
	if(squared_distance(c_r4, cr15.p[0]).get_ub() > vars.radii[3].get_lb()) {
		return false;
	}
	if(squared_distance(c_r3, cr25.p[0]).get_ub() > vars.radii[2].get_lb()) {
		return false;
	}

	// check that r6 covers the remaining area
	return squared_distance(c_r6, cr15.p[1]).get_ub() <= vars.radii[5].get_lb() && squared_distance(c_r6, cr25.p[1]).get_ub() <= vars.radii[5].get_lb();
}

__device__ bool circlecover::rectangle_size_bound::multi_disk_strip_vertical(const Variables& vars, const Intermediate_values& vals) {
	for(int i = 2; i <= 6; ++i) {
		if(multi_disk_strip_vertical_nd(vars, vals, i)) {
			return true;
		}
	}

	if(r1_r2_stacked_r3_r4_horizontal(vars,vals)) {
		return true;
	}

	if(r1_r2_stacked_r3_r4_and_r5_r6_horizontal(vars,vals)) {
		return true;
	}

	if(six_disk_strip_three_rows(vars, vals)) {
		return true;
	}

	return false;
}

__device__ static bool adv_multi_disk_strip_vertical_one_strip(const Variables& vars, const Intermediate_values& vals, unsigned disk_set) {
	using namespace circlecover;
	using namespace circlecover::rectangle_size_bound;

	double ub_weight = 0.0;
	for(int j = 0; j < 6; ++j) {
		bool disk_present = ((disk_set & (1 << j)) != 0);
		if(disk_present) {
			ub_weight = __dadd_ru(ub_weight, vars.radii[j].get_ub());
		}
	}

	// how much width is necessary to have good enough density
	double ub_width_necessary = __ddiv_ru(ub_weight, critical_ratio);
	double ub_width_sq = __dmul_ru(ub_width_necessary, ub_width_necessary);

	double htotal = 0.0;
	int largest_remaining = -1;
	for(int j = 0; j < 6; ++j) {
		bool disk_present = ((disk_set & (1 << j)) != 0);
		if(!disk_present) {
			if(largest_remaining < 0) {
				largest_remaining = j;
			}

			continue;
		}

		double hcur = __dadd_rd(__dmul_rd(4.0, vars.radii[j].get_lb()), -ub_width_sq);
		if(hcur <= 0.0) {
			return false;
		}

		htotal = __dadd_rd(htotal, __dsqrt_rd(hcur));
	}

	if(htotal < 1.0) {
		return false;
	}

	double ub_largest_rem = vars.radii[largest_remaining].get_ub();
	double lb_remaining_width = __dadd_rd(vars.la.get_lb(), -ub_width_necessary);
	return lb_remaining_width >= 1.0 || disk_satisfies_size_bound(ub_largest_rem, lb_remaining_width);
}

__device__ bool circlecover::rectangle_size_bound::advanced_multi_disk_strip_vertical(const Variables &vars, const Intermediate_values &vals) {
	for(unsigned s = 1; s < 63; ++s) {
		if(adv_multi_disk_strip_vertical_one_strip(vars, vals, s)) {
			return true;
		}
	}

	return false;
}
