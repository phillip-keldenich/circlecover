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
#include <algcuda/exit.cuh>

using namespace circlecover;
using namespace circlecover::rectangle_size_bound;

static __device__ bool five_disk_full_cover_3v2_recursion_center(const Variables& vars, const Intermediate_values& vals) {
	// fix lambda - we cover the biggest possible rectangle
	IV la_covered{vars.la.get_ub(),vars.la.get_ub()};
	IV R6 = vars.radii[5] + vals.R;

	Max_height_strip_2 bottom_row = two_disks_maximal_height_strip(la_covered, vars.radii[0], vars.radii[1]);
	if(bottom_row.lb_height <= 0.0) {
		return false;
	}

	const double r3 = vars.radii[2].get_lb();
	const double r4 = vars.radii[3].get_lb();
	const double r5 = vars.radii[4].get_lb();
	double ub_hrem = __dadd_ru(1.0, -bottom_row.lb_height);
	double ub_hrem_sq = __dmul_ru(ub_hrem, ub_hrem);
	
	double w4 = __dadd_rd(__dmul_rd(4.0, r4), -ub_hrem_sq);
	double w5 = __dadd_rd(__dmul_rd(4.0, r5), -ub_hrem_sq);
	if(w4 <= 0.0 || w5 <= 0.0) {
		return false;
	}
	w4 = __dsqrt_rd(w4);
	w5 = __dsqrt_rd(w5);

	IV y45 = 1.0 - IV{0.5,0.5}*ub_hrem;
	Circle c4{{la_covered-IV{0.5,0.5}*w4,y45}, {r4,r4}};
	Circle c5{{IV{0.5,0.5}*w5,y45}, {r5,r5}};

	double ub_wrem = __dadd_ru(la_covered.get_ub(), -__dadd_rd(w4,w5));
	double ub_wrem_sq = __dmul_ru(ub_wrem, ub_wrem);
	double h3 = __dadd_rd(__dmul_rd(4.0, r3), -ub_wrem_sq);
	if(h3 <= 0.0) {
		return false;
	}
	h3 = __dsqrt_rd(h3);

	// even only considering rectangles, r3 is big enough
	if(h3 >= ub_hrem) {
		return true;
	}
	
	// consider the intersections explicitly
	Circle c3{{w5 + IV{0.5,0.5}*ub_wrem,1.0-IV{0.5,0.5}*h3}, {r3,r3}};

	Intersection_points x15 = intersection(bottom_row.c1, c5);
	Intersection_points x12 = intersection(bottom_row.c1, bottom_row.c2);
	Intersection_points x24 = intersection(bottom_row.c2, c4);

	if(!x12.definitely_intersecting || !x15.definitely_intersecting || !x24.definitely_intersecting) {
		algcuda::trap();
	}

	// check that we can distinguish the intersection points
	if(x15.p[0].x.get_ub() >= x15.p[1].x.get_lb() || x24.p[0].x.get_ub() >= x24.p[1].x.get_lb()) {
		return false;
	}

	// find the higher intersection point of the bottom row
	UB x12_first_higher = (x12.p[0].y > x12.p[1].y);
	if(!x12_first_higher.is_certain()) {
		return false;
	}
	bool bx12_first_higher = x12_first_higher.get_lb();

	// check if c3 covers the entire remaining region
	bool x12_contained = definitely(squared_distance(x12.p[bx12_first_higher ? 0 : 1], c3.center) <= c3.squared_radius);
	bool x15_contained = definitely(squared_distance(x15.p[1], c3.center) <= c3.squared_radius);
	bool x24_contained = definitely(squared_distance(x24.p[0], c3.center) <= c3.squared_radius);
	if(x12_contained && x15_contained && x24_contained) {
		return true;
	}

	if(x15_contained && x24_contained) {
		Intersection_points x31 = intersection(bottom_row.c1, c3);
		Intersection_points x32 = intersection(bottom_row.c2, c3);
		if(x31.definitely_intersecting && x32.definitely_intersecting && definitely(x31.p[0].x < x31.p[1].x) && definitely(x32.p[0].x < x32.p[1].x)) {
			double ub_pocket_width  = (x32.p[0].x - x31.p[1].x).get_ub();
			double ub_y_31 = x31.p[1].y.get_ub();
			double ub_y_32 = x32.p[0].y.get_ub();
			double ub_y = ub_y_31 > ub_y_32 ? ub_y_31 : ub_y_32;
			double ub_pocket_height = __dadd_ru(ub_y, -x12.p[bx12_first_higher ? 0 : 1].y.get_lb());
			return can_recurse(R6.get_lb(), vars.radii[5].get_ub(), ub_pocket_width, ub_pocket_height);
		}
	}

	return false;
}

static inline __device__ bool five_disk_full_cover_border_cover32_concrete(const Variables& vars, const Intermediate_values& vals, int t1, int t2, int t3, int r1, int r2) {
	double lb_htop = three_disks_maximize_height(vars.radii[t1], vars.radii[t2], vars.radii[t3], vars.la.get_ub());
	if(lb_htop <= 0.0) {
		return false;
	}

	double ub_hrem = __dadd_ru(1.0, -lb_htop);
	double lb_wright = two_disks_maximize_height(vars.radii[r1], vars.radii[r2], ub_hrem);
	if(lb_wright <= 0.0) {
		return false;
	}
	
	double ub_wrem = __dadd_ru(vars.la.get_ub(), -lb_wright);
	double lb_R_6 = __dmul_rd(vars.la.get_lb(), critical_ratio);
	for(int i = 0; i < 5; ++i) {
		lb_R_6 = __dadd_rd(lb_R_6, -vars.radii[i].get_ub());
	}

	// the area we can cover recursively
	return can_recurse(lb_R_6, vars.radii[5].get_ub(), ub_wrem, ub_hrem);
}

static inline __device__ bool five_disk_full_cover_border_cover32(const Variables& vars, const Intermediate_values& vals) {
	return five_disk_full_cover_border_cover32_concrete(vars, vals, 0, 1, 2, 3, 4) || five_disk_full_cover_border_cover32_concrete(vars, vals, 0, 1, 4, 2, 3);
}

__device__ bool circlecover::rectangle_size_bound::five_disk_full_cover(const Variables& vars, const Intermediate_values& vals) {
	return /*five_disk_full_cover_3v2_recursion_on_border2(vars,vals) ||*/
		five_disk_full_cover_3v2_recursion_center(vars, vals) ||
		five_disk_full_cover_border_cover32(vars, vals)
	;
}

