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

static __device__ bool six_disk_full_cover_4_plus_2(const Variables& vars, const Intermediate_values& vals) {
	IV la{vars.la.get_ub(), vars.la.get_ub()};
	Max_height_strip_2 bottom = two_disks_maximal_height_strip(la, vars.radii[0], vars.radii[1]);
	if(bottom.lb_height <= 0.0) {
		return false;
	}

	double ub_hrem = __dadd_ru(1.0, -bottom.lb_height);
	double ub_hrem_sq = __dmul_ru(ub_hrem, ub_hrem);

	IV w4{vars.radii[3].get_lb(), vars.radii[3].get_lb()}; 
	w4 -= IV{0.25,0.25}*ub_hrem_sq;
	IV w3{vars.radii[2].get_lb(), vars.radii[2].get_lb()};
	w3 -= IV{0.25,0.25}*ub_hrem_sq;
	if(possibly(w4 <= 0.0) || possibly(w3 <= 0.0)) {
		return false;
	}
	w4 = sqrt(w4);
	w3 = sqrt(w3);

	IV y = 1.0 - IV{0.5,0.5}*ub_hrem;
	Circle c4{{w4, y}, {vars.radii[3].get_lb(), vars.radii[3].get_lb()}};
	Circle c3{{la-w3, y}, {vars.radii[2].get_lb(), vars.radii[2].get_lb()}};

	Intersection_points x34 = intersection(c3,c4);
	if(!x34.definitely_intersecting) {
		return false;
	}

	Point upper_intersection_top, lower_intersection_top;
	if(x34.p[0].y.get_lb() < x34.p[1].y.get_lb()) {
		upper_intersection_top = x34.p[1];
		lower_intersection_top = x34.p[0];
	} else {
		upper_intersection_top = x34.p[0];
		lower_intersection_top = x34.p[1];
	}

	Intersection_points x12 = intersection(bottom.c1, bottom.c2);
	if(!x12.definitely_intersecting) {
		algcuda::trap();
	}

	Point upper_intersection_bottom;
	if(x12.p[0].y.get_lb() < x12.p[1].y.get_lb()) {
		upper_intersection_bottom = x12.p[1];
	} else {
		upper_intersection_bottom = x12.p[0];
	}

	Intersection_points x14 = intersection(bottom.c1, c4);
	Intersection_points x23 = intersection(bottom.c2, c3);
	if(!x14.definitely_intersecting || !x23.definitely_intersecting) {
		return false;
	}

	Point top_c{upper_intersection_top.x, 0.5*(1.0+upper_intersection_top.y)};
	IV top_d1 = squared_distance(top_c, Point{2.0*w4, IV{1.0,1.0}});
	IV top_d2 = squared_distance(top_c, Point{la-2.0*w3, IV{1.0,1.0}});
	IV top_d3 = squared_distance(top_c, upper_intersection_top);

	Point bottom_c = center(lower_intersection_top, upper_intersection_bottom);
	IV bottom_d1 = squared_distance(bottom_c, x14.p[1]);
	IV bottom_d2 = squared_distance(bottom_c, x23.p[0]);
	IV bottom_d3 = squared_distance(bottom_c, lower_intersection_top);
	IV bottom_d4 = squared_distance(bottom_c, upper_intersection_bottom);

	const double r5 = vars.radii[4].get_lb();
	const double r6 = vars.radii[5].get_lb();
	
	if(definitely(top_d1 <= r6) && definitely(top_d2 <= r6) && definitely(top_d3 <= r6)) {
		// we can use r6 for the top
		return definitely(bottom_d1 <= r5) && definitely(bottom_d2 <= r5) && definitely(bottom_d3 <= r5) && definitely(bottom_d4 <= r5);
	} else if(definitely(top_d1 <= r5) && definitely(top_d2 <= r5) && definitely(top_d3 <= r5)) {
		// we have to use r5 for the top, and r5 suffices
		return definitely(bottom_d1 <= r6) && definitely(bottom_d2 <= r6) && definitely(bottom_d3 <= r6) && definitely(bottom_d4 <= r6);
	} else {
		return false;
	}
}

static __device__ bool six_disk_full_cover_2v3_with_recursion_r6_center(const Variables& vars, const Intermediate_values& vals) {
	Max_height_strip_wc_recursion_2 bottom_row = two_disks_maximal_height_strip_wc_recursion(vars.radii[0], vars.radii[1], vals.R, vars.la.get_ub());
	if(bottom_row.lb_height <= 0.0) {
		return false;
	}

	double ub_hrem = __dadd_ru(1.0, -bottom_row.lb_height);
	double ub_hrem_sq = __dmul_ru(ub_hrem, ub_hrem);
	double lb_w3 = __dadd_rd(__dmul_rd(4.0, vars.radii[2].get_lb()), -ub_hrem_sq);
	double lb_w4 = __dadd_rd(__dmul_rd(4.0, vars.radii[3].get_lb()), -ub_hrem_sq);
	if(lb_w3 <= 0 || lb_w4 <= 0) {
		return false;
	}
	lb_w3 = __dsqrt_rd(lb_w3);
	lb_w4 = __dsqrt_rd(lb_w4);
	double lb_w34 = __dadd_rd(lb_w3, lb_w4);
	double ub_wrem = __dadd_ru(vars.la.get_ub(), -lb_w34);
	if(ub_wrem <= 0.0) {
		return true;
	}
	double ub_wrem_sq = __dmul_ru(ub_wrem, ub_wrem);

	const double lb_r5 = vars.radii[4].get_lb();
	double lb_dy5 = __dadd_rd(lb_r5, -__dmul_ru(0.25, ub_wrem_sq));
	if(lb_dy5 <= 0.0) {
		return false;
	}
	lb_dy5 = __dsqrt_rd(lb_dy5);
	if(__dadd_rd(__dmul_rd(2.0, lb_dy5), bottom_row.lb_height) >= 1.0) {
		return true;
	}

	IV y34 = 1.0 - IV{0.5,0.5}*ub_hrem;
	Point c3{vars.la - IV{0.5,0.5}*lb_w3, y34};
	Point c4{IV{0.5,0.5}*lb_w4, y34};
	Point c5{lb_w4 + IV{0.5,0.5}*ub_wrem, 1.0 - IV{lb_dy5,lb_dy5}};
	Circle ci5{c5, {lb_r5,lb_r5}};

	Intersection_points x15 = intersection(bottom_row.c1, ci5), x25 = intersection(bottom_row.c2, ci5);
	if(!x15.definitely_intersecting || !x25.definitely_intersecting) {
		return false;
	}

	Point p15 = x15.p[1];
	Point p25 = x25.p[0];

	if(definitely(p15.y <= bottom_row.lb_height) && definitely(p25.y <= bottom_row.lb_height)) {
		return true;
	}

	double ub_wrem6 = __dadd_ru(p25.x.get_ub(), -p15.x.get_lb());
	double ub_ymax = p15.y.get_ub() < p25.y.get_ub() ? p25.y.get_ub() : p15.y.get_ub();
	double ub_hrem6 = __dadd_ru(ub_ymax, -bottom_row.lb_height);
	double ub_wrem6_sq = __dmul_ru(ub_wrem6, ub_wrem6);
	double ub_hrem6_sq = __dmul_ru(ub_hrem6, ub_hrem6);

	return __dadd_ru(ub_wrem6_sq, ub_hrem6_sq) <= __dmul_rd(4.0, vars.radii[5].get_lb());
}

static inline __device__ bool six_disk_full_cover_r12_r456_r3_recursion(const Variables& vars, const Intermediate_values& vals) {
	double lb_w12 = two_disks_maximize_height(vars.radii[0], vars.radii[1], 1.0);
	if(lb_w12 <= 0.0) {
		return false;
	}

	double lb_w456 = three_disks_maximize_height(vars.radii[3], vars.radii[4], vars.radii[5], 1.0);
	if(lb_w456 <= 0.0) {
		return false;
	}

	double wrem = __dadd_ru(vars.la.get_ub(), -__dadd_rd(lb_w12, lb_w456));
	if(wrem <= 0.0) {
		return true;
	}

	double lb_h3 = __dadd_rd(__dmul_rd(4.0, vars.radii[2].get_lb()), -__dmul_ru(wrem,wrem));
	if(lb_h3 <= 0.0) {
		return false;
	}
	lb_h3 = __dsqrt_rd(lb_h3);

	double hrem = __dadd_ru(1.0, -lb_h3);
	return can_recurse(vals.R.get_lb(), vars.radii[5].get_ub(), wrem, hrem);
}

__device__ bool circlecover::rectangle_size_bound::six_disk_full_cover(const Variables& vars, const Intermediate_values& vals) {
	return six_disk_full_cover_r12_r456_r3_recursion(vars, vals) ||
		six_disk_full_cover_4_plus_2(vars, vals) ||
		six_disk_full_cover_2v3_with_recursion_r6_center(vars, vals);
}

