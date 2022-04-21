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
#include "../operations.cuh"
#include "../tight_rectangle/strategies.cuh"
#include <cfloat>
#include <algcuda/exit.cuh>

// the values from the worst-case rectangle covering without size bound
static const double C      = 195.0/256.0;
static const double tolerated_lambda = 2.0898841580413813901;

// place r7 in the corner of the strip and recurse using our worst-case results
using namespace circlecover;
using namespace circlecover::rectangle_size_bound;

__device__ static inline bool r7_in_corner_and_recurse(IV R, IV r7, double ub_w) {
	double lb_R_no_7 = __dadd_rd(R.get_lb(), -r7.get_ub());
	double lb_h7 = __dadd_rd(__dmul_rd(4.0, r7.get_lb()), -__dmul_ru(ub_w,ub_w));
	if(lb_h7 < 0) {
		return false;
	}
	lb_h7 = __dsqrt_rd(lb_h7);
	double rem_h = __dadd_ru(1.0, -lb_h7);
	return can_recurse(lb_R_no_7, r7.get_ub(), ub_w, rem_h);
}

__device__ static inline bool three_disk_top_strip_can_cover(double la, double t1, double t2, double t3, const double* remaining_4, double ub_max_disk, double lb_R_no_7) {
	double lb_height_top = three_disks_maximize_height(IV(t1,t1), IV(t2,t2), IV(t3,t3), la); 
	if(lb_height_top <= 0) {
		return false;
	}

	double ub_hrem = __dadd_rd(1.0, -lb_height_top);
	if(ub_hrem <= 0) {
		return true;
	}
	double ub_hrem_sq = __dmul_ru(ub_hrem, ub_hrem);
	double ub_wrem;

	// if we make the recursive rectangle as big as we can for size-bounded recursion, does ub_max_disk fit?
	double lb_rec_area_sb = __ddiv_rd(lb_R_no_7, critical_ratio);
	double lb_rec_width_sb = __ddiv_rd(lb_rec_area_sb, ub_hrem);
	double lb_rec_min_sb = lb_rec_width_sb < ub_hrem ? lb_rec_width_sb : ub_hrem;
	if(disk_satisfies_size_bound(ub_max_disk, lb_rec_min_sb)) {
		// we can use size-bounded recursion
		ub_wrem = __dadd_ru(la, -lb_rec_width_sb);
	} else {
		// otherwise, we have to use worst-case recursion
		double lb_rec_area_wc = __ddiv_rd(lb_R_no_7, C);
		double lb_rec_width_wc = __ddiv_rd(lb_rec_area_wc, ub_hrem);
		double lb_rec_long_wc  = lb_rec_width_wc < ub_hrem ? ub_hrem : lb_rec_width_wc;
		double lb_rec_short_wc = lb_rec_width_wc < ub_hrem ? lb_rec_width_wc : ub_hrem;
		double ub_skew = __ddiv_ru(lb_rec_long_wc, lb_rec_short_wc);
		
		// if we are below la_bar skew, use C as critical ratio; otherwise, compute coverable width
		if(ub_skew > tolerated_lambda) {
			if(lb_rec_width_wc > ub_hrem) {
				lb_rec_width_wc = __dmul_rd(ub_hrem, tolerated_lambda);
			} else {
				lb_rec_width_wc = __dadd_rd(__dmul_rd(2.0, lb_R_no_7), -__dmul_ru(0.5, ub_hrem_sq));
				if(lb_rec_width_wc <= 0) {
					lb_rec_width_wc = 0.0;
				} else {
					lb_rec_width_wc = __dsqrt_rd(lb_rec_width_wc);
				}
			}
		}

		ub_wrem = __dadd_ru(la, -lb_rec_width_wc);
	}

	IV b1{remaining_4[1],remaining_4[1]}, b2{remaining_4[2],remaining_4[2]};
	double lb_bot_height = two_disks_maximize_height(b1, b2, ub_wrem);
	if(lb_bot_height <= 0) {
		return false;
	} else if(lb_bot_height >= ub_hrem) {
		return true;
	}

	// the remaining height to be covered by the last two disks
	double ub_hrem_l2 = __dadd_ru(ub_hrem, -lb_bot_height);
	IV l1{remaining_4[0], remaining_4[0]}, l2{remaining_4[3], remaining_4[3]};
	double lb_h_l2 = two_disks_maximize_height(l1,l2,ub_wrem);
	if(lb_h_l2 <= 0.0) {
		return false;
	} else if(lb_h_l2 >= ub_hrem_l2) {
		return true;
	}
	return false;
}

__device__ static inline bool restrict_r7_t_shaped_recursion_split_recursion_largest_r7_works(const Variables& vars, const Intermediate_values& vals, double ub_r7,
		double weight_rem, double wfirst, double hfirst, double wsecond, double hsecond)
{
	double cr_first = bound_critical_ratio(weight_rem, ub_r7, wfirst, hfirst);
	if(cr_first >= DBL_MAX*DBL_MAX) {
		return false;
	}

	double weight_needed = __dmul_ru(cr_first, __dmul_ru(wfirst, hfirst));
	double weight_needed_after_split = __dadd_ru(weight_needed, ub_r7);
	double wrem_for_second = __dadd_rd(weight_rem, -weight_needed_after_split);
	return can_recurse(wrem_for_second, ub_r7, wsecond, hsecond);
}

__device__ static inline void restrict_r7_t_shaped_recursion_split_recursion_largest_r7(const Variables& vars, const Intermediate_values& vals, IV& r7, double weight_rem, double wfirst, double hfirst, double wsecond, double hsecond) {
	// first try the actual upper bound of r7
	if(restrict_r7_t_shaped_recursion_split_recursion_largest_r7_works(vars, vals, r7.get_ub(), weight_rem, wfirst, hfirst, wsecond, hsecond)) {
		// if that works, make the r7 interval empty
		r7.set_lb(DBL_MAX*DBL_MAX);
		return;
	}

	double lb_r7_lb = r7.get_lb();
	double lb_r7_ub = r7.get_ub();
	for(;;) {
		double mid = 0.5 * (lb_r7_lb + lb_r7_ub);
		if(mid <= lb_r7_lb || mid >= lb_r7_ub) {
			r7.tighten_lb(lb_r7_lb);
			break;
		}

		if(restrict_r7_t_shaped_recursion_split_recursion_largest_r7_works(vars, vals, mid, weight_rem, wfirst, hfirst, wsecond, hsecond)) {
			lb_r7_lb = mid;
		} else {
			lb_r7_ub = mid;
		}
	}
}

__device__ static inline void restrict_r7_t_shaped_recursion_split_recursion(
		const Variables& vars, const Intermediate_values& vals, IV& r7,
		double weight_rem, int largest_disk_rem, double wfirst, double hfirst, double wsecond, double hsecond)
{
	if(largest_disk_rem > 5) {
		restrict_r7_t_shaped_recursion_split_recursion_largest_r7(vars, vals, r7, weight_rem, wfirst, hfirst, wsecond, hsecond);
		return;
	}

	double ub_largest_disk = vars.radii[largest_disk_rem].get_ub();
	double cr_first = bound_critical_ratio(weight_rem, ub_largest_disk, wfirst, hfirst);
	if(cr_first < DBL_MAX*DBL_MAX) {
		double weight_needed = __dmul_ru(cr_first, __dmul_ru(wfirst, hfirst));

		double weight_in_disks = 0.0;
		for(int i = largest_disk_rem; i <= 5; ++i) {
			weight_in_disks = __dadd_ru(weight_in_disks, vars.radii[i].get_ub());
		}

		if(weight_in_disks > weight_needed) {
			double lb_weight_in_disks = 0.0;
			for(int i = largest_disk_rem; i <= 5; ++i) {
				lb_weight_in_disks = __dadd_rd(lb_weight_in_disks, vars.radii[i].get_lb());
			}

			if(lb_weight_in_disks >= weight_needed) {
				double weight_rem_second = __dadd_rd(weight_rem, -weight_in_disks);
				r7.tighten_lb(recursion_bound_size(weight_rem_second, wsecond, hsecond));
			}

			return;
		}

		// first, try with r7's upper bound
		double split_cost = r7.get_ub();
		double weight_needed_after_split = __dadd_ru(weight_needed, split_cost);
		double weight_rem_second = __dadd_rd(weight_rem, -weight_needed_after_split);
		double rec_weight_bound = recursion_bound_size(weight_rem_second, wsecond, hsecond);
		r7.tighten_lb(rec_weight_bound);

		if(!r7.empty()) {
			// try to find a better lower bound
			double ub_r7_lb = r7.get_ub();
			double lb_r7_lb = r7.get_lb();

			for(;;) {
				double mid = 0.5 * (lb_r7_lb + ub_r7_lb);
				if(mid <= lb_r7_lb || mid >= ub_r7_lb) {
					r7.set_lb(lb_r7_lb);
					return;
				}

				double weight_needed_after_split = __dadd_ru(weight_needed, mid);
				double weight_rem_second = __dadd_rd(weight_rem, -weight_needed_after_split);
				double rec_weight_bound = recursion_bound_size(weight_rem_second, wsecond, hsecond);
				
				if(rec_weight_bound < mid) {
					// this value is too large
					ub_r7_lb = mid;
				} else {
					// this value of r7 works
					lb_r7_lb = mid;
				}
			}
		}
	}
}

namespace {
	struct r6_position {
		Point p6;
		double lb_left_intersection;
	};
}

// compute the position of r6 as defined by the two points x1,x2 and the distance up to which the strip is covered;
// this distance is set to 0 if the placement does not work
__device__ static inline r6_position restrict_r7_t_shaped_recursion_compute_r6(IV r6, const Point& x1, const Point& x2, IV lower_border, IV upper_border) {
	double left_coord = x1.x.get_lb() < x2.x.get_lb() ? x1.x.get_lb() : x2.x.get_lb();
	double top_coord  = x1.y.get_ub() < x2.y.get_ub() ? x2.y.get_ub() : x1.y.get_ub();
	double bot_coord  = x1.y.get_lb() < x2.y.get_lb() ? x1.y.get_lb() : x2.y.get_lb();
	double ub_height  = __dadd_ru(top_coord, -bot_coord);
	double lb_dx6 = __dadd_rd(r6.get_lb(), -__dmul_ru(0.25, __dmul_ru(ub_height,ub_height)));
	if(lb_dx6 <= 0.0) {
		r6_position pos;
		pos.lb_left_intersection = 0.0;
		return pos;
	}
	lb_dx6 = __dsqrt_rd(lb_dx6);

	r6_position pos;
	pos.lb_left_intersection = 0.0;
	pos.p6 = Point{left_coord + IV{lb_dx6, lb_dx6}, IV{__dmul_rd(0.5, __dadd_rd(bot_coord, top_coord)), __dmul_ru(0.5, __dadd_ru(bot_coord, top_coord))}};
	double dy_bot = (pos.p6.y - lower_border).get_ub();
	double dy_top = (upper_border - pos.p6.y).get_ub();

	dy_bot = __dmul_ru(dy_bot, dy_bot);
	dy_top = __dmul_ru(dy_top, dy_top);
	double max_dy_sq = dy_bot < dy_top ? dy_top : dy_bot;
	double r6_strip_width = __dadd_rd(r6.get_lb(), -max_dy_sq);
	if(r6_strip_width <= 0.0) {
		return pos;
	}
	r6_strip_width = __dsqrt_rd(r6_strip_width);	
	pos.lb_left_intersection = __dadd_rd(pos.p6.x.get_lb(), r6_strip_width);
	return pos;
}

__device__ static inline void restrict_r7_t_shaped_recursion_r5_large(const Variables& vars, const Intermediate_values& vals, IV& r7, const Circle& c1, const Circle& c3, const Circle& c5, double lb_R6, double w12, double wrem, double hrem, double strip_w5, IV lower_strip_border, IV upper_strip_border) {
	Intersection_points x15 = intersection(c1, c5), x35 = intersection(c3, c5);
	if(!x15.definitely_intersecting || !x35.definitely_intersecting) {
		algcuda::trap();
	}

	double lb_left_intersection = x15.p[1].x.get_lb();
	if(lb_left_intersection > x35.p[1].x.get_lb()) {
		lb_left_intersection = x35.p[1].x.get_lb();
	}

	// try direct recursion starting from disk r6
	double whoriz_rem = __dadd_ru(w12, -lb_left_intersection);
	if(whoriz_rem <= 0.0) {
		if(can_recurse(lb_R6, vars.radii[5].get_ub(), wrem, 1.0)) {
			r7.tighten_lb(DBL_MAX*DBL_MAX);
		}
		return;
	}
	restrict_r7_t_shaped_recursion_split_recursion(vars, vals, r7, lb_R6, 5, whoriz_rem, hrem, wrem, 1.0);
	restrict_r7_t_shaped_recursion_split_recursion(vars, vals, r7, lb_R6, 5, wrem, 1.0, whoriz_rem, hrem);
	if(r7.empty()) {
		return;
	}

	// otherwise, consider explicit placement of r6 as well
	double lb_R7 = __dadd_rd(lb_R6, -vars.radii[5].get_ub());
	if(lb_R7 <= 0.0) {
		return;
	}

	
	// if the intersection points are possibly out of the strip, replace them by the intersection points with the strip
	Point x1 = x15.p[1];
	Point x2 = x35.p[1];
	if(possibly(x1.y >= upper_strip_border) || possibly(x1.y <= lower_strip_border) || possibly(x2.y >= upper_strip_border) || possibly(x2.y <= lower_strip_border)) {
		x1 = Point{{strip_w5,strip_w5},upper_strip_border};
		x2 = Point{{strip_w5,strip_w5},lower_strip_border};
	}

	r6_position r6pos = restrict_r7_t_shaped_recursion_compute_r6(vars.radii[5], x1, x2, lower_strip_border, upper_strip_border);
	if(r6pos.lb_left_intersection <= 0.0) {
		return;
	}

	if(r6pos.lb_left_intersection >= w12) {
		r7.tighten_lb(recursion_bound_size(lb_R7, wrem, 1.0));
	} else {
		whoriz_rem = __dadd_ru(w12, -lb_left_intersection);
		restrict_r7_t_shaped_recursion_split_recursion(vars, vals, r7, lb_R7, 6, whoriz_rem, hrem, wrem, 1.0);
		restrict_r7_t_shaped_recursion_split_recursion(vars, vals, r7, lb_R7, 6, wrem, 1.0, whoriz_rem, hrem);
	}
}

__device__ static void restrict_r7_t_shaped_recursion(const Variables& vars, const Intermediate_values& vals, IV& r7) {
	double w2 = __dsqrt_rd(__dmul_rd(2.0, vars.radii[1].get_lb()));
	double h12 = w2;
	double ub_h12_sq = __dmul_ru(h12,h12);
	double w1 = __dsqrt_rd(__dadd_rd(__dmul_rd(4.0, vars.radii[0].get_lb()), -ub_h12_sq));
	double w12 = __dadd_rd(w1, w2);
	double h34 = two_disks_maximize_height(vars.radii[2], vars.radii[3], w12);
	if(h34 <= 0.0) {
		return;
	}

	// compute R5
	double wtot = __dmul_rd(vars.la.get_lb(), critical_ratio);
	double lb_R5 = __dadd_rd(wtot, -__dadd_ru(vars.radii[0].get_ub(), __dadd_ru(vars.radii[1].get_ub(), __dadd_ru(vars.radii[2].get_ub(), vars.radii[3].get_ub()))));

	double hrem = __dadd_ru(1.0, -__dadd_rd(h12, h34));
	double wrem = __dadd_ru(vars.la.get_ub(), -w12);

	// handle degenerate cases
	if(hrem <= 0.0) {
		if(wrem <= 0.0 || can_recurse(lb_R5, vars.radii[4].get_ub(), wrem, 1.0)) {
			r7.tighten_lb(DBL_MAX*DBL_MAX);
		}

		return;
	}
	if(wrem <= 0.0) {
		if(can_recurse(lb_R5, vars.radii[4].get_ub(), vars.la.get_ub(), hrem)) {
				r7.tighten_lb(DBL_MAX*DBL_MAX);
		}

		return;
	}

	// try immediate recursion starting with r5
	restrict_r7_t_shaped_recursion_split_recursion(vars, vals, r7, lb_R5, 4, w12, hrem, wrem, 1.0);
	restrict_r7_t_shaped_recursion_split_recursion(vars, vals, r7, lb_R5, 4, wrem, 1.0, w12, hrem);
	if(r7.empty()) {
		return;
	}

	// consider placing 5 disks explicitly
	double lb_R6 = __dadd_rd(lb_R5, -vars.radii[4].get_ub());
	if(lb_R6 <= 0.0) {
		return;
	}

	double ub_hrem_sq = __dmul_ru(hrem, hrem);
	double lb_x5 = __dadd_rd(vars.radii[4].get_lb(), -__dmul_ru(0.25, ub_hrem_sq));
	if(lb_x5 <= 0.0) {
		// we cannot place r5 properly
		return;
	}
	lb_x5 = __dsqrt_rd(lb_x5);
	double strip_w5 = __dmul_rd(2.0, lb_x5);

	// compute interesting points and disk placements
	IV y5{__dadd_rd(h12, __dmul_rd(0.5, hrem)), __dadd_ru(h12, __dmul_ru(0.5, hrem))};
	Point p5{{lb_x5,lb_x5}, y5};
	IV p2_x{__dmul_rd(0.5, w2), __dmul_ru(0.5, w2)};
	IV p2_y{__dmul_rd(0.5, h12), __dmul_ru(0.5,h12)};
	Point p2{p2_x,p2_y};
	Point p1{w2 + IV(0.5,0.5)*w1, p2_y}; 
	IV x4 = sqrt(vars.radii[3] - 0.25*IV(h34,h34).square());
	IV y4 = 1.0 - 0.5 * IV(h34,h34);
	Point p4{x4,y4};
	IV w3 = w12 - 2.0*p4.x;
	Point p3{w12 - 0.5*w3, y4};
	Circle c1{p1, {vars.radii[0].get_lb(), vars.radii[0].get_lb()}};
	Circle c2{p2, {vars.radii[1].get_lb(), vars.radii[1].get_lb()}};
	Circle c3{p3, {vars.radii[2].get_lb(), vars.radii[2].get_lb()}};
	Circle c4{p4, {vars.radii[3].get_lb(), vars.radii[3].get_lb()}};
	Circle c5{p5, {vars.radii[4].get_lb(), vars.radii[4].get_lb()}};
	Point p43{2.0*p4.x, IV(1.0,1.0) - h34};
	Point p21{{w2,w2}, {h12,h12}};

	// check for (and handle) the case of large r5
	IV lower_strip_border{h12,h12};
	IV upper_strip_border = IV(1.0,1.0) - h34;
	if(definitely(squared_distance(p5, p43) <= vars.radii[4]) && definitely(squared_distance(p5, p21) <= vars.radii[4])) {
		restrict_r7_t_shaped_recursion_r5_large(vars, vals, r7, c1, c3, c5, lb_R6, w12, wrem, hrem, strip_w5, lower_strip_border, upper_strip_border);
		return;
	}

	// compute intersection points
	Intersection_points x24 = intersection(c2, c4);
	Intersection_points x25 = intersection(c2, c5);
	Intersection_points x45 = intersection(c4, c5);

	// no intersection, this would be an error
	if(!x25.definitely_intersecting || !x45.definitely_intersecting) {
		algcuda::trap();
	}

	// this can only happen if r5 covers the remaining left boundary just barely
	if(x25.p[0].x.get_ub() >= x25.p[1].x.get_lb() || x45.p[0].x.get_ub() >= x45.p[1].x.get_lb()) {
		return;
	}

	// the two right intersections of r5 and r2,r4 determine what part of the remaining strip is covered
	Point x1 = x25.p[1];
	Point x2 = x45.p[1];
	
	// if the intersection points are possibly out of the strip, replace them by the intersection points with the strip
	if(possibly(x1.y >= upper_strip_border) || possibly(x1.y <= lower_strip_border) || possibly(x2.y >= upper_strip_border) || possibly(x2.y <= lower_strip_border)) {
		x1 = Point{{strip_w5,strip_w5},upper_strip_border};
		x2 = Point{{strip_w5,strip_w5},lower_strip_border};
	}

	// compute the x coordinate up to which everything is covered
	double lb_left_intersection = x1.x.get_lb();
	if(lb_left_intersection > x2.x.get_lb()) {
		lb_left_intersection = x2.x.get_lb();
	}

	// everything left of lb_left_intersection is covered; try directly recursing
	double whoriz_rem = __dadd_ru(w12, -lb_left_intersection);
	restrict_r7_t_shaped_recursion_split_recursion(vars, vals, r7, lb_R6, 5, whoriz_rem, hrem, wrem, 1.0);
	restrict_r7_t_shaped_recursion_split_recursion(vars, vals, r7, lb_R6, 5, wrem, 1.0, whoriz_rem, hrem);
	if(r7.empty()) {
		return;
	}

	// otherwise, consider explicit placement of r6 as well
	double lb_R7 = __dadd_rd(lb_R6, -vars.radii[5].get_ub());
	if(lb_R7 <= 0.0) {
		return;
	}

	r6_position r6pos = restrict_r7_t_shaped_recursion_compute_r6(vars.radii[5], x1,x2, lower_strip_border, upper_strip_border);
	if(r6pos.lb_left_intersection <= 0.0) {
		return;
	}
	Circle c6{r6pos.p6, {vars.radii[5].get_lb(), vars.radii[5].get_lb()}};
	Point d71, d72;

	Intersection_points x16 = intersection(c1, c6), x36 = intersection(c3, c6);
	if(!x16.definitely_intersecting || !x36.definitely_intersecting) {
		algcuda::trap();
	}

	bool r6_right_is_intersection = false;
	if(definitely(squared_distance(p43, r6pos.p6) <= vars.radii[5]) && definitely(squared_distance(p21, r6pos.p6) <= vars.radii[5])) {
		// the right boundary of the covered area is determined by the right intersection of c6 with c3 and c1
		lb_left_intersection = x16.p[1].x.get_lb() < x36.p[1].x.get_lb() ? x16.p[1].x.get_lb() : x36.p[1].x.get_lb();
		d71 = x36.p[1];
		d72 = x16.p[1];
		r6_right_is_intersection = true;
	} else {
		// the right boundary of the covered area is determined by the intersection of c6 with the strip boundary
		lb_left_intersection = r6pos.lb_left_intersection;
		d71 = Point{upper_strip_border, {lb_left_intersection,lb_left_intersection}};
		d72 = Point{lower_strip_border, {lb_left_intersection,lb_left_intersection}};
	}

	// finally, try recursing starting with r7
	whoriz_rem = __dadd_ru(w12, -lb_left_intersection);
	if(whoriz_rem <= 0.0) {
		r7.tighten_lb(recursion_bound_size(lb_R7, wrem, 1.0));
	} else {
		restrict_r7_t_shaped_recursion_split_recursion(vars, vals, r7, lb_R7, 6, whoriz_rem, hrem, wrem, 1.0);
		restrict_r7_t_shaped_recursion_split_recursion(vars, vals, r7, lb_R7, 6, wrem, 1.0, whoriz_rem, hrem);
	}
	if(r7.empty()) {
		return;
	}

	double lb_R8 = __dadd_rd(lb_R7, -r7.get_ub());
	if(lb_R8 <= 0.0) {
		return;
	}

	// now we hopefully have r7 large enough so we can use it on our strip
	Intersection_points x13 = intersection(c1,c3);
	if(r6_right_is_intersection && x13.definitely_intersecting && definitely(squared_distance(x13.p[0], r6pos.p6) <= vars.radii[5]) && definitely(x13.p[1].y >= lower_strip_border) && definitely(x13.p[1].y <= upper_strip_border)) {
		// everything left of the right intersection of c1 and c3 is covered
		d71 = x13.p[1];
		d72 = x13.p[1];
	}

	r6_position r7pos = restrict_r7_t_shaped_recursion_compute_r6(r7, d71, d72, lower_strip_border, upper_strip_border);
	if(r7pos.lb_left_intersection <= 0.0) {
		return;
	}

	if(r7pos.lb_left_intersection >= vars.la.get_ub()) {
		// r7 intersects the right rectangle border
		restrict_r7_t_shaped_recursion_split_recursion_largest_r7(vars, vals, r7, lb_R8, wrem, h12, wrem, h34);
		restrict_r7_t_shaped_recursion_split_recursion_largest_r7(vars, vals, r7, lb_R8, wrem, h34, wrem, h12);
	}
	
	if(r7pos.lb_left_intersection >= w12) {
		// the entire horizontal strip is covered
		r7.tighten_lb(recursion_bound_size(lb_R8, wrem, 1.0));
	} else {
		// some part of the horizontal strip remains
		whoriz_rem = __dadd_ru(w12, -r7pos.lb_left_intersection);
		restrict_r7_t_shaped_recursion_split_recursion_largest_r7(vars, vals, r7, lb_R8, whoriz_rem, hrem, wrem, 1.0);
		restrict_r7_t_shaped_recursion_split_recursion_largest_r7(vars, vals, r7, lb_R8, wrem, 1.0, whoriz_rem, hrem);
	}
}

__device__ static bool three_disk_top_strip(const Variables& vars, IV R, IV r7) {
	const double lb_R_no_7 = __dadd_rd(R.get_lb(), -r7.get_ub());
	const double lb_disks[7] = {
		vars.radii[0].get_lb(), vars.radii[1].get_lb(),
		vars.radii[2].get_lb(), vars.radii[3].get_lb(),
		vars.radii[4].get_lb(), vars.radii[5].get_lb(),
		r7.get_lb()
	};

	for(int i = 0; i < 5; ++i) {
		double t1 = lb_disks[i];
		for(int j = i+1; j < 6; ++j) {
			double t2 = lb_disks[j];
			for(int k = j+1; k < 7; ++k) {
				double t3 = lb_disks[k];
				double rem_disks[4];
				int disk_mask = 0x7f;
				disk_mask &= ~(1 << i);
				disk_mask &= ~(1 << j);
				disk_mask &= ~(1 << k);
				for(int g = 0, r = 0; g < 7; ++g) {
					if(disk_mask & (1 << g)) {
						rem_disks[r++] = lb_disks[g];
					}
				}

				if(three_disk_top_strip_can_cover(vars.la.get_ub(), t1, t2, t3, rem_disks, r7.get_ub(), lb_R_no_7)) {
					return true;
				}
			}
		}
	}

	return false;
}

__device__ static bool r435_strip_covering_r12_gap_r67_right(const Variables& vars, IV r7, const Intermediate_values& vals) {
	double lb_covered_435 = three_disks_maximize_height(vars.radii[2], vars.radii[3], vars.radii[4], 1.0);
	if(lb_covered_435 <= 0.0) {
		return false;
	}

	IV w435_sq{__dmul_rd(lb_covered_435,lb_covered_435), __dmul_ru(lb_covered_435,lb_covered_435)};
	
	// assume r3 has radius at its lower bound; otherwise, just ignore the additional radius
	IV r3{vars.radii[2].get_lb(), vars.radii[2].get_lb()};
	IV r4{vars.radii[3].get_lb(), vars.radii[3].get_lb()};

	IV h3 = 4.0*r3 - w435_sq;
	IV h4 = 4.0*r4 - w435_sq;
	if(possibly(h3 <= 0.0) || possibly(h4 <= 0.0)) {
		return false;
	}
	h3 = sqrt(h3);
	h4 = sqrt(h4);

	Point pos3{
		{__dmul_rd(0.5, lb_covered_435), __dmul_ru(0.5, lb_covered_435)},
		0.5*h3 + h4
	};
	
	// compute the maximum width we can use for r1, r2 such that their intersection lies within r3
	IV r1{vars.radii[0].get_lb(), vars.radii[0].get_lb()};
	IV r2{vars.radii[1].get_lb(), vars.radii[1].get_lb()};
	IV r1sqrt = sqrt(r1);
	IV r2sqrt = sqrt(r2);

	// if the two disks are not definitely enough to cover the entire height, fail.
	if(possibly(r1sqrt + r2sqrt <= 0.5)) {
		return false;
	}
	double lb = 0.0;

	// check if we can possibly cover the entire remaining square
	double ub = __dmul_rd(2.0, r2sqrt.get_lb());
	double ub_w_remaining = __dadd_ru(vars.la.get_ub(), -lb_covered_435);
	IV wrem_sq{__dmul_rd(ub_w_remaining,ub_w_remaining), __dmul_ru(ub_w_remaining,ub_w_remaining)};
	if(ub_w_remaining < ub) {
		IV pos1_y = r1 - 0.25*wrem_sq;
		IV pos2_y = r2 - 0.25*wrem_sq;
		if(definitely(pos1_y > 0.0) && definitely(pos2_y > 0.0)) {
			pos1_y = 1.0 - sqrt(pos1_y);
			pos2_y = sqrt(pos2_y);

			Point pos1{0.5*IV{ub_w_remaining, ub_w_remaining} + lb_covered_435, pos1_y};
			Point pos2{pos1.x, pos2_y};
			Circle c1{pos1, r1};
			Circle c2{pos2, r2};

			// make sure there definitely are two intersection points and the left one definitely lies in or on r3
			Intersection_points x12 = intersection(c1,c2);
			if(x12.definitely_intersecting && x12.p[0].x.get_ub() < x12.p[1].x.get_lb() && definitely(squared_distance(pos3, x12.p[0]) <= r3)) {
				// except for a small part in the middle, we can cover the entire rectangle
				double lb_h1 = __dmul_rd(2.0, __dsqrt_rd(__dadd_rd(r1.get_lb(), -__dmul_ru(0.25, wrem_sq.get_ub()))));
				double lb_h2 = __dmul_rd(2.0, __dsqrt_rd(__dadd_rd(r2.get_lb(), -__dmul_ru(0.25, wrem_sq.get_ub()))));
				double rem_height = __dadd_ru(1.0, -__dadd_rd(lb_h1,lb_h2));
				double rem_width  = __dadd_ru(vars.la.get_ub(), x12.p[1].x.get_lb());

				// can we cover the remaining area with the remaining disks?
				if(can_recurse(__dadd_rd(vals.R.get_lb(), vars.radii[5].get_lb()), vars.radii[5].get_ub(), rem_width, rem_height)) {
					return true;
				}
			}
		}

		
		// otherwise, covering the entire remaining area does not work
		ub = ub_w_remaining;
	}

	double w12;
	Intersection_points best_x12;
	best_x12.definitely_intersecting = false;

	for(;;) {
		double mid = 0.5*(lb+ub);
		if(mid <= lb || mid >= ub) {
			w12 = lb;
			break;
		}

		IV wsq{__dmul_rd(mid,mid), __dmul_ru(mid,mid)};
		IV x = 0.5*IV{mid,mid} + lb_covered_435;
		IV hhalf2 = r2 - 0.25*wsq;
		if(hhalf2.get_lb() <= 0) {
			ub = mid;
			continue;
		}
		hhalf2 = sqrt(hhalf2);
		IV hhalf1 = sqrt(r1 - 0.25*wsq);

		Point pos1{x, 1.0-hhalf1};
		Point pos2{x, hhalf2};
		Circle c1{pos1, r1};
		Circle c2{pos2, r2};

		Intersection_points x12 = intersection(c1,c2);
		if(x12.definitely_intersecting && x12.p[0].x.get_ub() < x12.p[1].x.get_lb() && definitely(squared_distance(pos3, x12.p[0]) <= r3)) {
			lb = mid;
			best_x12 = x12;
		} else {
			ub = mid;
		}
	}

	// we did not find a working width at all
	if(!best_x12.definitely_intersecting) {
		return false;
	}

	double ub_w67 = __dadd_ru(vars.la.get_ub(), -__dadd_rd(lb_covered_435, w12));
	double ub_w67sq = __dmul_ru(ub_w67, ub_w67);

	{ // simple recursive variant (single recursion): place r7 at the top of the strip
		double lb_h7 = __dadd_rd(__dmul_rd(4.0, r7.get_lb()), -ub_w67sq);
		double lb_h6 = __dadd_rd(__dmul_rd(4.0, vars.radii[5].get_lb()), -ub_w67sq);
		if(lb_h7 > 0 && lb_h6 > 0) {
			lb_h7 = __dsqrt_rd(lb_h7);
			lb_h6 = __dsqrt_rd(lb_h6);
			
			Point pos6{
				vars.la - IV{__dmul_rd(0.5, ub_w67), __dmul_ru(0.5, ub_w67)},
				1.0 - IV{lb_h7, lb_h7} - 0.5*IV{lb_h6,lb_h6}
			};

			double ub_height_rem = __dadd_ru(1.0, -__dadd_rd(lb_h7, lb_h6));
			if(definitely(squared_distance(best_x12.p[1], pos6) <= vars.radii[5].get_lb())) {
				if(can_recurse(__dadd_rd(vals.R.get_lb(), -r7.get_ub()), r7.get_ub(), ub_w67, ub_height_rem)) {
					return true;
				}
			}
		}
	}

	{ // variant for very large r7: cover wider strip (corresponding to right intersection point of r1 and r2)
		double ub_w67_wide = __dadd_ru(vars.la.get_ub(), -best_x12.p[1].x.get_lb());
		double ub_w67_widesq = __dmul_ru(ub_w67_wide, ub_w67_wide);
		double lb_h7 = __dadd_rd(__dmul_rd(4.0, r7.get_lb()), -ub_w67_widesq);
		double lb_h6 = __dadd_rd(__dmul_rd(4.0, vars.radii[5].get_lb()), -ub_w67_widesq);
		if(lb_h7 > 0 && lb_h6 > 0) {
			lb_h7 = __dsqrt_rd(lb_h7);
			lb_h6 = __dsqrt_rd(lb_h6);
			double ub_height_rem = __dadd_ru(1.0, -__dadd_rd(lb_h6, lb_h7));
			double ub_width_recursion;

			if(ub_height_rem <= best_x12.p[1].y.get_lb()) {
				ub_width_recursion = ub_w67;
			} else {
				ub_width_recursion = ub_w67_wide;
			}

			if(can_recurse(__dadd_rd(vals.R.get_lb(), -r7.get_ub()), r7.get_ub(), ub_width_recursion, ub_height_rem)) {
				return true;
			}
		}
	}

	return false;
}

/*__device__ Bottom_row_2 circlecover::rectangle_size_bound::compute_bottom_row(IV la, IV s31, IV s32, IV R_no_7) {
	s31.tighten_lb(s32.get_lb());
	IV diff_s31_s32 = s31 - s32;
	diff_s31_s32.tighten_lb(0.0);
	IV w1_no_recursion = 0.5*la + 2.0*diff_s31_s32/la;
	IV w2_no_recursion = la - w1_no_recursion;
	IV w1_nr_sq = w1_no_recursion.square();
	IV w2_nr_sq = w2_no_recursion.square();
	IV h_no_recursion;
	double lb_h_recursion = 0.0;
	Point ux_no_recursion, ux_recursion;
	Circle c1_no_recursion, c2_no_recursion, c1_recursion, c2_recursion;

	// try without recursion
	if(definitely((h_no_recursion = s31 - 0.25*w1_nr_sq) > 0.0) && definitely(s32 - 0.25*w2_nr_sq > 0.0)) {
		h_no_recursion  = 2.0 * sqrt(h_no_recursion);
		ux_no_recursion = Point{w1_no_recursion, h_no_recursion};
		c1_no_recursion = Circle{{la-0.5*w1_no_recursion, 0.5*h_no_recursion}, s31};
		c2_no_recursion = Circle{{0.5*w2_no_recursion, 0.5*h_no_recursion}, s32};
	} else {
		h_no_recursion = IV{0.0,0.0};
	}

	// try with recursion
	if(R_no_7.get_lb() > 0.0) {
		// the area we want to cover by recursion
		double recursion_area = __ddiv_rd(R_no_7.get_lb(), C);

		// initially bound the tolerable width based on the allowed skew
		double lb_wr = __dsqrt_ru(__ddiv_ru(recursion_area, tolerated_lambda));
		double ub_wr = __dsqrt_rd(__dmul_rd(recursion_area, tolerated_lambda));

		// potentially tighten the lower bound based on the minimum width
		if(h_no_recursion.get_lb() <= 0.0) {
			double lb_wr_width = __dadd_ru(la.get_ub(), -__dadd_rd(__dmul_rd(2.0, __dsqrt_rd(s31.get_lb())), __dmul_rd(2.0, __dsqrt_rd(s32.get_lb()))));
			if(lb_wr_width > lb_wr) {
				lb_wr = lb_wr_width;
				if(lb_wr <= ub_wr) {
					// we do not have enough width
					Bottom_row_2 result;
					result.lb_height_border = 0.0;
					return result;
				}
			}
		}

		// run binary search for the best width
		for(;;) {
			double wr = 0.5 * (lb_wr + ub_wr);
			if(wr <= lb_wr || wr >= ub_wr) {
				break;
			}

			double hr = __ddiv_rd(recursion_area, wr);
			IV wtot = la - IV{wr,wr};
			IV w1 = 0.5*wtot + 2.0*diff_s31_s32/wtot;
			IV w2 = wtot - w1;
			double lb_h = (s31 - 0.25*w1.square()).get_lb();
			if(lb_h <= 0.0) {
				lb_wr = nextafter(wr, DBL_MAX);
				continue;
			}
			lb_h = __dmul_rd(2.0, __dsqrt_rd(lb_h));

			IV y = IV{0.5,0.5} * lb_h;
			Circle c1{{la-0.5*w1, y}, s31};
			Circle c2{{0.5*w2, y}, s32};
			Intersection_points x = intersection(c1, c2);

			// if there is no intersection, the width is too high
			if(!x.definitely_intersecting) {
				ub_wr = wr;
				continue;
			}

			// if we cannot definitely tell which intersection point is lower, we consider the width too high
			UB first_is_lower = (x.p[0].y < x.p[1].y);
			if(first_is_lower.is_uncertain()) {
				ub_wr = wr;
				continue;
			}
			bool bfirst_is_lower = first_is_lower.get_lb();

			// the highest possible y-coordinate for the lower intersection point
			double ub_lower_y = bfirst_is_lower ? x.p[0].y.get_ub() : x.p[1].y.get_ub();
			
			// check whether the lower intersection point is within the recursively covered region; otherwise, the width is too high
			if(hr < ub_lower_y) {
				ub_wr = wr;
				continue;
			}

			// otherwise, this width works
			lb_wr = wr;
			if(lb_h > lb_h_recursion) {
				lb_h_recursion = lb_h;
				ux_recursion = bfirst_is_lower ? x.p[1] : x.p[0];
				c1_recursion = c1;
				c2_recursion = c2;
			}
		}
	}

	Bottom_row_2 result;
	result.lb_height_border = 0.0;

	if(lb_h_recursion > h_no_recursion.get_lb()) {
		if(lb_h_recursion > 0) {
			result.upper_intersection = ux_recursion;
			result.c1 = c1_recursion;
			result.c2 = c2_recursion;
			result.lb_height_border = lb_h_recursion;
		}
	} else {
		if(h_no_recursion.get_lb() > 0) {
			result.upper_intersection = ux_no_recursion;
			result.c1 = c1_no_recursion;
			result.c2 = c2_no_recursion;
			result.lb_height_border = h_no_recursion.get_lb();
		}
	}

	return result;
} */

static inline __device__ double five_disks_maximize_height_23(double ub_w, double t1, double t2, double m1, double m2, double m3) {
	Max_height_strip_2 bot_strip = two_disks_maximal_height_strip(IV(ub_w,ub_w), IV(t1,t1), IV(t2,t2));
	if(bot_strip.lb_height <= 0.0) {
		return 0.0;
	}
	
	// find the upper intersection point of the bottom row
	Intersection_points xt = intersection(bot_strip.c1, bot_strip.c2);
	UB first_is_lower;
	if(!xt.definitely_intersecting) {
		algcuda::trap();
	}
	first_is_lower = xt.p[0].y <= xt.p[1].y;
	if(!first_is_lower.is_certain()) {
		return false;
	}
	bool bfirst_is_lower = first_is_lower.get_lb();
	Point xtup = xt.p[bfirst_is_lower ? 1 : 0];

	// maximize height of middle strip
	double lb_hmid = 0.0, ub_hmid = __dmul_rd(2.0, __dsqrt_rd(m3));
	for(;;) {
		double hmid = 0.5 * (lb_hmid + ub_hmid);
		if(hmid <= lb_hmid || hmid >= ub_hmid) {
			if(lb_hmid == 0.0) {
				return 0.0;
			} else {
				return __dadd_rd(bot_strip.lb_height, lb_hmid);
			}
		}

		double hmid_sq = __dmul_ru(hmid, hmid);

		double w1 = __dadd_rd(__dmul_rd(4.0, m1), -hmid_sq);
		double w2 = __dadd_rd(__dmul_rd(4.0, m2), -hmid_sq);
		double w3 = __dadd_rd(__dmul_rd(4.0, m3), -hmid_sq);
		if(w1 <= 0.0 || w2 <= 0.0 || w3 <= 0.0) {
			ub_hmid = hmid;
			continue;
		}
		w1 = __dsqrt_rd(w1);
		w2 = __dsqrt_rd(w2);
		w3 = __dsqrt_rd(w3);

		double wtot = __dadd_rd(w1, __dadd_rd(w2, w3));
		if(wtot >= ub_w) {
			lb_hmid = hmid;
			continue;
		}

		double wrem = __dadd_ru(ub_w, -__dadd_rd(w2, w3));
		double wrem_sq_fourth = __dmul_ru(0.25, __dmul_ru(wrem, wrem));
		double h1 = __dadd_rd(m1, -wrem_sq_fourth);
		if(h1 <= 0.0) {
			ub_hmid = hmid;
			continue;
		}
		h1 = __dsqrt_rd(h1);

		IV y23{__dadd_rd(bot_strip.lb_height, __dmul_rd(0.5, hmid)), __dadd_ru(bot_strip.lb_height, __dmul_ru(0.5, hmid))};
		Point c2{IV(w2,w2) * 0.5, y23}, c3{ub_w - IV(w3,w3) * 0.5, y23};

		// place m1 in the middle between the rectangles covered by m2, m3
		IV x1 = 0.5 * (IV(w2,w2) + ub_w - IV(w3,w3));
		IV y1{__dadd_rd(bot_strip.lb_height, __dadd_rd(hmid, -h1)), __dadd_ru(bot_strip.lb_height, __dadd_ru(hmid, -h1))};
		Point c1{x1,y1};

		// the intersection point of the two bottom disks must be in m1
		if(squared_distance(c1, xtup).get_ub() > m1) {
			ub_hmid = hmid;
			continue;
		}

		Circle d1{c1, {m1,m1}};
		Circle d2{c2, {m2,m2}};
		Circle d3{c3, {m3,m3}};
		Intersection_points x12 = intersection(d1, d2);
		Intersection_points x13 = intersection(d1, d3);

		if(!x12.definitely_intersecting || !x13.definitely_intersecting) {
			ub_hmid = hmid;
			continue;
		}

		UB x12_first_is_lower = (x12.p[0].y <= x12.p[1].y);
		UB x13_first_is_lower = (x13.p[0].y <= x13.p[1].y);
		if(!x12_first_is_lower.is_certain() || !x13_first_is_lower.is_certain()) {
			ub_hmid = hmid;
			continue;
		}

		// check that the two intersection points lie in the disks of the bottom row
		Point x12_low = x12.p[x12_first_is_lower.get_lb() ? 0 : 1];
		Point x13_low = x13.p[x13_first_is_lower.get_lb() ? 0 : 1];
		if(squared_distance(x12_low, bot_strip.c1.center).get_ub() <= t1 && squared_distance(x13_low, bot_strip.c2.center).get_ub() <= t2) {
			lb_hmid = hmid;
		} else {
			ub_hmid = hmid;
		}
	}
}

static __device__ IV compute_r7_range(const Variables& vars, const Intermediate_values& vals, double lb_w_covered, double ub_w_remaining, double ub_weight) {
	IV r7_range{0.0, vars.radii[5].get_ub()};
	r7_range.tighten_ub(vals.R.get_ub());

	double lb_goal_efficiency = __ddiv_rd(vals.R.get_lb(), ub_w_remaining);

	// if we have a goal efficiency for the rest of at least critical_ratio, we can potentially recurse using our size bound
	if(lb_goal_efficiency >= critical_ratio) {
		// how wide can we make the strip at efficiency critical_ratio?
		double lb_max_strip_width = __ddiv_rd(vals.R.get_lb(), critical_ratio);
		if(lb_max_strip_width >= 1.0) {
			r7_range.set_lb(1.0);
			r7_range.set_ub(0.0);
			return r7_range;
		} else {
			double lb_allowed_weight = bound_allowed_weight(lb_max_strip_width);
			r7_range.tighten_lb(lb_allowed_weight);
		}
	}

	if(lb_goal_efficiency >= C && !r7_range.empty()) {
		// upper bound on the skew of the remaining rectangle
		double lb_actual_skew = __drcp_rd(ub_w_remaining);
		double ub_actual_skew = __drcp_ru(ub_w_remaining);
		double ub_critical_ratio_for_skew = tight_rectangle::critical_ratio(IV(lb_actual_skew,ub_actual_skew)).get_ub();

		if(ub_critical_ratio_for_skew <= lb_goal_efficiency) {
			// we can recurse using Theorem 1 (worst case)
			r7_range.set_lb(1.0);
			r7_range.set_ub(0.0);
			return r7_range;
		} else {
			// compute size bound for covering using corollary 3 (large size bound corollary)
			double lb_lambda_goal_eff_sqrt = __dsqrt_rd(__dadd_rd(__dmul_rd(4.0, __dmul_rd(lb_goal_efficiency, lb_goal_efficiency)), -2.0));
			double lb_lambda_goal_eff = __dadd_rd(__dmul_rd(2.0, lb_goal_efficiency), lb_lambda_goal_eff_sqrt);
			double lb_sigma_goal_eff = __dmul_rd(lb_goal_efficiency, __dadd_rd(lb_lambda_goal_eff, -__ddiv_ru(2.0, lb_lambda_goal_eff)));
			double lb_scaled_sigma_goal_eff = __dmul_rd(lb_sigma_goal_eff, __dmul_rd(ub_w_remaining, ub_w_remaining));
			r7_range.tighten_lb(lb_scaled_sigma_goal_eff);
		}
	}

	return r7_range;
}

static __device__ bool six_disks_222_with_recursion(const Variables& vars, const Intermediate_values& vals, IV r7) {
	double lb_covered_12 = two_disks_maximize_height(vars.radii[0], vars.radii[1], 1.0);
	double lb_covered_34 = two_disks_maximize_height(vars.radii[2], vars.radii[3], 1.0);

	if(lb_covered_34 <= 0.0) {
		return false;
	}

	double r5 = vars.radii[4].get_lb();
	double r6 = vars.radii[5].get_lb();

	double wrem = __dadd_ru(vars.la.get_ub(), -__dadd_rd(lb_covered_12, lb_covered_34));
	double wrem_sq = __dmul_ru(wrem, wrem);
	double h1 = __dadd_rd(__dmul_rd(4.0, r5), -wrem_sq);
	double h2 = __dadd_rd(__dmul_rd(4.0, r6), -wrem_sq);
	if(h2 <= 0.0) {
		return false;
	}
	h1 = __dsqrt_rd(h1);
	h2 = __dsqrt_rd(h2);

	double hrem = __dadd_ru(1.0, -__dadd_rd(h1,h2));
	if(hrem <= 0) {
		return true;
	}

	// distance between upper boundary and r2's center
	double r2dy = __dadd_ru(1.0, -__dadd_rd(h1, __dmul_rd(0.5, h2)));
	r2dy = __dmul_ru(r2dy,r2dy);
	double wtop = __dadd_rd(r6, -r2dy);
	if(wtop <= 0) {
		return false;
	}

	wtop = __dmul_rd(2.0, __dsqrt_rd(wtop));
	
	// the width of each of the two remaining rectangles; their height is hrem
	double wrect = __dmul_ru(0.5, __dadd_ru(wrem, -wtop));

	// we have two cases: r7 definitely suffices for covering one rectangle,
	// or the smaller group after greedy splitting suffices to recurse on the rectangles
	double wr_sq = __dmul_ru(wrect,wrect);
	double hr_sq = __dmul_ru(hrem,hrem);
	double required_squared_r7 = __dmul_ru(0.25, __dadd_ru(wr_sq, hr_sq));

	if(required_squared_r7 <= r7.get_lb()) {
		double lb_R_no_7 = __dadd_rd(vals.R.get_lb(), -r7.get_ub());
		return can_recurse(lb_R_no_7, r7.get_ub(), wrect, hrem);
	}

	// the minimum weight of the smaller group
	double min_weight_smaller = __dmul_rd(0.5, vals.R.get_lb());
	min_weight_smaller = __dadd_rd(min_weight_smaller, -__dmul_ru(0.5, r7.get_ub()));

	// check that this suffices for either rectangle
	return can_recurse(min_weight_smaller, r7.get_ub(), wrect, hrem);
}

static __device__ bool different_height_r12_and_two_by_two(const Variables& vars, const Intermediate_values& vals, IV r7) {
	double w12 = __dsqrt_rd(__dmul_rd(2.0, vars.radii[1].get_lb()));
	double h2 = w12, h1 = __dsqrt_rd(__dadd_rd(__dmul_rd(4.0, vars.radii[0].get_lb()), -w12));
	double h12 = __dadd_rd(h1, h2);
	double wrem = __dadd_ru(vars.la.get_ub(), -w12);
	double h34 = two_disks_maximize_height(vars.radii[2], vars.radii[3], wrem);
	double h56 = two_disks_maximize_height(vars.radii[4], vars.radii[5], wrem);
	double h3456 = __dadd_rd(h34, h56);
	double hrem12   = __dadd_ru(1.0, -h12);
	double hrem3456 = __dadd_ru(1.0, -h3456);

	double hhigh = (hrem12 < hrem3456 ? hrem3456 : hrem12);
	double hlow  = (hrem12 < hrem3456 ? hrem12 : hrem3456);
	double whigh = (hrem12 < hrem3456 ? wrem : w12);
	double wlow  = (hrem12 < hrem3456 ? w12 : wrem);
	double wr7 = __dadd_rd(__dmul_rd(4.0, r7.get_lb()), -__dmul_ru(hhigh,hhigh));
	if(wr7 <= 0.0) {
		return false;
	}
	wr7 = __dsqrt_rd(wr7);
	whigh = __dadd_rd(whigh, -wr7);

	double lb_R8 = __dadd_rd(vals.R.get_lb(), -r7.get_ub());
	if(lb_R8 <= 0.0) {
		return false;
	}

	if(whigh <= 0.0) {
		// the high side disappeared
		wlow = __dadd_ru(whigh, wlow);
		return can_recurse(lb_R8, r7.get_ub(), wlow, hlow);
	}

	// at what efficiency can we cover the remaining area based on Corollary 3?
	// what is the shortest side involved?
	double shortest = hlow < wlow ? (hlow < whigh ? hlow : whigh) : (wlow < whigh ? wlow : whigh);
	double shortest_sq = __dmul_rd(shortest, shortest);
	double ub_sigma = __ddiv_ru(r7.get_ub(), shortest_sq);
	ub_sigma = __dmul_ru(ub_sigma, ub_sigma);
	double cr = __dmul_ru(0.5, __dsqrt_ru(__dadd_ru(__dsqrt_ru(__dadd_ru(ub_sigma, 1.0)), 1.0)));
	if(cr < C) { cr = C; }

	// for splitting into two groups, we pay a price; the price corresponds to some area that is covered twice.
	double extra_area = __ddiv_ru(r7.get_ub(), __dmul_rd(cr, hhigh));
	extra_area = __dmul_ru(extra_area, __dadd_ru(hhigh, -hlow));
	double ub_area = extra_area;
	ub_area = __dadd_ru(ub_area, __dmul_ru(whigh, hhigh));
	ub_area = __dadd_ru(ub_area, __dmul_ru(wlow, hlow));
	double weight_needed = __dmul_ru(ub_area, cr);
	return lb_R8 >= weight_needed;
}

static inline __device__ void restrict_r7_four_disk_strip_r1_square(const Variables& vars, const Intermediate_values& vals, IV& r7) {
	double lb_h2345 = nd_maximize_covered(vars.la.get_ub(), vars.radii + 1, 4);
	if(lb_h2345 <= 0.0) {
		return;
	}

	double lb_wh1 = __dsqrt_rd(__dmul_rd(2.0, vars.radii[0].get_lb()));
	double ub_hrem = __dadd_ru(1.0, -lb_h2345);
	double ub_hrem_bot_left = __dadd_ru(1.0, -__dadd_rd(lb_h2345, lb_wh1));

	if(ub_hrem_bot_left <= 0.0) {
		// cover the entire remaining left boundary using r1
		double lb_w1 = __dsqrt_rd(__dadd_rd(__dmul_rd(4.0, vars.radii[0].get_lb()), -__dmul_ru(ub_hrem,ub_hrem)));
		double ub_wrem = __dadd_ru(vars.la.get_ub(), -lb_w1);
		double lb_R6 = __dmul_rd(critical_ratio, vars.la.get_lb());
		for(int i = 0; i < 5; ++i) {
			lb_R6 = __dadd_rd(lb_R6, -vars.radii[i].get_ub());
		}

		if(can_recurse(lb_R6, vars.radii[5].get_ub(), ub_wrem, ub_hrem)) {
			r7.set_lb(DBL_MAX*DBL_MAX);
			return;
		}
	} else {
		// can r6 cover the remaining strip in the bottom-left corner?
		double r6_needed_entire_strip = __dmul_ru(0.25, __dadd_ru(__dmul_ru(lb_wh1,lb_wh1), __dmul_ru(ub_hrem_bot_left,ub_hrem_bot_left)));
		double ub_wrem = __dadd_ru(vars.la.get_ub(), -lb_wh1);
		if(r6_needed_entire_strip <= vars.radii[5].get_lb()) {
			r7.tighten_lb(recursion_bound_size(vals.R.get_lb(), ub_wrem, ub_hrem));
		} else {
			// place r6 covering a part of the remaining strip in the bottom-left corner
			double lb_w6 = __dadd_rd(__dmul_rd(4.0, vars.radii[5].get_lb()), -__dmul_ru(ub_hrem_bot_left,ub_hrem_bot_left));
			if(lb_w6 <= 0.0) {
				// r6 is too small for that
				return;
			}
			lb_w6 = __dsqrt_rd(lb_w6);

			// restrict r7 by first recursing on the large remaining rectangle and then on the remaining part of the strip
			double ub_strip_wrem = __dadd_ru(lb_wh1, -lb_w6);
			restrict_r7_t_shaped_recursion_split_recursion_largest_r7(vars, vals, r7, vals.R.get_lb(), ub_wrem, ub_hrem, ub_strip_wrem, ub_hrem_bot_left);
		}
	}
}

__device__ bool circlecover::rectangle_size_bound::seven_disk_strategies(const Variables& vars, const Intermediate_values& vals) {
	constexpr int r7_subintervals = 16;
	
	// try to cover as much width as possible with r1,...,r6
	double lb_w_covered = six_disks_maximize_covered_width(vars);
	if(lb_w_covered >= vars.la.get_ub()) {
		return true;
	}

	IV w_remaining = vars.la - lb_w_covered;
	w_remaining.tighten_lb(0.0);

	// compute an upper bound on the weight we have used so far
	double ub_weight = 0.0;
	for(int i = 0; i < 6; ++i) {
		ub_weight = __dadd_ru(ub_weight, vars.radii[i].get_ub());
	}

	// compute a range for r7; cut off all pieces where we could simply cover the remaining rectangle by some form of recursion
	IV r7_range = compute_r7_range(vars, vals, lb_w_covered, w_remaining.get_ub(), ub_weight);
	if(r7_range.empty()) {
		return true;
	}

	// further restrict r7 by using T-shaped recursion
	restrict_r7_t_shaped_recursion(vars, vals, r7_range);
	if(r7_range.empty()) {
		return true;
	}

	// further restrict r7 by using a 4-disk strip and a r1-square
	restrict_r7_four_disk_strip_r1_square(vars, vals, r7_range);
	if(r7_range.empty()) {
		return true;
	}

	double h34 = two_disks_maximize_height(vars.radii[2], vars.radii[3], vars.la.get_ub());

	for(int r7_offset = 0; r7_offset < r7_subintervals; ++r7_offset) {
		IV r7 = get_subinterval(r7_range, r7_offset, r7_subintervals);

		// first place r7 in the corner maximizing the height
		if(r7_in_corner_and_recurse(vals.R, r7, w_remaining.get_ub())) {
			continue;
		}

		// place three disks at the top and split the remaining area,
		// covering one part explicitly and one part recursively
		if(three_disk_top_strip(vars, vals.R, r7)) {
			continue;
		}

		// try placing six disks in three groups of two
		if(six_disks_222_with_recursion(vars, vals, r7)) {
			continue;
		}

		// try covering disks in three columns
		if(r435_strip_covering_r12_gap_r67_right(vars, r7, vals)) {
			continue;
		}

		// cover using a strategy involving a rectangle covered by r1,r2 and a 2x2 cover using r3,...,r6
		if(different_height_r12_and_two_by_two(vars, vals, r7)) {
			continue;
		}

		double lb_height_5d = 0.0;
		if(h34 > 0.0) {
			lb_height_5d = five_disks_maximize_height_23(vars.la.get_ub(), vars.radii[0].get_lb(), vars.radii[1].get_lb(), vars.radii[4].get_lb(), vars.radii[5].get_lb(), r7.get_lb());
			if(__dadd_rd(h34, lb_height_5d) >= 1.0) {
				continue;
			}
		}

		double ub_area_remaining  = w_remaining.get_ub();
		double lb_goal_efficiency = __ddiv_rd(vals.R.get_lb(), ub_area_remaining);
		double lb_lambda_goal_eff_sqrt = __dsqrt_rd(__dadd_rd(__dmul_rd(4.0, __dmul_rd(lb_goal_efficiency, lb_goal_efficiency)), -2.0));
		double lb_lambda_goal_eff = __dadd_rd(__dmul_rd(2.0, lb_goal_efficiency), lb_lambda_goal_eff_sqrt);
		double lb_sigma_goal_eff = __dmul_rd(lb_goal_efficiency, __dadd_rd(lb_lambda_goal_eff, -__ddiv_ru(2.0, lb_lambda_goal_eff)));
		double lb_scaled_sigma_goal_eff = __dmul_rd(lb_sigma_goal_eff, __dmul_rd(w_remaining.get_ub(), w_remaining.get_ub()));
		
		printf("r1: [%.19g,%.19g], r2: [%.19g,%.19g], r3: [%.19g,%.19g], r4: [%.19g,%.19g], r5: [%.19g,%.19g], r6: [%.19g,%.19g], Open range for r7: [%.19g,%.19g]; problematic subinterval: [%.19g,%.19g], w_remaining: [%.19g,%.19g], goal efficiency >= %.19g, sigma_goal_eff >= %.19g, sigma_goal_eff_scaled >= %.19g, R_7 = [%.19g,%.19g], height_5d >= %.19g\n",
				vars.radii[0].get_lb(), vars.radii[0].get_ub(),
				vars.radii[1].get_lb(), vars.radii[1].get_ub(),
				vars.radii[2].get_lb(), vars.radii[2].get_ub(),
				vars.radii[3].get_lb(), vars.radii[3].get_ub(),
				vars.radii[4].get_lb(), vars.radii[4].get_ub(),
				vars.radii[5].get_lb(), vars.radii[5].get_ub(),
				r7_range.get_lb(), r7_range.get_ub(), r7.get_lb(), r7.get_ub(),
				w_remaining.get_lb(), w_remaining.get_ub(),
				lb_goal_efficiency, lb_sigma_goal_eff, lb_scaled_sigma_goal_eff,
				(vals.R - r7).get_lb(), (vals.R - r7).get_ub(),
				lb_height_5d
		);

		return false;
	}

	return true;
}

