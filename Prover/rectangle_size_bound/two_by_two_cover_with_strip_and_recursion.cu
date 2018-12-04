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

__device__ bool  circlecover::rectangle_size_bound::two_by_two_cover_with_strip_and_recursion(const Variables& vars, const Intermediate_values& vals) {
	const double r5 = vars.radii[4].get_lb();
	const double r6 = vars.radii[5].get_lb();
	const double R = vals.R.get_lb();

	double ub_w = __dadd_ru(vars.la.get_ub(), -vals.cover_2x2.width);

	// remaining weight including r5,r6 (or only r6)
	double R5 = __dmul_rd(critical_ratio, vars.la.get_lb());
	double R6 = __dadd_rd(R5, -vars.radii[4].get_ub());
	for(int i = 0; i < 4; ++i) {
		R5 = __dadd_rd(R5, -vars.radii[i].get_ub());
		R6 = __dadd_rd(R6, -vars.radii[i].get_ub());
	}
	
	// try recursion with disks starting from r5: worst-case and size-bounded
	if(can_recurse(R5, vars.radii[4].get_ub(), ub_w, 1.0)) {
		return true;
	}

	double ub_wsq = __dmul_ru(ub_w, ub_w);
	double lb_h5 = __dadd_rd(__dmul_rd(4.0, r5), -ub_wsq);
	if(lb_h5 <= 0.0) {
		return false;
	}
	lb_h5 = __dsqrt_rd(lb_h5);
	double ub_hrem = __dadd_ru(1.0, -lb_h5);

	// try recursion with disks starting from r6: worst-case and size-bounded
	if(lb_h5 >= 1.0 || can_recurse(R6, vars.radii[5].get_ub(), ub_w, ub_hrem)) {
		return true;
	}

	// try recursion with disks starting from r7
	double lb_h6 = __dadd_rd(__dmul_rd(4.0, r6), -ub_wsq);
	if(lb_h6 > 0.0) {
		lb_h6 = __dsqrt_rd(lb_h6);
		if(lb_h6 >= ub_hrem) {
			return true;
		} else {
			double ub_hrem6 = __dadd_ru(ub_hrem, -lb_h6);
			if(can_recurse(R, vars.radii[5].get_ub(), ub_w, ub_hrem6)) {
				return true;
			}
		}
	}

	// try another way of placing r6, making use of disk intersections
	Circle c5{{vars.la - IV{0.5,0.5}*ub_w, IV{0.5,0.5}*lb_h5}, {r5,r5}};
	double ub_hremsq = __dmul_ru(ub_hrem, ub_hrem);
	double lb_w6 = __dadd_rd(__dmul_rd(4.0, r6), -ub_hremsq);
	if(lb_w6 > 0.0) {
		lb_w6 = __dsqrt_rd(lb_w6);
		if(lb_w6 >= ub_w) {
			return true;
		}

		Circle c6{{vars.la - IV{0.5,0.5}*lb_w6, 1.0 - IV{0.5,0.5}*ub_hrem}, {r6,r6}};
		Intersection_points x56 = intersection(c5, c6);
		if(x56.definitely_intersecting && x56.p[0].x.get_ub() < x56.p[1].x.get_lb()) {
			// check if one intersection point is definitely covered
			if(definitely(squared_distance(x56.p[0], vals.cover_2x2.circles[3].center) <= vals.cover_2x2.circles[3].squared_radius)) {
				Intersection_points x46 = intersection(c6, vals.cover_2x2.circles[3]);
				if(x46.definitely_intersecting) {
					UB first_higher = (x46.p[0].y > x46.p[1].y);
					if(first_higher.is_certain()) {
						bool bfirst_higher = first_higher.get_lb();
						double lb_y = x46.p[bfirst_higher ? 0 : 1].y.get_lb();
						double ub_hrect = __dadd_ru(1.0, -lb_y);
						if(can_recurse(vals.R.get_lb(), vars.radii[5].get_ub(), __dadd_ru(ub_w, -lb_w6), ub_hrect)) {
							return true;
						}
					}
				}
			}
		}
	}

	return false;
}

