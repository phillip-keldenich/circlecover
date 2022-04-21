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

__device__ static inline bool six_disks_can_cover_width_r56_center(double w, double r1, double r2, double r3, double r4, double r5, double r6) {
	double lb_h14 = two_disks_maximize_height(IV(r1,r1), IV(r4,r4), w);
	double lb_h23 = two_disks_maximize_height(IV(r2,r2), IV(r3,r3), w);

	if(lb_h14 <= 0.0 || lb_h23 <= 0.0) {
		return false;
	}

	double ub_hrem = __dadd_ru(1.0, -__dadd_rd(lb_h14, lb_h23));
	double ub_hrem_sq = __dmul_ru(ub_hrem, ub_hrem);
	double ub_hrem_sq_fourth = __dmul_ru(0.25, ub_hrem_sq);

	double lb_x_r5 = __dadd_rd(r5, -ub_hrem_sq_fourth);
	double ub_x_r6 = __dadd_rd(r6, -ub_hrem_sq_fourth);
	if(lb_x_r5 <= 0.0 || ub_x_r6 <= 0.0) {
		return false;
	}
	lb_x_r5 = __dsqrt_rd(lb_x_r5);
	ub_x_r6 = __dsqrt_rd(ub_x_r6);
	// if r5,r6 can cover the remaining strip, we are done
	if(__dmul_rd(2.0, __dadd_rd(lb_x_r5, ub_x_r6)) >= w) {
		return true;
	}
	ub_x_r6 = __dadd_ru(w, -ub_x_r6);

	IV y56{__dadd_rd(lb_h14, __dmul_rd(0.5, ub_hrem)), __dadd_ru(lb_h14, __dmul_ru(0.5, ub_hrem))};
	IV y14{__dmul_rd(0.5, lb_h14), __dmul_ru(0.5, lb_h14)};
	IV y23{__dadd_rd(1.0, -__dmul_ru(0.5, lb_h23)), __dadd_ru(1.0, -__dmul_rd(0.5, lb_h23))};

	IV h14_sq{__dmul_rd(lb_h14, lb_h14), __dmul_ru(lb_h14, lb_h14)};
	IV h23_sq{__dmul_rd(lb_h23, lb_h23), __dmul_ru(lb_h23, lb_h23)};
	h14_sq *= 0.25;
	h23_sq *= 0.25;
	
	IV w1 = sqrt(r1 - h14_sq);
	IV w2 = sqrt(r2 - h23_sq);
	IV x1 = w - w1;
	IV x2 = w - w2;
	w1 *= 2.0;
	w2 *= 2.0;

	IV x5{lb_x_r5,lb_x_r5};
	IV x6{ub_x_r6,ub_x_r6};

	Point p1{x1,y14};
	Point p2{x2,y23};
	Point p5{x5,y56};
	Point p6{x6,y56};
	Point ps1{w-w1, {lb_h14,lb_h14}};
	Point ps2{w-w2, 1.0 - IV{lb_h23,lb_h23}};

	Circle c1{p1, {r1,r1}};
	Circle c2{p2, {r2,r2}};

	Intersection_points x12 = intersection(c1,c2);
	if(!x12.definitely_intersecting || !definitely(x12.p[0].x < x12.p[1].x)) {
		return false;
	}

	return squared_distance(p5, ps1).get_ub() <= r5 &&
		squared_distance(p5, ps2).get_ub() <= r5 &&
		squared_distance(p5, x12.p[0]).get_ub() <= r5 &&
		squared_distance(p6, x12.p[1]).get_ub() <= r6;
}

__device__ static inline bool six_disks_can_cover_width_r56_center(const Variables& vars, double width) {
	return six_disks_can_cover_width_r56_center(width, vars.radii[0].get_lb(), vars.radii[1].get_lb(), vars.radii[2].get_lb(), vars.radii[3].get_lb(), vars.radii[4].get_lb(), vars.radii[5].get_lb());
}

namespace {
struct Combination {
	int first [3];
	int second[3];
};

struct Combination3 {
	int first[2];
	int second[2];
	int third[2];
};

static __device__ const Combination combinations[] = {
	{ {0,1,2}, {3,4,5} },
	{ {0,1,3}, {2,4,5} },
	{ {0,1,4}, {2,3,5} },
	{ {0,1,5}, {2,3,4} },
	{ {0,2,3}, {1,4,5} },
	{ {0,2,4}, {1,3,5} },
	{ {0,2,5}, {1,3,4} },
	{ {0,3,4}, {1,2,5} },
	{ {0,3,5}, {1,2,4} },
	{ {0,4,5}, {1,2,3} }
};

static __device__ const Combination3 combinations3[] = {
	{ {0,1}, {2,3}, {4,5} },
	{ {0,1}, {2,4}, {3,5} },
	{ {0,1}, {2,5}, {3,4} },
	{ {0,2}, {1,3}, {4,5} },
	{ {0,2}, {1,4}, {3,5} },
	{ {0,2}, {1,5}, {3,4} },
	{ {0,3}, {1,2}, {4,5} },
	{ {0,3}, {1,4}, {2,5} },
	{ {0,3}, {1,5}, {2,4} },
	{ {0,4}, {1,2}, {3,5} },
	{ {0,4}, {1,3}, {2,5} },
	{ {0,4}, {1,5}, {2,3} },
	{ {0,5}, {1,2}, {3,4} },
	{ {0,5}, {1,3}, {2,4} },
	{ {0,5}, {1,4}, {2,3} }
};
}

__device__ static double six_disks_maximize_covered_width_2times3(const Variables& vars) {
	double result = 0.0;

	for(const Combination& c : combinations) {
		double wfirst  = three_disks_maximize_height( vars.radii[c.first[0]],  vars.radii[c.first[1]],  vars.radii[c.first[2]], 1.0);
		double wsecond = three_disks_maximize_height(vars.radii[c.second[0]], vars.radii[c.second[1]], vars.radii[c.second[2]], 1.0);
		double wtotal = __dadd_rd(wfirst,wsecond);
		
		if(wtotal > result) {
			result = wtotal;
		}
	}

	return result;
}

__device__ static double six_disks_maximize_covered_width_3times2(const Variables& vars) {
	double result = 0.0;

	for(const Combination3& c : combinations3) {
		double w1 = two_disks_maximize_height( vars.radii[c.first[0]],  vars.radii[c.first[1]], 1.0);
		double w2 = two_disks_maximize_height(vars.radii[c.second[0]], vars.radii[c.second[1]], 1.0);
		double w3 = two_disks_maximize_height( vars.radii[c.third[0]],  vars.radii[c.third[1]], 1.0);
		double wtotal = __dadd_rd(w1, __dadd_rd(w2,w3));
		if(wtotal > result) {
			result = wtotal;
		}
	}

	return result;
}

__device__ double  circlecover::rectangle_size_bound::six_disks_maximize_covered_width(const Variables& vars) {
	double w = six_disks_maximize_covered_width_3times2(vars);
	double w2 = six_disks_maximize_covered_width_2times3(vars);
	if(w2 > w) {
		w = w2;
	}

	if(six_disks_can_cover_width_r56_center(vars, w)) {
		double lb = w;
		double ub = nextafter(vars.la.get_ub(), DBL_MAX);

		for(;;) {
			double mid = 0.5 * (lb+ub);
			if(mid <= lb || mid >= ub) {
				w = lb;
				break;
			}

			if(six_disks_can_cover_width_r56_center(vars, mid)) {
				lb = mid;
			} else {
				ub = mid;
			}
		}
	}

	return w;
}

