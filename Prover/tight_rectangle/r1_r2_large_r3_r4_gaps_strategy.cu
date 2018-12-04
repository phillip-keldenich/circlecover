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

using namespace circlecover;
using namespace circlecover::tight_rectangle;

static inline bool __device__ can_realize_gap_width(double gap_width, double r1, double w1, double r2, double w2, double r4) {
	IV y4 = r4 - 0.25*IV{gap_width,gap_width}*gap_width;
	if(possibly(y4 <= 0.0)) {
		return false;
	}
	
	IV x1 = IV{0.5,0.5} * w1;
	IV x2 = IV{0.5,0.5} * w2 + gap_width + w1;
	Circle c1{{x1, {0.5,0.5}},{r1,r1}};
	Circle c2{{x2, {0.5,0.5}},{r2,r2}};

	Intersection_points x12 = intersection(c1,c2);
	if(!x12.definitely_intersecting) {
		return false;
	}

	UB first_is_lower = x12.p[0].y <= x12.p[1].y;
	if(!first_is_lower.is_certain()) {
		return false;
	}

	bool bfirst_lower = first_is_lower.get_lb();
	IV x4 = IV{0.5,0.5}*gap_width + w1;
	y4 = sqrt(y4);

	Point p4{x4,y4};
	Point px = x12.p[bfirst_lower ? 0 : 1];
	return definitely(squared_distance(p4,px) <= r4);
}

bool __device__ circlecover::tight_rectangle::r1_r2_large_r3_r4_gaps_strategy(const Variables& vars, IV R) {
	const double r1 = vars.radii[0].get_lb();
	const double r2 = vars.radii[1].get_lb();
	const double r4 = vars.radii[3].get_lb();

	double lb_w1 = __dadd_rd(__dmul_rd(4.0, r1), -1.0);
	double lb_w2 = __dadd_rd(__dmul_rd(4.0, r2), -1.0);
	if(lb_w1 <= 0.0 || lb_w2 <= 0.0) {
		return false;
	}

	lb_w1 = __dsqrt_rd(lb_w1);
	lb_w2 = __dsqrt_rd(lb_w2);

	double lb_gw = 0.0, ub_gw = __dadd_rd(__dsqrt_rd(r1),__dsqrt_rd(r2));
	for(;;) {
		double mid = 0.5 * (lb_gw + ub_gw);
		if(mid <= lb_gw || mid >= ub_gw) {
			break;
		}

		if(can_realize_gap_width(mid, r1, lb_w1, r2, lb_w2, r4)) {
			lb_gw = mid;
		} else {
			ub_gw = mid;
		}
	}

	double lb_wtot = __dadd_rd(lb_w1, __dadd_rd(lb_gw, lb_w2));
	double ub_wrem = __dadd_ru(vars.la.get_ub(), -lb_wtot);
	if(ub_wrem <= 0.0) {
		return true;
	}

	double ub_r5 = R.get_ub();
	if(ub_r5 > vars.radii[3].get_ub()) {
		ub_r5 = vars.radii[3].get_ub();
	}
	return can_recurse(ub_wrem, 1.0, ub_r5, R.get_lb());
}

