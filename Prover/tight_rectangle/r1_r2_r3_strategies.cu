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
using namespace circlecover::tight_rectangle;

static inline bool __device__ r1_r2_r3_strategy_horizontal(IV la, IV r1, IV r2, IV r3, double ub_4) {
	double lb_w3 = __dadd_rd(__dmul_rd(4.0, r3.get_lb()), -1.0);
	if(lb_w3 <= 0.0) {
		return false;
	}

	lb_w3 = __dsqrt_rd(lb_w3);
	double lb_w2 = __dsqrt_rd(__dadd_rd(__dmul_rd(4.0, r2.get_lb()), -1.0));
	double lb_w1 = __dsqrt_rd(__dadd_rd(__dmul_rd(4.0, r1.get_lb()), -1.0));

	double lb_wcov = __dadd_rd(lb_w1, __dadd_rd(lb_w2, lb_w3));
	double lb_R4 = (required_weight_for(la) - r1 - r2 - r3).get_lb();
	double ub_wrem = __dadd_ru(la.get_ub(), -lb_wcov);
	if(ub_wrem <= 0.0) {
		return true;
	}

	return can_recurse(ub_wrem, 1.0, ub_4, lb_R4);
}

static inline bool __device__ can_realize_strip_width(double lb_r1, double lb_r2, double lb_r3, double w) {
	double lb_h1 = __dadd_rd(__dmul_rd(4.0, lb_r1), -__dmul_ru(w,w));
	if(lb_h1 <= 0.0) {
		return false;
	}
	lb_h1 = __dsqrt_rd(lb_h1);
	double ub_hrem = __dadd_ru(1.0, -lb_h1);
	double h_sq = __dmul_ru(ub_hrem, ub_hrem);
	double lb_w2 = __dadd_rd(__dmul_rd(4.0, lb_r2), -h_sq);
	double lb_w3 = __dadd_rd(__dmul_rd(4.0, lb_r3), -h_sq);

	if(lb_w3 <= 0.0 || lb_w2 <= 0.0) {
		return false;
	}

	lb_w2 = __dsqrt_rd(lb_w2);
	lb_w3 = __dsqrt_rd(lb_w3);
	double lb_w23 = __dadd_rd(lb_w2, lb_w3);
	return lb_w23 >= w;
}

static inline bool __device__ r1_r2_r3_strategy_cover_strip(IV la, IV r1, IV r2, IV r3, double ub_4) {
	double lb_w = 0.3;
	double ub_w = la.get_ub();

	if(!can_realize_strip_width(r1.get_lb(), r2.get_lb(), r3.get_lb(), lb_w)) {
		return false;
	}

	for(;;) {
		double mid = 0.5 * (lb_w + ub_w);
		if(mid <= lb_w || mid >= ub_w) {
			break;
		}

		if(can_realize_strip_width(r1.get_lb(), r2.get_lb(), r3.get_lb(), mid)) {
			lb_w = mid;
		} else {
			ub_w = mid;
		}
	}

	// use lb_w (this is a strip width we can definitely achieve)
	double ub_wrem = __dadd_ru(la.get_ub(), -lb_w);
	double lb_R4 = (required_weight_for(la) - r1 - r2 - r3).get_lb();
	return can_recurse(ub_wrem, 1.0, ub_4, lb_R4);
}

bool __device__ circlecover::tight_rectangle::r1_r2_r3_strategies(IV la, IV r1, IV r2, IV r3, double ub_4) {
	return r1_r2_r3_strategy_horizontal(la, r1, r2, r3, ub_4) ||
		r1_r2_r3_strategy_cover_strip(la, r1, r2, r3, ub_4);
}

static inline __device__ bool three_disk_recursion_big_strip(double ub_la, double r1, double r2, double r3, double ub_r4, double lb_R4) {
	double S_sq_lb = __dadd_rd(__dadd_rd(__dmul_rd(2.0,r2), __dadd_rd(__dmul_rd(2.0,r3), -0.25)), __dadd_rd(__dadd_rd(-__dmul_ru(4.0,__dmul_ru(r2,r2)), __dmul_rd(8.0,__dmul_rd(r2,r3))), -__dmul_ru(4.0,__dmul_ru(r3,r3))));
	if(S_sq_lb <= 0.0) {
		return false;
	}

	double S_lb = __dsqrt_rd(S_sq_lb);
	double la_rem_ub = __dadd_ru(ub_la, -S_lb);
	double h1_sq_lb = __dadd_rd(__dmul_rd(4.0, r1), -__dmul_ru(la_rem_ub, la_rem_ub));

	if(h1_sq_lb <= 0.0) {
		return false;
	}

	if(h1_sq_lb >= 1.0) {
		return true;
	}

	double h1_rem_ub = __dadd_ru(1.0, -__dsqrt_rd(h1_sq_lb));
	return can_recurse(la_rem_ub, h1_rem_ub, ub_r4, lb_R4);
}

static inline bool __device__ three_disk_recursion_small_corner(double ub_la, double r1, double r2, double r3, double ub_r4, double lb_R4) {
	// r3 covers this much of both sides it intersects
	double lb_r3_covered = __dsqrt_rd(__dmul_rd(2.0, r3));
	
	if(lb_r3_covered >= 1.0) {
		// this should not happen; in this case, this strategy is not the way to go!
		return false;
	}

	double ub_la_remains = __dadd_ru(ub_la, -lb_r3_covered);
	double ub_la_r_half  = __dmul_ru(ub_la_remains, 0.5);
	double ub_1_remains  = __dadd_ru(1.0, -lb_r3_covered);

	if(ub_la_r_half <= __dsqrt_rd(r2)) {
		// if we can, place r1 above r3
		double lb_r1_covered = __dsqrt_rd(__dadd_rd(__dmul_rd(4.0, r1), -__dmul_ru(ub_1_remains, ub_1_remains)));
		double ub_la_remains_r1 = __dadd_ru(ub_la, -lb_r1_covered);
		double lb_r2_covered = __dsqrt_rd(__dadd_rd(__dmul_rd(4.0, r2), -__dmul_ru(ub_la_remains, ub_la_remains)));
		double ub_1_remains_r2 = __dadd_ru(1.0, -lb_r2_covered);

		// a ub_la_remains_r1 x ub_1_remains_r2 rectangle remains to be covered
		return can_recurse(ub_la_remains_r1, ub_1_remains_r2, ub_r4, lb_R4);
	} else {
		// r2 is too small
		if(ub_la_r_half >= __dsqrt_rd(r1)) {
			// r1 is too small as well
			return false;
		}

		if(ub_1_remains >= __dmul_rd(2.0, __dsqrt_rd(r2))) {
			// r2 is too small to cover the shorter side
			return false;
		}

		double lb_r2_covered = __dsqrt_rd(__dadd_rd(__dmul_rd(4.0, r2), -__dmul_ru(ub_1_remains, ub_1_remains)));
		double ub_la_remains_r2 = __dadd_ru(ub_la, -lb_r2_covered);
		double lb_r1_covered = __dsqrt_rd(__dadd_rd(__dmul_rd(4.0, r1), -__dmul_ru(ub_la_remains, ub_la_remains)));
		double ub_1_remains_r1 = __dadd_ru(1.0, -lb_r1_covered);
		
		// a ub_la_remains_r2 x ub_1_remains_r1 rectangle remains to be covered
		return can_recurse(ub_la_remains_r2, ub_1_remains_r1, ub_r4, lb_R4);
	}
}

static inline bool __device__ three_disk_recursion_r1_r3_pocket(IV la, IV r1, IV r2, IV r3, double ub_r4, double lb_R4) {
	if(r1.get_lb() <= 0.25) {
		return false;
	}
	IV S1 = 2.0 * sqrt(r1 - 0.25);

	IV la_m_S1 = la - S1;
	IV h2_rest = 0.25*la_m_S1*la_m_S1;
	if(r2.get_lb() <= h2_rest.get_ub()) {
		return false;
	}
	IV h2 = 2.0 * sqrt(r2 - h2_rest);

	IV v1_m_h2 = 1.0 - h2;
	IV S3_rest = 0.25*v1_m_h2*v1_m_h2;
	if(r3.get_lb() <= S3_rest.get_ub()) {
		return false;
	}
	IV S3 = 2.0 * sqrt(r3 - S3_rest);

	IV X = la - S1 - S3;
	if(X.get_ub() <= 0.0) {
		return true;
	}

	IV la_m_S1h_S3 = la - 0.5*S1 - S3;
	IV Y_rest = la_m_S1h_S3*la_m_S1h_S3; 
	if(r1.get_lb() <= Y_rest.get_ub()) {
		return false;
	}

	IV Y = 0.5 - sqrt(r1 - Y_rest);
	if(Y.get_ub() <= 0.0) {
		return true;
	}

	double x = X.get_ub();
	double y = Y.get_ub();
	return can_recurse(x, y, ub_r4, lb_R4);
}

static inline bool __device__ manually_proved_three_disk_pocket(IV la, IV r1, IV r2, IV r3) {
	if(la.get_ub() > 1.0499999999999998) {
		return false;
	}
	IV lasq = square(la);
	// the weight of a disk at the three-disk worst-case
	IV rstar = 0.0625 * lasq + 0.15625 + 0.03515625 / lasq;
	// check if we are fully in the manually-proved range
	IV d1 = rstar - r1;
	if(d1.get_lb() < -0.0009999999999999998 || d1.get_ub() > 0.0009999999999999998) {
		return false;
	}
	IV d2 = rstar - r2;
	if(d2.get_lb() < -0.0019999999999999996 || d2.get_ub() > 0.0019999999999999996) {
		return false;
	}
	IV d3 = rstar - r3;
	if(d3.get_lb() < -0.0029999999999999996 || d3.get_ub() > 0.0029999999999999996) {
		return false;
	}
	// compute X = lambda - S1 - S3
	IV S1 = 2.0 * sqrt(r1 - 0.25);
	IV la_m_S1 = la - S1;
	IV h2_rest = 0.25*la_m_S1*la_m_S1;
	if(r2.get_lb() <= h2_rest.get_ub()) {
		return false;
	}
	IV h2 = 2.0 * sqrt(r2 - h2_rest);
	IV v1_m_h2 = 1.0 - h2;
	IV S3_rest = 0.25*v1_m_h2*v1_m_h2;
	if(r3.get_lb() <= S3_rest.get_ub()) {
		return false;
	}
	IV S3 = 2.0 * sqrt(r3 - S3_rest);
	IV X = la - S1 - S3;
	if(X.get_ub() > 0.046999999999999993) {
		return false;
	}
	return true;
}

bool __device__ circlecover::tight_rectangle::r1_r2_r3_strategies(const Variables& vars, IV R4) {
	const double ub_la = vars.la.get_ub();
	const double r1 = vars.radii[0].get_lb();
	const double r2 = vars.radii[1].get_lb();
	const double r3 = vars.radii[2].get_lb();
	const double ub_r4 = vars.radii[3].get_ub();
	const double lb_R4 = R4.get_lb();
	return three_disk_recursion_big_strip(ub_la, r1, r2, r3, ub_r4, lb_R4) || 
		three_disk_recursion_small_corner(ub_la, r1, r2, r3, ub_r4, lb_R4) ||
		three_disk_recursion_r1_r3_pocket(vars.la, vars.radii[0], vars.radii[1], vars.radii[2], ub_r4, lb_R4) ||
		manually_proved_three_disk_pocket(vars.la, vars.radii[0], vars.radii[1], vars.radii[2]);
}

