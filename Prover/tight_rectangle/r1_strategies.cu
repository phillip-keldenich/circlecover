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
#include "../rectangle_size_bound/strategies.cuh"
#include <algcuda/exit.cuh>

bool __device__ circlecover::tight_rectangle::r1_strategies(IV la, IV r1, double ub_r2) {
	double square_width = __dsqrt_rd(__dmul_rd(2.0, r1.get_lb()));
	double strip_width  = __dadd_rd(__dmul_rd(4.0, r1.get_lb()), -1.0);
	const double lb_rem_weight = (required_weight_for(la) - r1).get_lb();

	if(strip_width > 0.0) {
		strip_width = __dsqrt_rd(strip_width);
		
		IV wr = la-strip_width;
		IV hr{1.0,1.0};

		if(can_recurse(wr.get_ub(), 1.0, ub_r2, lb_rem_weight)) {
			return true;
		}
	}

	if(square_width < 1.0) {
		double ub_weight_right;
		double wright = __dadd_ru(la.get_ub(), -square_width);
		double minright = wright < 1.0 ? wright : 1.0;

		if(rectangle_size_bound::disk_satisfies_size_bound(ub_r2, minright)) {
			// compute the cost of covering the remaining strip using size bound
			ub_weight_right = __dmul_ru(wright, rectangle_size_bound::critical_ratio);
		} else {
			// compute the cost of covering the remaining strip using recursion
			ub_weight_right = required_weight_for(IV(wright,wright), IV(1.0,1.0)).get_ub();
		}

		// additional cost by splitting
		ub_weight_right = __dadd_ru(ub_weight_right, ub_r2);
		double lb_rem_weight_top = __dadd_rd(lb_rem_weight, -ub_weight_right);

		if(lb_rem_weight_top > 0) {
			double htop = __dadd_ru(1.0, -square_width);
			
			if(can_recurse(square_width, htop, ub_r2, lb_rem_weight_top)) {
				return true;
			}
		}
	}

	double lb_S1 = __dadd_rd(__dmul_rd(4.0, r1.get_lb()), -1.0);
	if(lb_S1 > 0.0) {
		lb_S1 = __dsqrt_rd(lb_S1);
		double la_m_S1half = __dadd_ru(la.get_ub(), -__dmul_rd(0.5, lb_S1));
		la_m_S1half = __dmul_ru(la_m_S1half, la_m_S1half);
		double lb_h1 = __dadd_rd(r1.get_lb(), -la_m_S1half);
		if(lb_h1 > 0.0) {
			lb_h1 = __dmul_rd(2.0, __dsqrt_rd(lb_h1));
			double ub_pocket_height = __dmul_ru(0.5, __dadd_ru(1.0, -lb_h1));
			double ub_pocket_width  = __dadd_ru(la.get_ub(), -lb_S1);
			double lb_rem_weight_uneven = __dmul_rd(0.5, __dadd_rd(lb_rem_weight, -ub_r2));
			if(lb_rem_weight_uneven > 0 && can_recurse(ub_pocket_width, ub_pocket_height, ub_r2, lb_rem_weight_uneven)) {
				return true;
			}
		}
	}

	return false;
}

