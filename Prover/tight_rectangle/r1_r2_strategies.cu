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

using namespace circlecover;
using namespace circlecover::tight_rectangle;

static inline bool __device__ r1_r2_strategies_simple(IV la, IV r1, IV r2, double ub_r3) {
	double lb_width = rectangle_size_bound::two_disks_maximize_height(r1, r2, 1.0);
	if(lb_width <= 0.0) {
		return false;
	}

	double lb_w2 = __dadd_rd(__dmul_rd(4.0, r2.get_lb()), -1.0);
	if(lb_w2 > 0.0) {
		lb_w2 = __dsqrt_rd(lb_w2);
		double lb_w1 = __dsqrt_rd(__dadd_rd(__dmul_rd(4.0, r1.get_lb()), -1.0));
		double lb_w = __dadd_rd(lb_w1, lb_w2);
		if(lb_w > lb_width) {
			lb_width = lb_w;
			if(lb_w > la.get_ub()) {
				return true;
			}
		}
	}

	double wrem = __dadd_rd(la.get_ub(), lb_width);
	double lb_R3 = (required_weight_for(la) - r1 - r2).get_lb();
	return can_recurse(wrem, 1.0, ub_r3, lb_R3);
}

static inline bool __device__ r1_r2_strategies_stacked(IV la, IV r1, IV r2, double ub_r3) {
	double lb_square_side = __dsqrt_rd(__dmul_rd(2.0, r1.get_lb()));
	if(lb_square_side > 1.0) {
		return false;
	}

	double ub_top_hrem = __dadd_ru(1.0, -lb_square_side);
	double lb_w2 = __dadd_rd(__dmul_rd(4.0, r2.get_lb()), -__dmul_ru(ub_top_hrem,ub_top_hrem));
	if(lb_w2 < 0.0) {
		return false;
	}
	lb_w2 = __dsqrt_rd(lb_w2);

	double w_top = __dadd_ru(la.get_ub(), -lb_w2);
	double w_bot = __dadd_ru(la.get_ub(), -lb_square_side);

	double top_min_side = w_top < ub_top_hrem ? w_top : ub_top_hrem;
	double bot_min_side = w_bot < lb_square_side ? w_bot : lb_square_side;
	double lb_R3 = (required_weight_for(la) - r1 - r2).get_lb();

	if(rectangle_size_bound::disk_satisfies_size_bound(ub_r3, top_min_side)) {
		// the weight requirement for the top rectangle
		double ub_weight_required_top = required_weight_for_bounded_size(w_top, ub_top_hrem);
		
		// additional cost for splitting
		ub_weight_required_top = __dadd_ru(ub_weight_required_top, ub_r3);
		
		double lb_weight_remaining = __dadd_rd(lb_R3, -ub_weight_required_top);
		if(lb_weight_remaining > 0.0) {
			if(can_recurse(w_bot, lb_square_side, ub_r3, lb_weight_remaining)) {
				return true;
			}
		}
	} else if(rectangle_size_bound::disk_satisfies_size_bound(ub_r3, bot_min_side)) {
		// the weight requirement for the bottom rectangle
		double ub_weight_required_bot = required_weight_for_bounded_size(w_bot, lb_square_side);

		// additional cost for splitting
		ub_weight_required_bot = __dadd_ru(ub_weight_required_bot, ub_r3);

		double lb_weight_remaining = __dadd_rd(lb_R3, -ub_weight_required_bot);
		if(lb_weight_remaining > 0.0) {
			if(can_recurse(w_top, ub_top_hrem, ub_r3, lb_weight_remaining)) {
				return true;
			}
		}
	} else {
		double ub_weight_required_top = required_weight_for(IV(w_top,w_top), IV(ub_top_hrem,ub_top_hrem)).get_ub();
		ub_weight_required_top = __dadd_ru(ub_weight_required_top, ub_r3);
		double lb_weight_remaining = __dadd_rd(lb_R3, -ub_weight_required_top);
		if(lb_weight_remaining > 0.0) {
			return can_recurse(w_bot, lb_square_side, ub_r3, lb_weight_remaining);
		}
	}

	return false;
}

bool __device__ circlecover::tight_rectangle::r1_r2_strategies(IV la, IV r1, IV r2, double ub_r3) {
	return r1_r2_strategies_simple(la, r1, r2, ub_r3) || r1_r2_strategies_stacked(la, r1, r2, ub_r3);
}

