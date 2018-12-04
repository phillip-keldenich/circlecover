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

static inline __device__ bool r1_large(const Variables& vars, IV R) {
	double lb_w1 = __dsqrt_rd(__dadd_rd(__dmul_rd(4.0, vars.radii[0].get_lb()), -1.0));
	double ub_wrem = __dadd_ru(vars.la.get_ub(), -lb_w1);
	double ub_wrem_sq = __dmul_ru(ub_wrem, ub_wrem);

	double lb_h4 = __dadd_rd(__dmul_rd(4.0, vars.radii[3].get_lb()), -ub_wrem_sq);
	if(lb_h4 <= 0.0) {
		return false;
	}
	lb_h4 = __dsqrt_rd(lb_h4);
	double lb_h3 = __dsqrt_rd(__dadd_rd(__dmul_rd(4.0, vars.radii[2].get_lb()), -ub_wrem_sq));
	double lb_h2 = __dsqrt_rd(__dadd_rd(__dmul_rd(4.0, vars.radii[1].get_lb()), -ub_wrem_sq));

	double htot = __dadd_rd(lb_h2, __dadd_rd(lb_h3, lb_h4));
	if(htot >= 1.0) {
		return true;
	}

	double ub_hrem = __dadd_ru(1.0, -htot);
	double ub_r5 = vars.radii[3].get_ub() < R.get_ub() ? vars.radii[3].get_ub() : R.get_ub();
	
	return can_recurse(ub_wrem, ub_hrem, ub_r5, R.get_lb());
}

static inline __device__ bool lambda_small(const Variables& vars, IV R) {
	double lb_square_side = __dsqrt_rd(__dmul_rd(2.0, vars.radii[0].get_lb()));

	if(lb_square_side >= 1.0) {
		return r1_large(vars, R);
	}

	double ub_wrem = __dadd_ru(vars.la.get_ub(), -lb_square_side);
	double ub_hrem = __dadd_ru(1.0, -lb_square_side);

	double lb_h2 = __dadd_rd(__dmul_rd(4.0, vars.radii[1].get_lb()), -__dmul_ru(ub_wrem,ub_wrem));
	double lb_w4 = __dadd_rd(__dmul_rd(4.0, vars.radii[3].get_lb()), -__dmul_ru(ub_hrem,ub_hrem));

	if(lb_h2 <= 0.0 || lb_w4 <= 0.0) {
		return false;
	}

	lb_h2 = __dsqrt_rd(lb_h2);
	lb_w4 = __dsqrt_rd(lb_w4);
	if(lb_w4 > lb_square_side) {
		lb_w4 = lb_square_side;
	}
	if(lb_h2 > lb_square_side) {
		lb_h2 = lb_square_side;
	}

	double ub_h = __dadd_ru(1.0, -lb_h2);
	double ub_w = __dadd_ru(vars.la.get_ub(), -lb_w4);
	double a = __dadd_ru(__dmul_ru(ub_h,ub_h), __dmul_ru(ub_w,ub_w));
	return __dmul_rd(4.0, vars.radii[2].get_lb()) >= a;
}

static inline __device__ bool lambda_big(const Variables& vars, IV R) {
	double lb_w;

	{
		double lb_w12 = rectangle_size_bound::two_disks_maximize_height(vars.radii[0], vars.radii[1], 1.0);
		double lb_w34 = rectangle_size_bound::two_disks_maximize_height(vars.radii[2], vars.radii[3], 1.0);
		lb_w = __dadd_rd(lb_w12, lb_w34);
	}

	{
		double lb_w14 = rectangle_size_bound::two_disks_maximize_height(vars.radii[0], vars.radii[3], 1.0);
		double lb_w23 = rectangle_size_bound::two_disks_maximize_height(vars.radii[1], vars.radii[2], 1.0);
		double lb_1423 = __dadd_rd(lb_w14, lb_w23);
		if(lb_1423 > lb_w) { lb_w = lb_1423; }
	}

	{
		double lb_w13 = rectangle_size_bound::two_disks_maximize_height(vars.radii[0], vars.radii[2], 1.0);
		double lb_w24 = rectangle_size_bound::two_disks_maximize_height(vars.radii[1], vars.radii[3], 1.0);
		double lb_1324 = __dadd_rd(lb_w13, lb_w24);
		if(lb_1324 > lb_w) { lb_w = lb_1324; }
	}

	if(lb_w <= 0.0) { return false; }
	double ub_r5 = R.get_ub() < vars.radii[3].get_ub() ? R.get_ub() : vars.radii[3].get_ub();
	return can_recurse(__dadd_ru(vars.la.get_ub(), -lb_w), 1.0, ub_r5, R.get_lb());
}

bool __device__ circlecover::tight_rectangle::two_by_two_strategies(const Variables& vars, IV R) {
	return lambda_small(vars, R) || lambda_big(vars, R);
}

