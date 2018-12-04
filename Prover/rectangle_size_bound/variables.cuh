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

#ifndef CIRCLECOVER_RECTANGLE_SIZE_BOUND_VARIABLES_CUH_INCLUDED_
#define CIRCLECOVER_RECTANGLE_SIZE_BOUND_VARIABLES_CUH_INCLUDED_

#include "variables.hpp"
#include "../operations.cuh"
#include <algcuda/interval.cuh>

__device__ circlecover::rectangle_size_bound::Intermediate_values::Intermediate_values(const Variables& vars) {
	critical_ratio_sq = IV(__dmul_rd(critical_ratio,critical_ratio), __dmul_ru(critical_ratio,critical_ratio));
	R = vars.la * critical_ratio;

	for(int i = 0; i < 6; ++i) {
		R -= vars.radii[i];
	}

	if(R.get_lb() < 0.0) {
		R.set_lb(0.0);
	}

	IV additive = IV{critical_ratio,critical_ratio}.reciprocal() * sqrt(4.0 * critical_ratio_sq - 1.0);
	min_width_factor = sqrt(2.0 - additive);
	max_width_factor = sqrt(2.0 + additive);

	const double weight_bound_wall_building_den = __dsqrt_rd(__dadd_rd(1.0, __dsqrt_rd(__dadd_rd(1.0, -__ddiv_ru(0.25, critical_ratio_sq.get_lb())))));
	const double weight_bound_wall_building_factor = __dadd_rd(1.0, -__drcp_ru(weight_bound_wall_building_den));
	weight_bound_wall_building = __dmul_rd(0.5, __dmul_rd(weight_bound_wall_building_factor,weight_bound_wall_building_factor));
	phase_factor_wall_building = 4.0 * critical_ratio_sq - critical_ratio * sqrt(16.0*critical_ratio_sq-4.0);
	lb_w_infty_denom = __dadd_rd(1.0, -__dsqrt_ru(phase_factor_wall_building.get_ub()));
	width_fraction_incomplete = __drcp_ru(__dsqrt_rd(__dadd_rd(1.0, __dsqrt_rd(__dadd_rd(1.0, -__ddiv_ru(0.25, critical_ratio_sq.get_lb()))))));
}

#endif

