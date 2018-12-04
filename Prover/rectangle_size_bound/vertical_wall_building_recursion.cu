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

__device__ bool circlecover::rectangle_size_bound::vertical_wall_building_recursion(const Variables& vars, const Intermediate_values& vals) {
	// the highest possible width of a column is sqrt(2) * r_6
	double ub_column_width = __dsqrt_ru(__dmul_ru(vars.radii[5].get_ub(), 2.0));
	double lb_rem_width    = __dadd_rd(vars.la.get_lb(), ub_column_width);

	// check there is remaining weight
	if(vars.radii[5].get_lb() <= 0 || vals.R.get_lb() <= 0) {
		return false;
	}

	// check that r_1 fits in the remainder if we build a column; if we can build a wall, this is the only thing that has to hold
	if(lb_rem_width < 1.0 && !disk_satisfies_size_bound(vars.radii[0].get_ub(), lb_rem_width)) {
		return false;
	}

	// check that r6 is small enough for wall building
	if(vars.radii[5].get_ub() > vals.weight_bound_wall_building) {
		return false;
	}

	// run through phases - 32 phases should really suffice...
	double lb_R6 = __dadd_rd(vars.radii[5].get_lb(), vals.R.get_lb());
	double ub_disk_weight_below_phase = __dmul_ru(vars.radii[5].get_ub(), vals.phase_factor_wall_building.get_ub());
	double ub_weight_in_phases = __dmul_ru(critical_ratio, __dsqrt_ru(__dmul_ru(2.0, vars.radii[5].get_ub())));

	for(int k = 0; k < 32; ++k) {
		double lb_total_weight_below_phase = __dadd_rd(lb_R6, -ub_weight_in_phases);
		if(lb_total_weight_below_phase <= 0) {
			break;
		}

		// an upper bound on the width needed for disks below ub_disk_weight_below_phase
		double ub_width_below_phase = bound_required_height(ub_disk_weight_below_phase);
		double ub_req_weight_below_phase = __dmul_ru(critical_ratio, ub_width_below_phase);

		// is there enough weight for this phase?
		if(ub_req_weight_below_phase <= lb_total_weight_below_phase) {
			double ub_actual_weight_used = __dadd_ru(ub_req_weight_below_phase, ub_disk_weight_below_phase);
			double ub_actual_strip_width = __ddiv_ru(ub_actual_weight_used, critical_ratio);
			double lb_actual_rem_width   = __dadd_rd(vars.la.get_lb(), ub_actual_strip_width);

			// check that r1 fits if we recurse like this
			if(lb_actual_rem_width >= 1.0 || disk_satisfies_size_bound(vars.radii[0].get_ub(), lb_actual_rem_width)) {
				return true;
			}
		}

		// go to the next phase
		ub_weight_in_phases = __dadd_ru(ub_weight_in_phases, __dmul_ru(critical_ratio, __dsqrt_ru(__dmul_ru(2.0, ub_disk_weight_below_phase))));
		ub_disk_weight_below_phase = __dmul_ru(ub_disk_weight_below_phase, vals.phase_factor_wall_building.get_ub());
	}

	return false;
}

