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
using namespace circlecover::rectangle_size_bound;

__device__ bool circlecover::rectangle_size_bound::r1_in_corner_wall_building_recursion(const Variables& vars, const Intermediate_values& vals) {
	// we place r1 covering a square in the lower left corner
	IV w1 = sqrt(2.0 * vars.radii[0]);
	// the weight we are saving by placing r1 in this efficient way
	double extra_weight_r1 = __dmul_rd(__dadd_rd(__dmul_rd(2.0, critical_ratio), -1.0), vars.radii[0].get_lb());
	IV w_B = vars.la - w1;
	IV h_A = 1.0 - w1;

	IV min_A{h_A.get_lb() < w1.get_lb() ? h_A.get_lb() : w1.get_lb(), h_A.get_ub() < w1.get_ub() ? h_A.get_ub() : w1.get_ub()};
	IV max_A{h_A.get_lb() < w1.get_lb() ? w1.get_lb() : h_A.get_lb(), h_A.get_ub() < w1.get_ub() ? w1.get_ub() : h_A.get_ub()};
	IV min_A_sq = min_A.square();
	IV max_A_sq = max_A.square();
	IV weight_bound_A = ub_disk_weight * min_A_sq;

	// if r6 is below the weight bound for recursion on A, we can use recursion on A directly
	// and pay at most r6 additional weight
	double extra_cost_A = weight_bound_A.get_ub();
	if(extra_cost_A >= vars.radii[5].get_ub()) {
		extra_cost_A = vars.radii[5].get_ub();
	}

	// in case (I), we recurse and spend weight_bound_A extra cost on A which must be covered by the gained weight from r1
	if(weight_bound_A.get_lb() <= 0.0 || extra_cost_A > extra_weight_r1) {
		return false;
	}

	// in either case, r2 has to fit into B
	if(possibly(w_B < 1.0) && !disk_satisfies_size_bound(vars.radii[1].get_ub(), w_B.get_lb())) {
		// if it does not: we can try getting rid of it in several ways; one way is to place r2,r3 or r2,r3,r4 at the bottom of B
		double extra_weight23 = -DBL_MAX*DBL_MAX;

		{ // try r2,r3
			double lb_h23 = two_disks_maximize_height(vars.radii[1], vars.radii[2], w_B.get_ub());
			double lb_hrem_B = __dadd_rd(1.0, -lb_h23); 
			double min_B = lb_hrem_B < w_B.get_lb() ? lb_hrem_B : w_B.get_lb();
			
			if(disk_satisfies_size_bound(vars.radii[3].get_ub(), min_B)) {
				// r4 is small enough to be used in size-bounded recursion on B
				double ub_weight23 = __dadd_ru(vars.radii[1].get_ub(), vars.radii[2].get_ub());
				double lb_a23 = __dmul_rd(lb_h23, w_B.get_lb());
				extra_weight23 = __dadd_rd(__dmul_rd(critical_ratio, lb_a23), -ub_weight23);
			}
		}
		
		double extra_weight234 = -DBL_MAX*DBL_MAX;
		{ // try r2,r3,r4
			double lb_h234 = three_disks_maximize_height(vars.radii[1], vars.radii[2], vars.radii[3], w_B.get_ub());
			double lb_hrem_B = __dadd_rd(1.0, -lb_h234); 
			double min_B = lb_hrem_B < w_B.get_lb() ? lb_hrem_B : w_B.get_lb();

			if(disk_satisfies_size_bound(vars.radii[4].get_ub(), min_B)) {
				// r5 is small enough to be used in size-bounded recursion on B
				double ub_weight234 = __dadd_ru(vars.radii[1].get_ub(), __dadd_ru(vars.radii[2].get_ub(), vars.radii[3].get_ub()));
				double lb_a234 = __dmul_rd(lb_h234, w_B.get_lb());
				extra_weight234 = __dadd_rd(__dmul_rd(critical_ratio, lb_a234), -ub_weight234);
			}
		}

		double better = extra_weight23 > extra_weight234 ? extra_weight23 : extra_weight234;
		extra_weight_r1 = __dadd_rd(extra_weight_r1, better);
		if(extra_weight_r1 <= 0.0) {
			return false;
		}
	}

	// in case (II), r6 must be small enough to participate in wall-building; ideally, we can even use it in wall-building along the shorter side of A
	double weight_bound_wb_shorter = __dmul_rd(vals.weight_bound_wall_building, min_A_sq.get_lb());
	double weight_bound_wb_longer =  __dmul_rd(vals.weight_bound_wall_building, max_A_sq.get_lb());
	if(weight_bound_wb_longer < vars.radii[5].get_ub()) {
		return false;
	}
	bool can_use_shorter_side = (weight_bound_wb_shorter >= vars.radii[5].get_ub());
	
	// check that r1 gives us enough extra weight to deal with covering an additional row
	double extra_cost_A_wall_building = __dmul_ru(__dmul_ru(critical_ratio, can_use_shorter_side ? min_A.get_ub() : max_A.get_ub()), __dsqrt_ru(__dmul_ru(2.0, vars.radii[5].get_ub())));
	if(extra_cost_A_wall_building > extra_cost_A) {
		extra_cost_A = extra_cost_A_wall_building;
	}

	if(extra_cost_A > extra_weight_r1) {
		return false;
	}

	// if we are definitely in case (I) and have enough extra weight, we are done
	if(vars.radii[5].get_ub() <= weight_bound_A.get_lb()) {
		return true;
	}

	// make sure that we do not drop too much weight to B
	// first compute a lower bound on the disk weight starting from r_6
	double lb_R6 = __dmul_rd(critical_ratio, vars.la.get_lb());
	for(int i = 0; i < 5; ++i) {
		lb_R6 = __dadd_rd(lb_R6, -vars.radii[i].get_ub());
	}

	// the maximal width of an incomplete row
	double ub_max_width_incomplete = __dmul_ru(can_use_shorter_side ? min_A.get_ub() : max_A.get_ub(), vals.width_fraction_incomplete);
	
	// begin by computing w_w^\infty, i.e., using all disks starting from r_6
	double ub_dropped_weight_infty_numerator = __dmul_ru(critical_ratio, __dmul_ru(ub_max_width_incomplete, __dsqrt_ru(__dmul_ru(2.0, vars.radii[5].get_ub()))));
	double ub_dropped_weight_infty = __ddiv_ru(ub_dropped_weight_infty_numerator, vals.lb_w_infty_denom);
	double non_dropped_weight_infty = __dadd_rd(lb_R6, -ub_dropped_weight_infty);
	double ub_area_A = __dmul_ru(w1.get_ub(), h_A.get_ub());
	double ub_weight_needed_A = __dmul_ru(critical_ratio, ub_area_A); 
	if(non_dropped_weight_infty >= ub_weight_needed_A) {
		return true;
	}

	// otherwise, we have to deal with up to critical_ratio * A dropped weight from the beginning
	double ub_dropped_weight = ub_weight_needed_A;

	// the size of a disk in the current phase
	double ub_current_disk_weight = vars.radii[5].get_ub();
	while(ub_current_disk_weight > weight_bound_A.get_lb()) {
		double ub_dropped_weight_current = __dmul_ru(critical_ratio, __dmul_ru(ub_max_width_incomplete, __dsqrt_ru(__dmul_ru(2.0, ub_current_disk_weight))));
		ub_dropped_weight = __dadd_ru(ub_dropped_weight, ub_dropped_weight_current);
		ub_current_disk_weight = __dmul_ru(ub_current_disk_weight, vals.phase_factor_wall_building.get_ub());
	}
	
	double non_dropped_weight = __dadd_rd(lb_R6, -ub_dropped_weight);
	return non_dropped_weight >= ub_weight_needed_A;
}

