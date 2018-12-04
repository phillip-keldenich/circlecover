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

__device__ bool circlecover::rectangle_size_bound::shortcut_r1_r2_opposite_corners_recursion(IV la, IV r1, IV r2, IV r3) {
	IV w1{__dsqrt_rd(__dmul_rd(2.0, r1.get_lb())), __dsqrt_ru(__dmul_ru(2.0, r1.get_ub()))};
	IV w2{__dsqrt_rd(__dmul_rd(2.0, r2.get_lb())), __dsqrt_ru(__dmul_ru(2.0, r2.get_ub()))};

	IV combined = w1+w2;
	if(combined.get_ub() < 1.0) {
		// definitely not intersecting and not covering all height
		IV w_A = la - w2;
		IV w_B = la - w1;
		IV h_A = 1.0 - w1;
		IV h_B = w1;

		double lb_min_A = w_A.get_lb();
		if(h_A.get_lb() < lb_min_A) { lb_min_A = h_A.get_lb(); }
		double lb_min_B = w_B.get_lb();
		if(h_B.get_lb() < lb_min_B) { lb_min_B = h_B.get_lb(); }
		double lb_min_AB = lb_min_A < lb_min_B ? lb_min_A : lb_min_B;
		if(!disk_satisfies_size_bound(r3.get_ub(), lb_min_AB)) {
			return false;
		}

		IV w_C = w2;
		IV h_C = 1.0 - combined;

		double lb_min_C = w_C.get_lb();
		if(h_C.get_lb() < lb_min_C) { lb_min_C = h_C.get_lb(); }
		if(disk_satisfies_size_bound(r3.get_ub(), lb_min_C)) {
			if(__dmul_ru(2.0, r3.get_ub()) > __dmul_rd(__dadd_rd(__dmul_rd(2.0, critical_ratio), -1.0), __dadd_rd(r1.get_lb(), r2.get_lb()))) {
				return false;
			}
		} else {
			double lb_remaining_before_C = __dmul_rd(__dadd_rd(__dmul_rd(2.0, critical_ratio), -1.0), __dadd_rd(r1.get_lb(), r2.get_lb()));
			lb_remaining_before_C = __dadd_rd(lb_remaining_before_C, -__dmul_ru(2.0, r3.get_ub()));
			double ub_critical_ratio_C = bound_worst_case_ratio(w_C, h_C);
			double ub_area_C = __dmul_ru(w_C.get_ub(), h_C.get_ub());
			double extra_weight_C = __dmul_ru(__dadd_ru(ub_critical_ratio_C, -critical_ratio), ub_area_C);
			if(lb_remaining_before_C < extra_weight_C) {
				return false;
			}
		}

		return true;
	}

/*	if(combined.get_lb() >= 1.0 && combined.get_ub() < la.get_lb()) {
		// definitely not intersecting, but covering all height
		IV w_A = la - w2;
		IV h_A = 1.0 - w1;
		IV w_B = la - w1;
		IV h_B = 1.0 - w2;

		double lb_min_A = w_A.get_lb();
		if(h_A.get_lb() < lb_min_A) { lb_min_A = h_A.get_lb(); }
		double lb_min_B = w_B.get_lb();
		if(h_B.get_lb() < lb_min_B) { lb_min_B = h_B.get_lb(); }
		double lb_min_AB = lb_min_A < lb_min_B ? lb_min_A : lb_min_B;
		if(!disk_satisfies_size_bound(r3.get_ub(), lb_min_AB)) {
			return false;
		}

		IV w_C = la-combined;
		IV h_C = combined - 1.0;
		double lb_min_C = w_C.get_lb();
		if(h_C.get_lb() < lb_min_C) { lb_min_C = h_C.get_lb(); }
		if(lb_min_C <= 0.0) { return false; }

		if(disk_satisfies_size_bound(r3.get_ub(), lb_min_C)) {
			if(__dmul_ru(2.0, r3.get_ub()) > __dmul_rd(__dadd_rd(__dmul_rd(2.0, critical_ratio), -1.0), __dadd_rd(r1.get_lb(), r2.get_lb()))) {
				return false;
			}
		} else {
			double lb_remaining_before_C = __dmul_rd(__dadd_rd(__dmul_rd(2.0, critical_ratio), -1.0), __dadd_rd(r1.get_lb(), r2.get_lb()));
			lb_remaining_before_C = __dadd_rd(lb_remaining_before_C, -__dmul_ru(2.0, r3.get_ub()));
			double ub_critical_ratio_C = bound_worst_case_ratio(w_C, h_C);
			double ub_area_C = __dmul_ru(w_C.get_ub(), h_C.get_ub());
			double extra_weight_C = __dmul_ru(__dadd_ru(ub_critical_ratio_C, -critical_ratio), ub_area_C);
			if(lb_remaining_before_C < extra_weight_C) {
				return false;
			}
		}

		return true;
	}

	if(combined.get_lb() >= la.get_ub()) {
		// definitely intersecting
		IV w_A = la - w2;
		IV h_A = 1.0 - w1;
		IV w_B = la - w1;
		IV h_B = 1.0 - w2;

		double lb_min_A = w_A.get_lb();
		if(h_A.get_lb() < lb_min_A) { lb_min_A = h_A.get_lb(); }
		double lb_min_B = w_B.get_lb();
		if(h_B.get_lb() < lb_min_B) { lb_min_B = h_B.get_lb(); }
		double lb_min_AB = lb_min_A < lb_min_B ? lb_min_A : lb_min_B;
		if(!disk_satisfies_size_bound(r3.get_ub(), lb_min_AB)) {
			return false;
		}

		IV w_C = combined - la;
		IV h_C = combined - 1.0;
		double ub_area_C = __dmul_ru(w_C.get_ub(), h_C.get_ub());
		double ub_weight_C = __dmul_ru(critical_ratio, ub_area_C);
		double extra_weight_r12 = __dmul_rd(__dadd_rd(__dmul_rd(2.0, critical_ratio), -1.0), combined.get_lb());
		double extra_weight_after_A = __dadd_rd(extra_weight_r12, -r3.get_ub());
		return extra_weight_after_A >= ub_weight_C;
	} */

	return false;
}

namespace {
	struct Shorter_longer_side {
		IV shorter, longer;
	};
}

static __device__ Shorter_longer_side compute_shorter_longer(IV w, IV h) {
	UB cmp = (w <= h);
	if(cmp.is_certain()) {
		if(cmp.get_lb()) {
			return {w,h};
		} else {
			return {h,w};
		}
	}

	const double a = w.get_lb();
	const double b = w.get_ub();
	const double c = h.get_lb();
	const double d = h.get_ub();

	Shorter_longer_side result;
	if(a < c) {
		result.shorter.set_lb(a);
		result.longer.set_lb(c);
	} else {
		result.shorter.set_lb(c);
		result.longer.set_lb(a);
	}

	if(b < d) {
		result.shorter.set_ub(b);
		result.longer.set_ub(d);
	} else {
		result.shorter.set_ub(d);
		result.longer.set_ub(b);
	}

	return result;
}

namespace {
	struct Size_bound_check {
		double ub_extra_cost_bigger_min, ub_extra_cost_smaller_min;
		bool works;
	};
}

static __device__ Size_bound_check r1_r2_opposite_corners_recursion_check_size_bounds(const Variables& vars, const Intermediate_values& vals, IV w_A, IV h_A, IV w_B, IV h_B, int start_from_disk) {
	double lb_min_A = w_A.get_lb();
	if(h_A.get_lb() < lb_min_A) { lb_min_A = h_A.get_lb(); }
	double lb_min_B = w_B.get_lb();
	if(h_B.get_lb() < lb_min_B) { lb_min_B = h_B.get_lb(); }
	
	double lb_area_A = __dmul_rd(w_A.get_lb(), h_A.get_lb());
	double lb_area_B = __dmul_rd(w_B.get_lb(), h_B.get_lb());

	double bigger_min  = lb_min_A < lb_min_B ? lb_min_B : lb_min_A;
	double smaller_min = lb_min_A < lb_min_B ? lb_min_A : lb_min_B;

	if(start_from_disk < 6) {
		if(!disk_satisfies_size_bound(vars.radii[start_from_disk].get_ub(), bigger_min)) {
			return { 0.0, 0.0, false };
		}

		double bigger_min_area = lb_min_A < lb_min_B ? lb_area_B : lb_area_A;
		double bigger_min_weight = __dmul_rd(critical_ratio, bigger_min_area);
		double ub_weight_in_bigger = 0.0;

		// check how many disks we can definitely put into the bigger part
		for(; start_from_disk < 6; ++start_from_disk) {
			ub_weight_in_bigger = __dadd_ru(ub_weight_in_bigger, vars.radii[start_from_disk].get_ub());
			if(ub_weight_in_bigger >= bigger_min_weight) {
				break;
			}
		}
	}

	if(start_from_disk < 6) {
		// at some point, we have to stop adding explicit disks to the region with bigger min side length
		if(!disk_satisfies_size_bound(vars.radii[start_from_disk].get_ub(), smaller_min)) {
			return { 0.0, 0.0, false };
		}

		double ub_extra_cost_bigger_min = vars.radii[start_from_disk].get_ub();
		double smaller_min_area = lb_min_A < lb_min_B ? lb_area_A : lb_area_B;
		double smaller_min_weight = __dmul_rd(critical_ratio, smaller_min_area);
		double ub_weight_in_smaller = 0.0;
		
		for(; start_from_disk < 6; ++start_from_disk) {
			ub_weight_in_smaller = __dadd_ru(ub_weight_in_smaller, vars.radii[start_from_disk].get_ub());
			if(ub_weight_in_smaller >= smaller_min_weight) {
				break;
			}
		}

		if(start_from_disk < 6) {
			return { ub_extra_cost_bigger_min, vars.radii[start_from_disk].get_ub(), true };
		} else {
			double ub_r7 = vars.radii[5].get_ub();
			if(vals.R.get_ub() < ub_r7) { ub_r7 = vals.R.get_ub(); }
			return { ub_extra_cost_bigger_min, ub_r7, true };
		}
	} else {
		double ub_r7 = vars.radii[5].get_ub();
		if(vals.R.get_ub() < ub_r7) { ub_r7 = vals.R.get_ub(); }
		if(!disk_satisfies_size_bound(ub_r7, smaller_min)) {
			return { 0.0, 0.0, false };
		}

		return { ub_r7, ub_r7, true };
	}
}

__device__ bool  circlecover::rectangle_size_bound::r1_r2_opposite_corners_recursion(const Variables& vars, const Intermediate_values& vals) {
	const IV& r1 = vars.radii[0];
	const IV& r2 = vars.radii[1];

	IV w1{__dsqrt_rd(__dmul_rd(2.0, r1.get_lb())), __dsqrt_ru(__dmul_ru(2.0, r1.get_ub()))};
	IV w2{__dsqrt_rd(__dmul_rd(2.0, r2.get_lb())), __dsqrt_ru(__dmul_ru(2.0, r2.get_ub()))};
	IV combined = w1+w2;
	double lb_extra_weight_r1_r2 = __dmul_rd(__dadd_rd(__dmul_rd(2.0, critical_ratio), -1.0), __dadd_rd(r1.get_lb(), r2.get_lb()));

	if(combined.get_ub() < 1.0) {
		// definitely not intersecting and not covering all height
		IV w_A = vars.la - w2;
		IV w_B = vars.la - w1;
		IV h_A = 1.0 - w1;
		IV h_B = w1;
		IV w_C = w2;
		IV h_C = 1.0 - combined;

		Shorter_longer_side sides_C = compute_shorter_longer(w_C, h_C);
		IV r_bounds_rect_C = compute_efficient_rectangle_cover_weight_range(sides_C.longer.get_lb(), sides_C.shorter);

		int max_disk;
		for(max_disk = 2; max_disk < 6; ++max_disk) {
			if(r_bounds_rect_C.get_lb() > vars.radii[max_disk].get_lb() || vars.radii[max_disk].get_ub() > r_bounds_rect_C.get_ub()) {
				break;
			}

			// check the covered strip length
			IV strip_length_covered = 4.0*vars.radii[max_disk]-sides_C.shorter.square();
			if(possibly(strip_length_covered <= 0.0)) {
				break;	
			}

			// we can place this disk on C with efficiency at least critical_ratio
			strip_length_covered = sqrt(strip_length_covered);
			sides_C.longer -= strip_length_covered;
			sides_C.longer.tighten_lb(0.0);
			if(sides_C.longer.empty()) {
				// C disappeared (we covered it using efficiency critical_ratio)
				Size_bound_check sb = r1_r2_opposite_corners_recursion_check_size_bounds(vars, vals, w_A, h_A, w_B, h_B, max_disk+1);
				if(!sb.works) {
					return false;
				}
				return sb.ub_extra_cost_bigger_min <= lb_extra_weight_r1_r2;
			}

			// recompute shorter/longer side
			sides_C = compute_shorter_longer(sides_C.shorter, sides_C.longer);
			r_bounds_rect_C = compute_efficient_rectangle_cover_weight_range(sides_C.longer.get_lb(), sides_C.shorter);
		}

		Size_bound_check sb_first = r1_r2_opposite_corners_recursion_check_size_bounds(vars, vals, w_A, h_A, w_B, h_B, max_disk);
		if(sb_first.works) {
			double ub_remaining_disk_weight = sb_first.ub_extra_cost_smaller_min;
			if(disk_satisfies_size_bound(ub_remaining_disk_weight, sides_C.shorter.get_lb())) {
				// the (remaining part of) region C uses size-bounded recursion; we only have to pay for splitting on A and B
				return __dadd_ru(sb_first.ub_extra_cost_smaller_min, sb_first.ub_extra_cost_bigger_min) <= lb_extra_weight_r1_r2;
			}

			// we have to use worst-case recursion on region C; we have to pay for splitting on A and B and for the worse critical ratio on C;
			// first try to put some bigger disks into C
		}

		double min_area_C   = __dmul_rd(w_C.get_lb(), h_C.get_lb());
		double critical_ratio_C = bound_worst_case_ratio(w_C, h_C);
		double min_weight_C = __dmul_rd(critical_ratio_C, min_area_C);
		double big_disks_in_C = 0.0;
		for(; max_disk < 6; ++max_disk) {
			big_disks_in_C = __dadd_ru(big_disks_in_C, vars.radii[max_disk].get_ub());
			if(big_disks_in_C >= min_weight_C) {
				break;
			}
		}

		// we can put all disks up to (not including) max_disk into C
		Size_bound_check sb_second = r1_r2_opposite_corners_recursion_check_size_bounds(vars, vals, w_A, h_A, w_B, h_B, max_disk);
		if(sb_second.works) {
			// what we pay for splitting into A, B, C
			double extra_cost_splitting = __dadd_ru(sb_second.ub_extra_cost_smaller_min, sb_second.ub_extra_cost_bigger_min);
			// what we pay for covering C inefficiently
			double extra_cost_C = __dmul_ru(__dadd_ru(critical_ratio_C, -critical_ratio), __dmul_ru(w_C.get_ub(), h_C.get_ub()));
			return __dadd_ru(extra_cost_splitting, extra_cost_C) <= lb_extra_weight_r1_r2;
		}
	}

	return false;
}

