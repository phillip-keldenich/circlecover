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

#include "strategies.cuh"  // For documentation comments, see declarations in strategies.cuh.
#include "../tight_rectangle/strategies.cuh"


__device__ bool circlecover::rectangle_size_bound::can_recurse(double lb_weight_remaining, double ub_largest_disk_weight, double ub_w, double ub_h) {
	// the best covering ratio we can obtain with Lemma 3
	const double C = 195.0/256.0;
	// the smallest lambda from which Lemma 3 starts being better than Theorem 1
	const double tolerated_lambda = 2.0898841580413813901;
	
	// if there is no remaining weight:
	if(lb_weight_remaining <= 0.0) {
		// strictly negative weight remaining:
		// err on the side of caution and return false here
		if(lb_weight_remaining < 0.0) {
			return false;
		}

		// possibly no weight remaining: check for empty rectangle
		return (ub_w <= 0.0) | (ub_h <= 0.0);
	}

	// otherwise, if the rectangle is empty, return true
	if(ub_w <= 0.0 || ub_h <= 0.0) {
		return true;
	}

	// find maximum and minimum side length
	double side_min = ub_w < ub_h ? ub_w : ub_h;
	double side_max = ub_w < ub_h ? ub_h : ub_w;
	
	// first, try recursion using the strong size-bounded result (Lemma 4);
	// for that purpose, we try to enlarge the area:
	// compute the maximum area we CAN cover using the weight (rounding down)
	double sb_max_area = __ddiv_rd(lb_weight_remaining, critical_ratio);
	// compute the maximum side length of the shorter rectangle side (rounding down)
	double sb_max_side_min = __ddiv_rd(sb_max_area, side_max);
	if(sb_max_side_min < side_min) {
		// we cannot cover the area with the remaining weight, even using the strong size-bounded result;
		// recursion cannot work!
		return false;
	}

	if(sb_max_side_min > side_max) {
		// we can enlarge the smaller side to a length greater than the current longer side;
		// increase both side lengths to the square root of the area, making R' square
		sb_max_side_min = __dsqrt_rd(sb_max_area);
	}

	// check the size bound: scale weight-bound by square of shorter side length
	double scaled_size_bound = __dmul_rd(ub_disk_weight, __dmul_rd(sb_max_side_min, sb_max_side_min));
	if(scaled_size_bound >= ub_largest_disk_weight) {
		// size bound and weight bound work out
		return true;
	}

	// the strong size-bounded result does not work
	double ub_skew = __ddiv_ru(side_max, side_min);
	if(ub_skew <= tolerated_lambda) {
		// the worst-case result (Theorem 1) is the best that we have;
		// compute (an upper bound on) the critical ratio given by it
		double cr = tight_rectangle::critical_ratio(IV(__ddiv_rd(side_max, side_min),ub_skew)).get_ub();
		// check if we have the necessary weight
		double necessary_weight = __dmul_ru(cr, __dmul_ru(ub_w, ub_h));
		return necessary_weight <= lb_weight_remaining;
	} else {
		// try Lemma 3 first; scale the size bound to our largest disk weight
		double sigma_scaled = __ddiv_ru(ub_largest_disk_weight, __dmul_rd(side_min, side_min));
		sigma_scaled = __dmul_ru(sigma_scaled, sigma_scaled);
		double cr = __dmul_ru(0.5, __dsqrt_ru(__dadd_ru(__dsqrt_ru(__dadd_ru(sigma_scaled, 1.0)), 1.0)));
		// we cannot get better than C
		if(cr < C) { cr = C; }
		// compute and check necessary weight
		double necessary_weight = __dmul_ru(cr, __dmul_ru(ub_w, ub_h));
		if(necessary_weight <= lb_weight_remaining) {
			return true;
		}

		// try the worst-case bound
		necessary_weight = __dadd_ru(__dmul_ru(0.25, __dmul_ru(side_max,side_max)), __dmul_ru(0.5, __dmul_ru(side_min,side_min)));
		return necessary_weight <= lb_weight_remaining;
	}
}

__device__ double circlecover::rectangle_size_bound::bound_critical_ratio(double lb_weight_remaining, double ub_largest_disk_weight, double ub_w, double ub_h) {
	const double C = 195.0/256.0;

	if(lb_weight_remaining < 0.0) {
		// infinity if there is no remaining disks
		return DBL_MAX*DBL_MAX;
	}

	if(ub_w <= 0.0 || ub_h <= 0.0) {
		// empty rectangle
		return 0.0;
	}

	// find maximum and minimum side length
	double side_min = ub_w < ub_h ? ub_w : ub_h;
	double side_max = ub_w < ub_h ? ub_h : ub_w;
	double best_critical_ratio = DBL_MAX*DBL_MAX;
	double ub_area = __dmul_ru(ub_w, ub_h);

	// first, try using strong size-bounded result (Lemma 4) directly
	double required_side_length = __dsqrt_ru(__ddiv_ru(ub_largest_disk_weight, ub_disk_weight));
	if(side_min >= required_side_length) {
		double weight_required = __dmul_ru(critical_ratio, ub_area);
		if(weight_required <= lb_weight_remaining) {
			return critical_ratio;
		} else {
			return DBL_MAX*DBL_MAX;
		}
	} else {
		// otherwise, try using Lemma 4 after growing the rectangle.
		// this increases the critical ratio.
		if(required_side_length > side_max) {
			double square_area = __ddiv_rd(lb_weight_remaining, critical_ratio);
			double square_side_length = __dsqrt_rd(square_area);
			if(square_side_length >= required_side_length) {
				// cover a square of required_side_length at critical_ratio
				double actual_square_area   = __dmul_ru(required_side_length, required_side_length);
				double actual_square_weight = __dmul_ru(actual_square_area, critical_ratio);
				double lb_orig_area = __dmul_rd(ub_w, ub_h);
				best_critical_ratio = __ddiv_ru(actual_square_weight, lb_orig_area);
			}
		} else {
			double required_weight = __dmul_ru(critical_ratio, __dmul_ru(required_side_length, side_max));
			if(required_weight <= lb_weight_remaining) {
				double lb_orig_area = __dmul_rd(ub_w, ub_h);
				best_critical_ratio = __ddiv_ru(required_weight, lb_orig_area);
			}
		}
	}

	// second, try size bounded result for large size bound (Lemma 3)
	double scaled_sigma = __ddiv_ru(ub_largest_disk_weight, __dmul_rd(side_min,side_min));
	scaled_sigma = __dmul_ru(scaled_sigma, scaled_sigma);
	double eff_cor = __dmul_ru(0.5, __dsqrt_ru(__dadd_ru(__dsqrt_ru(__dadd_ru(scaled_sigma, 1.0)), 1.0)));
	if(eff_cor < C) {
		eff_cor = C;
	}
	if(eff_cor < best_critical_ratio) {
		double weight_required = __dmul_ru(eff_cor, ub_area);
		if(weight_required <= lb_weight_remaining) {
			best_critical_ratio = eff_cor;
		}
	}

	// finally, try worst-case result
 	IV skew(__ddiv_rd(side_max, side_min), __ddiv_ru(side_max, side_min));
	double cr = tight_rectangle::critical_ratio(skew).get_ub();
	if(cr < best_critical_ratio) {
		double weight_required = __dmul_ru(cr, ub_area);
		if(weight_required <= lb_weight_remaining) {
			best_critical_ratio = cr;
		}
	}

	return best_critical_ratio;
}

__device__ double circlecover::rectangle_size_bound::recursion_bound_size(double lb_weight_remaining, double ub_w, double ub_h) {
	const double C = 195.0/256.0;

	if(lb_weight_remaining <= 0.0) {
		if(lb_weight_remaining < 0.0) {
			return 0.0;
		}

		return (ub_w <= 0.0 || ub_h <= 0.0) ? DBL_MAX*DBL_MAX : 0.0;
	}

	if(ub_w <= 0.0 || ub_h <= 0.0) {
		return DBL_MAX*DBL_MAX;
	}

	double ub_area = __dmul_ru(ub_w, ub_h);
	double max_critical_ratio = __ddiv_rd(lb_weight_remaining, ub_area);
	double side_min = ub_w < ub_h ? ub_w : ub_h;
	double side_max = ub_w < ub_h ? ub_h : ub_w;
	IV skew(__ddiv_rd(side_max,side_min), __ddiv_ru(side_max,side_min));
	double critical_ratio_wc = tight_rectangle::critical_ratio(skew).get_ub();
	if(critical_ratio_wc <= max_critical_ratio) {
		// we can handle disks of any size
		return DBL_MAX*DBL_MAX;
	}

	// compute the maximum coverable area at efficiency critical_ratio
	double sb_max_area = __ddiv_rd(lb_weight_remaining, critical_ratio);
	double sb_max_side_min = __ddiv_rd(sb_max_area, side_max);

	if(sb_max_side_min < side_min) {
		// recursion is not possible
		return 0.0;
	}

	if(sb_max_side_min > side_max) {
		// we can enlarge the smaller side to a length greater than the current longer side; increase both side lengths
		sb_max_side_min = __dsqrt_rd(sb_max_area);
	}

	// compute the largest size bound possible using the strong size-bounded result
	double best_size_bound = __dmul_rd(ub_disk_weight, __dmul_rd(sb_max_side_min, sb_max_side_min));

	// we can try to use the large size-bounded result
	if(max_critical_ratio >= C) {
		double lb_lambda = __dmul_rd(2.0, max_critical_ratio);
		lb_lambda = __dadd_rd(lb_lambda, __dsqrt_rd(__dadd_rd(__dmul_rd(4.0, __dmul_rd(max_critical_ratio,max_critical_ratio)), -2.0)));
		double lb_sigma = __dmul_rd(max_critical_ratio, __dadd_rd(lb_lambda, -__ddiv_ru(2.0, lb_lambda)));
		double lb_scaled_sigma = __dmul_rd(lb_sigma, __dmul_rd(side_min,side_min));
		if(lb_scaled_sigma >= best_size_bound) {
			best_size_bound = lb_scaled_sigma;
		}
	}

	return best_size_bound;
}


__device__ double circlecover::rectangle_size_bound::bound_worst_case_ratio(IV w, IV h) {
	const double C = 195.0/256.0;

	if(w.get_lb() <= 0.0 || h.get_lb() <= 0.0) {
		return DBL_MAX*DBL_MAX;
	}

	double ub_w_by_h = __ddiv_ru(w.get_ub(), h.get_lb());
	double ub_h_by_w = __ddiv_ru(h.get_ub(), w.get_lb());
	double ub_skew = (ub_w_by_h < ub_h_by_w ? ub_h_by_w : ub_w_by_h);
	double result = __dadd_ru(__dmul_ru(0.25, ub_skew), __dmul_ru(0.5, __drcp_ru(ub_skew))); 

	if(result < C) { result = C; }
	return result;
}

__device__ bool circlecover::rectangle_size_bound::can_recurse_worst_case(double lb_weight_remaining, double ub_w, double ub_h) {
	const double C = 195.0/256.0;
	const double tolerated_lambda = 2.0898841580413813901;

	double ub_lambda, l, s;
	if(ub_w < ub_h) {
		ub_lambda = __ddiv_ru(ub_h, ub_w);
		l = ub_h;
		s = ub_w;
	} else {
		ub_lambda = __ddiv_ru(ub_w, ub_h);
		l = ub_w;
		s = ub_h;
	}

	if(ub_lambda <= tolerated_lambda) {
		double ub_A = __dmul_ru(ub_w, ub_h);
		return __dmul_ru(ub_A, C) <= lb_weight_remaining;
	} else {
		l = __dmul_ru(l,l);
		s = __dmul_ru(s,s);
		double ub_weight_required = __dadd_ru(__dmul_ru(0.5, s), __dmul_ru(0.25, l));
		return ub_weight_required <= lb_weight_remaining;
	}
}

