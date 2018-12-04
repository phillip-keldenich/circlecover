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

bool __device__ circlecover::tight_rectangle::can_recurse(IV w, IV h, double lb_remaining_weight) {
	IV cr = critical_ratio(w,h);
	double ub_area = __dmul_ru(w.get_ub(), h.get_ub());
	return __dmul_ru(ub_area, cr.get_ub()) <= lb_remaining_weight;
}

bool __device__ circlecover::tight_rectangle::can_recurse(double ub_w, double ub_h, double ub_largest_disk, double lb_remaining_weight) {
	// normal recursion
	double ub_critical_ratio = critical_ratio(IV(ub_w,ub_w), IV(ub_h,ub_h)).get_ub();
	double ub_area = __dmul_ru(ub_w, ub_h);

	// using corollary to theorem 1, we may achieve a lower critical ratio
	// normalize ub_largest_disk to a height of 1.0
	double min_side = ub_w < ub_h ? ub_w : ub_h;
	double ub_scale_fact = __drcp_ru(min_side);
	double ub_scale_fact_sq = __dmul_ru(ub_scale_fact, ub_scale_fact);
	double ub_largest_disk_norm = __dmul_ru(ub_scale_fact_sq, ub_largest_disk);

	// compute the critical ratio we achieve by the corollary
	double ub_critical_ratio_cor = __dmul_ru(0.5, __dsqrt_ru(__dadd_ru(__dsqrt_ru(__dadd_ru(__dmul_ru(ub_largest_disk_norm, ub_largest_disk_norm), 1.0)), 1.0)));
	if(ub_critical_ratio_cor < 195.0/256) {
		ub_critical_ratio_cor = 195.0/256;
	}
	
	// if that critical ratio is better than ub_critical_ratio, use it instead
	if(ub_critical_ratio > ub_critical_ratio_cor) {
		ub_critical_ratio = ub_critical_ratio_cor;
	}

	// check whether we can use the rectangle_size_bound critical ratio
	if(ub_largest_disk_norm <= rectangle_size_bound::ub_disk_weight) {
		ub_critical_ratio = rectangle_size_bound::critical_ratio;
	}

	double ub_weight_needed = __dmul_ru(ub_critical_ratio, ub_area);
	return lb_remaining_weight >= ub_weight_needed;
}

