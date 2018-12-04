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
#include "../rectangle_size_bound/values.hpp"
#include "../rectangle_size_bound/strategies.cuh"

bool __device__ circlecover::tight_rectangle::can_apply_size_bounded_covering(IV w, IV h, double ub_largest_disk, double lb_remaining_weight) {
	IV longer, shorter;

	if(definitely(w <= h)) {
		longer = h;
		shorter = w;
	} else if(definitely(w >= h)) {
		longer = w;
		shorter = h;
	} else {
		longer  = IV{w.get_lb() < h.get_lb() ? h.get_lb() : w.get_lb(), w.get_ub() < h.get_ub() ? h.get_ub() : w.get_ub()};
		shorter = IV{w.get_lb() < h.get_lb() ? w.get_lb() : h.get_lb(), w.get_ub() < h.get_ub() ? w.get_ub() : h.get_ub()};
	}

	if(!rectangle_size_bound::disk_satisfies_size_bound(ub_largest_disk, shorter.get_ub())) {
		return false;
	} else {
		double ub_area = __dmul_ru(longer.get_ub(), shorter.get_ub());
		return __dmul_ru(rectangle_size_bound::critical_ratio, ub_area) <= lb_remaining_weight;
	}
}

double __device__ circlecover::tight_rectangle::required_weight_for_bounded_size(double ub_w, double ub_h) {
	double ub_area = __dmul_ru(ub_w, ub_h);
	return __dmul_ru(rectangle_size_bound::critical_ratio, ub_area);
}

