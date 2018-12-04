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

__device__ circlecover::IV circlecover::rectangle_size_bound::compute_efficient_rectangle_cover_weight_range(double lb_w, IV h) {
	double result_ub_wmax = __dmul_rd(critical_ratio, __dmul_rd(h.get_lb(), lb_w));
	double result_lb_skew_esq_hsq = __dmul_ru(__dmul_ru(critical_ratio, critical_ratio), __dmul_ru(h.get_ub(), h.get_ub()));
	double result_ub_skew_esq_hsq = __dmul_rd(__dmul_rd(critical_ratio, critical_ratio), __dmul_rd(h.get_lb(), h.get_lb()));
	double result_sqrt_4_m_1_b_esq = __dsqrt_rd(__dadd_rd(4.0, -__drcp_ru(__dmul_rd(critical_ratio, critical_ratio))));

	double result_lb_skew = __dmul_ru(result_lb_skew_esq_hsq, __dadd_ru(2.0, -result_sqrt_4_m_1_b_esq));
	double result_ub_skew = __dmul_rd(result_ub_skew_esq_hsq, __dadd_rd(2.0,  result_sqrt_4_m_1_b_esq));

	return {result_lb_skew, result_ub_wmax < result_ub_skew ? result_ub_wmax : result_ub_skew};
}

