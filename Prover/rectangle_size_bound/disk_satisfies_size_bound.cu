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

__device__ bool circlecover::rectangle_size_bound::disk_satisfies_size_bound(double r_squared_ub, double height_lb) {
	double height_scale_ub = __drcp_ru(height_lb);
	return __dmul_ru(r_squared_ub, __dmul_ru(height_scale_ub, height_scale_ub)) <= ub_disk_weight;
}

__device__ double circlecover::rectangle_size_bound::bound_required_height(double r_squared_ub) {
	return __ddiv_ru(__dsqrt_ru(r_squared_ub), ub_disk_radius);
}

__device__ double circlecover::rectangle_size_bound::bound_allowed_weight(double lb_height) {
	double lb_h_sq = __dmul_rd(lb_height, lb_height);
	return __dmul_rd(ub_disk_weight, lb_h_sq);
}

