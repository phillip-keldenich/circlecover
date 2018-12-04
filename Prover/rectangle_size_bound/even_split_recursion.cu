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

bool __device__ circlecover::rectangle_size_bound::even_split_recursion(const Variables& vars, const Intermediate_values& vals) {
	// compute distance bound
	double distance_ub = compute_even_split_distance_bound(vars, vals);

	// compute width of smaller part
	double fraction_smaller_lb = __dadd_rd(0.5, -__ddiv_ru(distance_ub, __dmul_rd(2.0, __dmul_rd(vars.la.get_lb(), critical_ratio))));
	double width_smaller_lb = __dmul_rd(fraction_smaller_lb, vars.la.get_lb());

	// if the width is at least 1, everything is fine
	if(width_smaller_lb >= 1.0) {
		return true;
	}

	// check whether r1 still fits into the smaller part
	return disk_satisfies_size_bound(vars.radii[0].get_ub(), width_smaller_lb);
}

bool __device__ circlecover::rectangle_size_bound::shortcut_even_split_recursion(IV la, IV r1, IV r2) {
	// perform split such that r1 is in the bigger group; this means that r1 has to fit into a la/2-strip
	// moreover, the larger group has weight at most critical_ratio*la/2 + r2;
	// the smaller group has weight at least critical_ratio*la/2 - r2 and area at least
	// la/2 - r2/critical_ratio

	// check that r1 fits
	double lb_bigger_width = __dmul_rd(0.5, la.get_lb());
	if(lb_bigger_width < 1.0 && !disk_satisfies_size_bound(r1.get_ub(), lb_bigger_width)) {
		return false;
	}

	// check that r2 fits
	double lb_smaller_width = __dadd_rd(lb_bigger_width, -__ddiv_ru(r2.get_ub(), critical_ratio));
	return lb_smaller_width >= 1.0 || disk_satisfies_size_bound(r2.get_ub(), lb_smaller_width);
}

