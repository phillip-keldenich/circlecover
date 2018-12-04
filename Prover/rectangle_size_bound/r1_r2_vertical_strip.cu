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

__device__ bool circlecover::rectangle_size_bound::shortcut_r1_r2_vertical_strip(IV la, IV r1, IV r2, IV r3) {
	double lb_w = two_disks_maximize_height(r1, r2, 1.0);
	if(lb_w <= 0.0) {
		return false;
	}

	double lb_R3 = __dadd_rd(__dmul_rd(la.get_lb(), critical_ratio), -__dadd_ru(r1.get_ub(), r2.get_ub()));
	IV wrem = la - lb_w;
	if(wrem.get_lb() >= 1.0 || disk_satisfies_size_bound(r3.get_ub(), wrem.get_lb())) {
		return lb_R3 >= __dmul_ru(critical_ratio, wrem.get_ub());
	} else {
		return can_recurse(lb_R3, r3.get_ub(), wrem.get_ub(), 1.0);
	}
}

__device__ bool circlecover::rectangle_size_bound::r1_r2_vertical_strip(const Variables& vars, const Intermediate_values& vals) {
	const IV r1 = vars.radii[0];
	const IV r2 = vars.radii[1];

	double lb_w = two_disks_maximize_height(r1, r2, 1.0);
	if(lb_w <= 0.0) {
		return false;
	}

	double lb_R3 = __dadd_rd(__dmul_rd(vars.la.get_lb(), critical_ratio), -__dadd_ru(r1.get_ub(), r2.get_ub()));
	IV wrem = vars.la - lb_w;

	int max_disk;
	double ub_wrem = wrem.get_ub();
	double ub_hrem = 1.0;
	double lb_R = lb_R3;

	for(max_disk = 2; max_disk < 6; ++max_disk) {
		// try recursion/worst-case recursion with the disk
		if(can_recurse(lb_R, vars.radii[max_disk].get_ub(), ub_wrem, ub_hrem)) {
			return true;
		}

		// try to place the disk covering a strip of the remaining area
		double minside = ub_wrem < ub_hrem ? ub_wrem : ub_hrem;
		double long_side_cutoff = __dadd_rd(__dmul_rd(4.0, vars.radii[max_disk].get_lb()), -__dmul_ru(minside,minside));
		if(long_side_cutoff <= 0) {
			// if that does not work, we have no idea what to do with this disk and have to fail
			return false;
		}

		// cut off a part of the long side now covered by the new disk
		long_side_cutoff = __dsqrt_rd(long_side_cutoff);
		if(ub_wrem < ub_hrem) {
			ub_hrem = __dadd_ru(ub_hrem, -long_side_cutoff);
			if(ub_hrem <= 0.0) {
				return true;
			}
		} else {
			ub_wrem = __dadd_ru(ub_wrem, -long_side_cutoff);
			if(ub_wrem <= 0.0) {
				return true;
			}
		}

		// update the remaining weight
		lb_R = __dadd_rd(lb_R, -vars.radii[max_disk].get_ub());
	}

	// we found explicit placements for all six explicit disks
	double ub_r7 = (vars.radii[5].get_ub() < vals.R.get_ub() ? vars.radii[5].get_ub() : vals.R.get_ub());
	return can_recurse(lb_R, ub_r7, ub_wrem, ub_hrem);
}

