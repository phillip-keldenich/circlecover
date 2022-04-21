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

static inline __device__ bool nd_can_cover_no_small(double ub_width, double ub_height, const IV* disks, int n) {
	double htot = 0.0, wtot = 0.0;
	double wsq = __dmul_ru(ub_width, ub_width);
	double hsq = __dmul_ru(ub_height, ub_height);

	for(int i = 0; i < n; ++i) {
		double lb_wcur = __dadd_rd(__dmul_rd(4.0, disks[i].get_lb()), -hsq);
		double lb_hcur = __dadd_rd(__dmul_rd(4.0, disks[i].get_lb()), -wsq);
		if(lb_wcur <= 0) {
			lb_wcur = 0;
		} else {
			lb_wcur = __dsqrt_rd(lb_wcur);
		}
		if(lb_hcur <= 0) {
			lb_hcur = 0;
		} else {
			lb_hcur = __dsqrt_rd(lb_hcur);
		}
		wtot = __dadd_rd(wtot, lb_wcur);
		htot = __dadd_rd(htot, lb_hcur);
	}

	return htot >= ub_height || wtot >= ub_width;
}

__device__ bool circlecover::rectangle_size_bound::nd_can_cover(double ub_width, double ub_height, const IV* disks, int n) {
	if(n == 1) {
		double wsq = __dmul_ru(ub_width, ub_width);
		double hsq = __dmul_ru(ub_height, ub_height);
		return __dmul_rd(4.0, disks[0].get_lb()) >= __dadd_ru(wsq, hsq);
	}

	if(n == 2) {
		double h = two_disks_maximize_height(disks[0], disks[1], ub_width);
		double w = two_disks_maximize_height(disks[0], disks[1], ub_height);
		return h >= ub_height || w >= ub_width;
	}

	if(n == 3) {
		return three_disks_can_cover(disks[0], disks[1], disks[2], ub_width, ub_height);
	}

	return nd_can_cover_no_small(ub_width, ub_height, disks, n);
}

__device__ double circlecover::rectangle_size_bound::nd_maximize_covered(double ub_width, const IV* disks, int n) {
	if(n == 1) {
		double wsq = __dmul_ru(ub_width, ub_width);
		double h = __dadd_rd(__dmul_rd(4.0, disks[0].get_lb()), -wsq);
		return h <= 0 ? 0.0 : __dsqrt_rd(h);
	}

	if(n == 2) {
		return two_disks_maximize_height(disks[0], disks[1], ub_width);
	}

	if(n == 3) {
		return three_disks_maximize_height(disks[0], disks[1], disks[2], ub_width);
	}

	double smallest_disk = disks[n-1].get_lb();
	double ub = __dmul_rd(2.0, __dsqrt_rd(smallest_disk));
	double lb = 0.0;

	for(;;) {
		double h = 0.5*(lb+ub);
		if(h <= lb || h >= ub) {
			return lb;
		}

		if(nd_can_cover_no_small(ub_width, h, disks, n)) {
			lb = h;
		} else {
			ub = h;
		}
	}
}

__device__ double circlecover::rectangle_size_bound::two_disks_maximize_height(IV r1_, IV r2_, double ub_w) {
	const double r1 = r1_.get_lb();
	const double r2 = r2_.get_lb();

	double w_sq = __dmul_ru(ub_w,ub_w);
	double rdiff_lb = __dadd_rd(r1, -r2);
	double rdiff_ub = __dadd_ru(r1, -r2);
	double rdiff_sq = __dmul_ru(rdiff_ub, rdiff_ub);

	// check that the disks are large enough to cover their respective part of the square
	double ub_w1 = __dadd_ru(__dmul_ru(0.5, ub_w), __dmul_ru(__ddiv_ru(2.0, ub_w), rdiff_ub));
	double ub_w2 = __dadd_ru(__dmul_ru(0.5, ub_w), -__dmul_rd(__ddiv_rd(2.0, ub_w), rdiff_lb));
	ub_w1 = __dmul_ru(ub_w1, ub_w1);
	ub_w2 = __dmul_ru(ub_w2, ub_w2);
	if(r1 <= __dmul_ru(0.25, ub_w1) || r2 <= __dmul_ru(0.25, ub_w2)) {
		return 0.0;
	}

	double lb_h = __dadd_rd(__dmul_rd(2.0, __dadd_rd(r1, r2)), -__dadd_ru(__dmul_ru(0.25, w_sq), __dmul_ru(4.0, __ddiv_ru(rdiff_sq, w_sq))));
	if(lb_h <= 0) {
		return 0.0;
	}

	return __dsqrt_rd(lb_h);
}

__device__ Max_height_strip_2 circlecover::rectangle_size_bound::two_disks_maximal_height_strip(IV la_, IV r1_, IV r2_) {
	const double la = la_.get_ub();
	const double r1 = r1_.get_lb();
	const double r2 = r2_.get_lb();

	IV rdiff{__dadd_rd(r1, -r2), __dadd_ru(r1, -r2)};
	IV wdiff = 2.0 * rdiff / la;
	IV w1    = IV{0.5,0.5} * la + wdiff;
	IV w2    = IV{0.5,0.5} * la - wdiff;
	IV h1    = r1 - 0.25 * w1.square();
	IV h2    = r2 - 0.25 * w2.square();

	if(possibly(h1 <= 0) || possibly(h2 <= 0)) {
		return {{{{0.0,0.0},{0.0,0.0}}, {r1,r1}}, {{{0.0,0.0},{0.0,0.0}}, {r2,r2}}, 0.0};
	}

	IV h = sqrt(h1);
	return {{{0.5*w1, h}, {r1,r1}}, {{la-0.5*w2, h}, {r2,r2}}, __dmul_rd(2.0, h.get_lb())};
}

/*__device__ Max_height_strip_2 circlecover::rectangle_size_bound::two_disks_maximal_width_strip(IV r1_, IV r2_) {
	const double r1 = r1_.get_lb();
	const double r2 = r2_.get_lb();

	IV rdiff{__dadd_rd(r1, -r2), __dadd_ru(r1, -r2)};
	IV hdiff = 2.0 * rdiff;
	IV h1    = IV{0.5,0.5} + hdiff;
	IV h2    = IV{0.5,0.5} - hdiff;
	IV w1    = r1 - 0.25 * h1.square();
	IV w2    = r2 - 0.25 * h2.square();

	if(possibly(w1 <= 0) || possibly(w2 <= 0)) {
		return {{{{0.0,0.0},{0.0,0.0}}, {r1,r1}}, {{{0.0,0.0},{0.0,0.0}}, {r2,r2}}, 0.0};
	}

	IV w = sqrt(w1);
	return {{{w, 0.5*h1}, {r1,r1}}, {{w, 1.0-0.5*h2}, {r2,r2}}, __dmul_rd(2.0, w.get_lb())};
}*/

__device__ bool circlecover::rectangle_size_bound::three_disks_can_cover(IV r1, IV r2, IV r3, double width, double height) {
	double h_sq = __dmul_ru(height, height);
	double lb_w3 = __dadd_rd(__dmul_rd(4.0, r3.get_lb()), -h_sq);
	if(lb_w3 <= 0) {
		lb_w3 = 0;
	} else {
		lb_w3 = __dsqrt_rd(lb_w3);
	}

	double lb_w2 = __dadd_rd(__dmul_rd(4.0, r2.get_lb()), -h_sq);
	if(lb_w2 <= 0) {
		lb_w2 = 0;
	} else {
		lb_w2 = __dsqrt_rd(lb_w2);
	}

	double lb_w1 = __dadd_rd(__dmul_rd(4.0, r1.get_lb()), -h_sq);
	if(lb_w1 <= 0) {
		lb_w1 = 0;
	} else {
		lb_w1 = __dsqrt_rd(lb_w1);
	}

	double lb_wtot = __dadd_rd(lb_w1, __dadd_rd(lb_w2, lb_w3));
	return lb_wtot >= width;
}

__device__ double circlecover::rectangle_size_bound::three_disks_maximize_height(IV r1, IV r2, IV r3, double ub_w) {
	// the explicit solution is extremely complicated; use bisection
	double lb = 0.0, ub = __dmul_rd(2.0, __dsqrt_rd(r3.get_lb()));

	for(;;) {
		double mid = 0.5*(lb+ub);
		if(mid <= lb || mid >= ub) {
			return lb;
		}

		double h_sq = __dmul_ru(mid, mid);
		double lb_w3 = __dadd_rd(__dmul_rd(4.0, r3.get_lb()), -h_sq);
		if(lb_w3 <= 0) {
			lb_w3 = 0;
		} else {
			lb_w3 = __dsqrt_rd(lb_w3);
		}

		double lb_w2 = __dadd_rd(__dmul_rd(4.0, r2.get_lb()), -h_sq);
		if(lb_w2 <= 0) {
			lb_w2 = 0;
		} else {
			lb_w2 = __dsqrt_rd(lb_w2);
		}

		double lb_w1 = __dadd_rd(__dmul_rd(4.0, r1.get_lb()), -h_sq);
		if(lb_w1 <= 0) {
			lb_w1 = 0;
		} else {
			lb_w1 = __dsqrt_rd(lb_w1);
		}

		double lb_wtot = __dadd_rd(lb_w1, __dadd_rd(lb_w2, lb_w3));
		if(lb_wtot < ub_w) {
			ub = mid;
		} else {
			lb = mid;
		}
	}
}

__device__ Max_height_strip_wc_recursion_2 circlecover::rectangle_size_bound::two_disks_maximal_height_strip_wc_recursion(IV r1, IV r2, IV R, double ub_w) {
	double lb = 0.0, ub = __dmul_rd(2.0, r2.get_lb());
	const double lb_R = R.get_lb();
	const double lb_r1 = r1.get_lb();
	const double lb_r2 = r2.get_lb();
	IV c1x, c2x;

	for(;;) {
		double mid = 0.5*(lb+ub);
		if(mid <= lb || mid >= ub) {
			break;
		}

		double h_sq = __dmul_ru(mid,mid);
		double lb_w1 = __dadd_rd(__dmul_rd(4.0, lb_r1), -h_sq);
		double lb_w2 = __dadd_rd(__dmul_rd(4.0, lb_r2), -h_sq);
		if(lb_w1 < 0 || lb_w2 < 0) {
			ub = mid;
			continue;
		}

		lb_w1 = __dsqrt_rd(lb_w1);
		lb_w2 = __dsqrt_rd(lb_w2);
		double ub_wrem = __dadd_ru(ub_w, -__dadd_rd(lb_w1,lb_w2));
		if(ub_wrem <= 0.0 || can_recurse_worst_case(lb_R, ub_wrem, mid)) {
			lb = mid;
			c1x = IV{__dmul_rd(0.5, lb_w1), __dmul_ru(0.5, lb_w1)};
			c2x = IV{__dadd_rd(ub_w, -__dmul_ru(0.5, lb_w2)), __dadd_ru(ub_w, -__dmul_rd(0.5, lb_w2))};
		} else {
			ub = mid;
		}
	}

	if(lb <= 0.0) {
		return {{{{0.0,0.0},{0.0,0.0}}, {0.0,0.0}}, {{{0.0,0.0},{0.0,0.0}}, {0.0,0.0}}, 0.0};
	}

	IV y{__dmul_rd(0.5, lb), __dmul_ru(0.5, lb)};
	return {{{c1x,y},{lb_r1,lb_r1}},{{c2x,y}, {lb_r2,lb_r2}},lb};
}

