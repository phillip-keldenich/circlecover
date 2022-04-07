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

namespace circlecover {
namespace tight_rectangle {
/**
 * Compute the critical covering ratio,
 * i.e., the weight required per area
 * for a rectangle of width w and height h.
 */
IV __device__ critical_ratio(IV w, IV h) {
	// internally, we use critical_ratio(lambda)
	if(w.get_ub() <= h.get_lb()) {
		// taller than wide
		return critical_ratio(h/w);
	} else if(h.get_ub() <= w.get_lb()) {
		// wider than tall
		return critical_ratio(w/h);
	} else {
		// could be either case;
		// compute critical ratio for both
		// cases and take union of resulting ranges
		IV hbw = h/w;
		hbw.tighten_lb(1.0);
		IV wbh = w/h;
		wbh.tighten_lb(1.0);
		IV c1 = critical_ratio(hbw);
		IV c2 = critical_ratio(wbh);
		return {c1.get_lb() < c2.get_lb() ? c1.get_lb() : c2.get_lb(), c1.get_ub() < c2.get_ub() ? c2.get_ub() : c1.get_ub()};
	}
}


/**
 * Compute critical ratio for some given lambda.
 */
IV __device__ critical_ratio(IV la) {
	// bounds on our break-even point lambda_2
	const double lb_break_even = __dsqrt_rd(__dadd_rd(__dmul_rd(0.5, __dsqrt_rd(7.0)), -0.25));
	const double ub_break_even = __dsqrt_ru(__dadd_ru(__dmul_ru(0.5, __dsqrt_ru(7.0)), -0.25));

	if(definitely(la <= lb_break_even)) {
		// below lambda_2
		IV la_sq = la.square();
		return ((3.0/256.0) / la) * (16.0*la_sq + 40.0 + 9.0/la_sq);
	} else if(definitely(la >= ub_break_even)) {
		// above lambda_2
		return 0.25 * la + 0.5 * la.reciprocal();
	} else {
		// could be either case;
		// compute for both cases and take union of resulting ranges
		IV la_sq = la.square();
		IV la1 = ((3.0/256.0) / la) * (16.0*la_sq + 40.0 + 9.0/la_sq);
		IV la2 = 0.25 * la + 0.5 * la.reciprocal();
		return {la1.get_lb() < la2.get_lb() ? la1.get_lb() : la2.get_lb(), la1.get_ub() < la2.get_ub() ? la2.get_ub() : la1.get_ub()};
	}
}

IV __device__ required_weight_for(IV w, IV h) {
	return critical_ratio(w,h) * w * h;
}

IV __device__ required_weight_for(IV la) {
	return critical_ratio(la) * la;
}
}
}

