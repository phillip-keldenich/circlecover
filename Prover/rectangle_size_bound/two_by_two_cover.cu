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
#include "../operations.cuh"

using namespace circlecover;
using namespace circlecover::rectangle_size_bound;

static __device__ Two_by_two_cover compute_two_by_two_cover_concrete(double r1, double r2, double r3, double r4) {
	Two_by_two_cover result;
	result.width = 0.0;

	// first column
	IV w1;
	{
		IV rdiff = IV{r1,r1} - r2;
		IV rdiff_sq = rdiff.square();
		IV h1 = 0.5 + 2.0 * rdiff;
		IV h2 = 0.5 - 2.0 * rdiff; // this must be positive (we never have 0.5 radius)
		w1  = 2.0 * (IV(r1,r1) + r2) - 0.25 - 4.0*rdiff_sq;

		if(possibly(r1 - 0.25*h1.square() <= 0.0) || possibly(r2 - 0.25*h2.square() <= 0.0) || possibly(w1 <= 0.0)) {
			w1 = IV{0.0,0.0};
			result.circles[0] = Circle{{{0.0,0.0}, {0.0,0.0}}, {r1,r1}};
			result.circles[1] = Circle{{{0.0,0.0}, {0.0,0.0}}, {r2,r2}};
		} else {
			w1 = sqrt(w1);
			IV x = 0.5 * w1;
			result.circles[0] = Circle{{x, 0.5*h1}, {r1,r1}};
			result.circles[1] = Circle{{x, 1.0 - 0.5*h2}, {r2,r2}};
			result.width = w1.get_lb();
		}
	}

	// second column
	{
		IV rdiff = IV{r3,r3} - r4;
		IV rdiff_sq = rdiff.square();
		IV h1 = 0.5 + 2.0 * rdiff;
		IV h2 = 0.5 - 2.0 * rdiff; // this must be positive (we never have 0.5 radius)
		IV w2  = 2.0 * (IV(r3,r3) + r4) - 0.25 - 4.0*rdiff_sq;

		if(possibly(r3 - 0.25*h1.square() <= 0.0) || possibly(r4 - 0.25*h2.square() <= 0.0) || possibly(w2 <= 0.0)) {
			result.circles[3] = Circle{{{0.0,0.0}, {0.0,0.0}}, {r3,r3}};
			result.circles[2] = Circle{{{0.0,0.0}, {0.0,0.0}}, {r4,r4}};
		} else {
			w2 = sqrt(w2);
			IV x = w1 + 0.5 * w2;
			result.circles[3] = Circle{{x, 1.0 - 0.5*h1}, {r3,r3}};
			result.circles[2] = Circle{{x, 0.5 * h2}, {r4, r4}};
			result.width = __dadd_rd(result.width, w2.get_lb());
		}
	}

	return result;
}

__device__ Two_by_two_cover circlecover::rectangle_size_bound::compute_two_by_two_cover(const IV* disks) {
	Two_by_two_cover c1 = compute_two_by_two_cover_concrete(disks[0].get_lb(), disks[1].get_lb(), disks[2].get_lb(), disks[3].get_lb());
	Two_by_two_cover c2 = compute_two_by_two_cover_concrete(disks[0].get_lb(), disks[2].get_lb(), disks[1].get_lb(), disks[3].get_lb());
	Two_by_two_cover c3 = compute_two_by_two_cover_concrete(disks[0].get_lb(), disks[3].get_lb(), disks[1].get_lb(), disks[2].get_lb());

	if(c1.width > c2.width) {
		return c1.width > c3.width ? c1 : c3;
	} else {
		return c2.width > c3.width ? c2 : c3;
	}
}

