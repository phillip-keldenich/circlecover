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

#include "operations.cuh"

namespace circlecover {
	__device__ IV squared_distance(const Point& p1, const Point& p2) {
		return (p1.x - p2.x).square() + (p1.y - p2.y).square();
	}

	__device__ Intersection_points intersection(const Circle& c1, const Circle& c2) {
		Intersection_points result;
		result.definitely_intersecting = false;

		IV sq_dist = squared_distance(c1.center, c2.center);
		if(sq_dist.get_lb() == 0) {
			return result;
		}

		IV rdiff = c1.squared_radius - c2.squared_radius;
		IV rdiff_sq = rdiff.square();

		IV fac1 = 2.0*(c1.squared_radius + c2.squared_radius)/sq_dist - 1.0 - rdiff_sq/sq_dist.square();
		if(fac1.get_lb() < 0) {
			return result;
		}
		fac1 = 0.5*sqrt(fac1);
		result.definitely_intersecting = true;

		IV fac2 = 0.5*rdiff/sq_dist;

		IV x1 = c1.center.x;
		IV x2 = c2.center.x;
		IV y1 = c1.center.y;
		IV y2 = c2.center.y;

		IV cross_x = 0.5*(x1+x2);
		IV cross_y = 0.5*(y1+y2);
		IV xdiff = x2-x1;
		IV ydiff = y2-y1;

		cross_x += fac2 * xdiff;
		cross_y += fac2 * ydiff;

		IV cr1_x = cross_x + fac1*ydiff;
		IV cr1_y = cross_y - fac1*xdiff;
		IV cr2_x = cross_x - fac1*ydiff;
		IV cr2_y = cross_y + fac1*xdiff;

		// if one of the points is definitely lexicographically less than the other, make sure its the first one
		if(ydiff.get_ub() < 0 || (ydiff.get_ub() == 0 && xdiff.get_lb() >= 0)) {
			result.p[0].x = cr1_x;
			result.p[0].y = cr1_y;
			result.p[1].x = cr2_x;
			result.p[1].y = cr2_y;
		} else {
			result.p[0].x = cr2_x;
			result.p[0].y = cr2_y;
			result.p[1].x = cr1_x;
			result.p[1].y = cr1_y;
		}

		return result;
	}

	__device__ Point center(const Point& p1, const Point& p2) {
		return { p1.x + 0.5*(p2.x-p1.x), p1.y + 0.5*(p2.y-p1.y) };
	}
}

