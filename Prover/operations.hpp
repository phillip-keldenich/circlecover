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

#ifndef CIRCLECOVER_OPERATIONS_HPP_INCLUDED_
#define CIRCLECOVER_OPERATIONS_HPP_INCLUDED_

#include <algcuda/interval.hpp>

namespace circlecover {
/**
 * @brief An interval type with double endpoints.
 */
using IV = algcuda::Interval<double>;
/**
 * @brief An uncertain boolean type (false, true, indeterminate).
 */
using UB = algcuda::Uncertain<bool>;

/**
 * @brief A points with interval coordinates.
 */
struct Point {
	IV x, y;
};

/**
 * @brief Result type for intersection of two circles.
 */
struct Intersection_points {
	Point p[2];
	bool definitely_intersecting;
};

/**
 * @brief Circle with interval center and radius.
 */
struct Circle {
	Point      center;
	IV squared_radius;
};
}

#endif

