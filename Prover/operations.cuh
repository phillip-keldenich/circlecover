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

#ifndef CIRCLECOVER_OPERATIONS_CUH_INCLUDED_
#define CIRCLECOVER_OPERATIONS_CUH_INCLUDED_

#include "operations.hpp"
#include <algcuda/interval.cuh>

namespace circlecover {
/**
 * @brief Compute the squared distance between two points p1, p2 (in interval arithmetic).
 * 
 * @param p1 Point p1.
 * @param p2 Point p2.
 * @return IV Interval for the squared distance. 
 */
__device__ IV  squared_distance(const Point& p1, const Point& p2);

/**
 * @brief Compute the intersection point(s) of two circles in interval arithmetic.
 * 
 * The most general result this function can return is "not sure if there is an intersection".
 * In that case, the output of the method cannot be realistically used (because there might be an intersection,
 * but there might also be none).
 * 
 * @param c1 
 * @param c2 
 * @return Intersection_points A structure describing the points and whether the intersection is certain. 
 */
__device__ Intersection_points intersection(const Circle& c1, const Circle& c2);

/**
 * @brief Compute the center point between two points (in interval arithmetic).
 * 
 * @param p1 
 * @param p2 
 * @return Point The center. 
 */
__device__ Point center(const Point& p1, const Point& p2);
}

#endif
