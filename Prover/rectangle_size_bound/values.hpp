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

#ifndef CIRCLECOVER_RECTANGLE_SIZE_BOUND_VALUES_HPP_INCLUDED_
#define CIRCLECOVER_RECTANGLE_SIZE_BOUND_VALUES_HPP_INCLUDED_

namespace circlecover {
	namespace rectangle_size_bound {
		static const double lb_proof_lambda = 1.0; // the lower bound on lambda
		static const double ub_proof_lambda = 2.5; // the lambda for up to which we want to prove

		static const double critical_ratio = 0.61; // the double value is less than the actual 0.61

		// max radius 0.375 / squared radius 0.140625
		static const double ub_disk_radius = 3.0 / 8.0;  // exactly representable
		static const double ub_disk_weight = 9.0 / 64.0; // exactly representable
	}
}

#endif

