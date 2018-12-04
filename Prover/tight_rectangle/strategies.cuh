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

#ifndef CIRCLECOVER_TIGHT_RECTANGLE_STRATEGIES_CUH_INCLUDED_
#define CIRCLECOVER_TIGHT_RECTANGLE_STRATEGIES_CUH_INCLUDED_

#include "../operations.cuh"
#include "../rectangle_size_bound/strategies.cuh"
#include <iostream>
#include <algorithm>

namespace circlecover {
	namespace tight_rectangle {
		struct Variables {
			IV la;
			IV radii[4];

			bool __host__ operator< (const Variables& other) const noexcept {
				algcuda::Interval_compare comp;

				if(comp(la, other.la)) {
					return true;
				}
				
				if(comp(other.la, la)) {
					return false;
				}

				return std::lexicographical_compare(+radii, radii+4, +other.radii, other.radii+4, comp);
			}
		};

		std::ostream& __host__ operator<<(std::ostream& o, const Variables& vars);

		// compute an interval containing the critical ratio for the given width/height or lambda
		IV __device__ critical_ratio(IV w, IV h);
		IV __device__ critical_ratio(IV la);
		IV __device__ required_weight_for(IV w, IV h);
		IV __device__ required_weight_for(IV la);
	
		// whether we can apply our size-bounded covering result
		bool __device__ can_apply_size_bounded_covering(IV w, IV h, double ub_largest_disk, double lb_remaining_weight);
		double __device__ required_weight_for_bounded_size(double ub_w, double ub_h);

		// whether we can recurse
		bool __device__ can_recurse(IV w, IV h, double lb_remaining_weight);

		// whether we can recurse, using several approaches:
		//  - size bounded covering,
		//  - size bounded covering as guaranteed by corollary to theorem 1
		//  - regular recursion
		bool __device__ can_recurse(double ub_w, double ub_h, double ub_largest_disk, double lb_remaining_weight);

		// check whether there is a placement for r1 that allows us to recurse using all other disks
		bool __device__ r1_strategies(IV la, IV r1, double ub_r2);

		// check whether there is a placement for r1 and r2 that allows us to recurse using all other disks
		bool __device__ r1_r2_strategies(IV la, IV r1, IV r2, double ub_r3);

		// check whether there is a placement for r1, r2 and r3 that allows us to recurse using all other disks
		bool __device__ r1_r2_r3_strategies(IV la, IV r1, IV r2, IV r3, double ub_4);
		bool __device__ r1_r2_r3_strategies(const Variables& vars, IV R4);

		// place r1 in a corner and use r2,r3,r4 to cover the rest
		bool __device__ two_by_two_strategies(const Variables& vars, IV R);

		// place r1 and r2 horizontally next to each other, close the gap at the bottom and the top with r3,r4
		bool __device__ r1_r2_large_r3_r4_gaps_strategy(const Variables& vars, IV R);
	}
}

#endif

