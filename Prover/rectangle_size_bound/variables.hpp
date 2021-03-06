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

#ifndef CIRCLECOVER_RECTANGLE_SIZE_BOUND_VARIABLES_HPP_INCLUDED_
#define CIRCLECOVER_RECTANGLE_SIZE_BOUND_VARIABLES_HPP_INCLUDED_

#include <algcuda/interval.hpp>
#include "values.hpp"
#include "../operations.hpp"
#include <iostream>

namespace circlecover {
	namespace rectangle_size_bound {
		struct Variables {
			IV la;               // the lambda we are using
			IV radii[6]; // note: these are squared radii!

			static Variables from_la_r1_r2(const Variables& v, IV la, IV r1, IV r2) {
				Variables result;

				result.la = v.la.intersect(la);
				result.radii[0] = v.radii[0].intersect(r1);
				result.radii[1] = v.radii[1].intersect(r2);

				for(int i = 2; i < 6; ++i) {
					result.radii[i] = v.radii[i];
					result.radii[i].tighten_ub(r2.get_ub());
				}

				result = result.tighten();
				return result;
			}

			__device__ __host__ Variables tighten() const {
				Variables result(*this);

				double lb = 0.0;
				for(int i = 5; i > 0; --i) {
					if(lb < radii[i].get_lb()) {
						lb = radii[i].get_lb();
					} else {
						result.radii[i].set_lb(lb);
					}
				}

				return result;
			}

			bool __host__ __device__ empty() const noexcept {
				if(la.empty()) {
					return true;
				}

				for(int i = 0; i < 6; ++i) {
					if(radii[i].get_lb() >= radii[i].get_ub()) {
						return true;
					}
				}

				return false;
			}

			bool __host__ operator< (const Variables& other) const noexcept {
				algcuda::Interval_compare comp;

				if(comp(la, other.la)) {
					return true;
				}
				
				if(comp(other.la, la)) {
					return false;
				}

				return std::lexicographical_compare(+radii, radii+6, +other.radii, other.radii+6, comp);
			}

			bool __host__ operator==(const Variables& other) const noexcept {
				if(la.get_lb() != other.la.get_lb())
					return false;

				if(la.get_ub() != other.la.get_ub())
					return false;

				for(int i = 0; i < 6; ++i) {
					if(radii[i].get_lb() != other.radii[i].get_lb() || radii[i].get_ub() != other.radii[i].get_ub())
						return false;
				}

				return true;
			}
		};

		inline __host__ std::ostream& operator<<(std::ostream& o, const Variables& v) {
			o << "λ = " << v.la;
			for(int i = 0; i < 6; ++i) {
				o << ", r²_" << (i+1) << " = " << v.radii[i];
			}
			return o;
		}

		struct Two_by_two_cover {
			Circle circles[4];
			double width;
		};

		struct Intermediate_values {
			Intermediate_values() noexcept = default;
			inline __device__ Intermediate_values(const Variables& vars);

			IV R;
			IV min_width_factor;               // a disk of radius r must cover a rectangle of width >= min_width_factor * r to be efficient enough
			IV max_width_factor;               // a disk of radius r must cover a rectangle of width <= max_width_factor * r to be efficient enough
			IV critical_ratio_sq;              // squared critical ratio
			double weight_bound_wall_building; // at most this weight is allowed in wall building for a length of 1; for other lengths, multiply by squared length!
			double width_fraction_incomplete;  // the maximal fraction of width which can be covered in an incomplete row during wall-building
			IV phase_factor_wall_building;     // the factor between the weight of the largest and the weight of the smallest possible disk in a row during wall building
			double lb_w_infty_denom;           // a lower bound on the denominator for the dropped weight for wall building when placing r1 in a corner
			Two_by_two_cover cover_2x2;        // a covering of a vertical strip by 4 disks in a 2x2 pattern
		};
	}
}

#endif

