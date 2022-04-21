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
		/**
		 * @brief A datastructure containing the variables of our prover.
		 * These variables are lambda and the squared radii of the 6 largest disks.
		 * Note that we replace radii by squared radii practically everywhere in the prover
		 * since the squared radii occur much more often in our formulas.
		 */
		struct Variables {
			/// An interval for lambda >= 1, the width of our rectangle (the height is normalized to 1).
			IV la;
			/// Squared radii r1^2 to r6^2. 
			IV radii[6];

			/**
			 * @brief Create a Variables object given a Variables object and bounds on lambda, r1^2 and r2^2.
			 * 
			 * @param v A variables object with some original bounds.
			 * @param la Finer bounds on lambda.
			 * @param r1 Finer bounds on the squared radius of the largest disk.
			 * @param r2 Finer bounds on the squared radius of the second-largest disk.
			 * @return Variables A new Variables object incorporating the new bounds.
			 */
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

			/**
			 * @brief Tighten the bounds in this Variables object.
			 * Applies the fact that r_i >= r_{i+1} >= 0.
			 * @return Variables  
			 */
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

			/**
			 * @brief Check whether this Variables object is empty.
			 * A Variables object is empty if any of the intervals is empty,
			 * i.e., has lower bound > upper bound.
			 * 
			 * @return true If empty.
			 */
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

			/**
			 * @brief Lexicographically compare two variables objects.
			 * 
			 * @param other 
			 * @return true 
			 * @return false 
			 */
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

			/**
			 * @brief Check whether two Variables objects describe exactly the same hyperrectangle.
			 * 
			 * @param other 
			 * @return true 
			 * @return false 
			 */
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

		/**
		 * @brief Output a Variables object to a stream.
		 * 
		 * @param o 
		 * @param v 
		 * @return std::ostream&
		 */
		inline __host__ std::ostream& operator<<(std::ostream& o, const Variables& v) {
			o << "λ = " << v.la;
			for(int i = 0; i < 6; ++i) {
				o << ", r²_" << (i+1) << " = " << v.radii[i];
			}
			return o;
		}

		/**
		 * @brief A placement of four disks in a two-by-two grid-like position.
		 */
		struct Two_by_two_cover {
			/// The position of the disks.
			Circle circles[4];
			/// A lower bound on the total rectangle width covered by the placement.
			double width;
		};

		/**
		 * @brief A collection of intermediate values that we occasionally need
		 * and don't want to constantly recompute.
		 */
		struct Intermediate_values {
			Intermediate_values() noexcept = default;
			inline __device__ Intermediate_values(const Variables& vars);

			IV R;                              //< The remaining weight after all six disks in vars are removed.
			IV min_width_factor;               //< A disk of radius r must cover a rectangle of width >= min_width_factor * r to be efficient enough
			IV max_width_factor;               //< A disk of radius r must cover a rectangle of width <= max_width_factor * r to be efficient enough
			IV critical_ratio_sq;              //< Squared critical ratio
			double weight_bound_wall_building; //< At most this weight is allowed in wall building for a length of 1; for other lengths, multiply by squared length!
			double width_fraction_incomplete;  //< The maximal fraction of width which can be covered in an incomplete row during wall-building
			IV phase_factor_wall_building;     //< The factor between the weight of the largest and the weight of the smallest possible disk in a row during wall building
			double lb_w_infty_denom;           //< A lower bound on the denominator for the dropped weight for wall building when placing r1 in a corner
			Two_by_two_cover cover_2x2;        //< A covering of a vertical strip by 4 disks in a 2x2 pattern
		};
	}
}

#endif

