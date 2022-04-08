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

#ifndef CIRCLECOVER_RECTANGLE_SIZE_BOUND_STRATEGIES_CUH_INCLUDED_
#define CIRCLECOVER_RECTANGLE_SIZE_BOUND_STRATEGIES_CUH_INCLUDED_

#include "variables.cuh"
#include "../subintervals.hpp"
#include "../operations.cuh"

namespace circlecover {
namespace rectangle_size_bound {

// compute a bound for the weight distance between the two groups after greedy splitting
__device__ double compute_even_split_distance_bound(const Variables& vars, const Intermediate_values& vals);

// check whether an even split can be used for recursion
__device__ bool   even_split_recursion(const Variables& vars, const Intermediate_values& vals); // Subsection Even and uneven recursive splitting
__device__ bool   shortcut_even_split_recursion(IV la, IV r1, IV r2);

// for a rectangle of given height (smaller side length), check that a disk characterized by r_squared is small enough to satisfy the size bound
__device__ bool   disk_satisfies_size_bound(double r_squared_ub, double height_lb);

// check how large the height (smaller side length) has to be to contain a disk of given weight
__device__ double bound_required_height(double r_squared_ub);

// compute a lower bound on the maximum weight allowed in a rectangle of given height
__device__ double bound_allowed_weight(double lb_height);

// try recursion using an uneven split
__device__ bool uneven_split_recursion(const Variables& vars, const Intermediate_values& vals); // Subsection Even and uneven recursive splitting
__device__ bool shortcut_uneven_split_recursion(IV la, IV r1, IV r2);
__device__ bool shortcut_uneven_split_recursion(IV la, IV r1, IV r2, IV r3);

// try covering a vertical strip with some subset of the first 6 disks
__device__ bool multi_disk_strip_vertical(const Variables& vars, const Intermediate_values& vals); // Subsection Building a Strip
// an advanced, more expensive version of constructing a multi-disk strip that allows us to use recursion on the remainder, designed to avoid as many avoidable criticals as possible
__device__ bool advanced_multi_disk_strip_vertical(const Variables& vars, const Intermediate_values& vals); // Subsection Building a Strip

// try covering one corner with r1, the opposite corner with r2 and recurse (using both types of recursion) on the rest
__device__ bool r1_r2_opposite_corners_recursion(const Variables& vars, const Intermediate_values& vals); // Subsection Placing r_1 and r_2 in opposing corners
__device__ bool shortcut_r1_r2_opposite_corners_recursion(IV la, IV r1, IV r2, IV r3);

// try covering a vertical strip with r1 and r2, combined with several strategies on the remainder
__device__ bool r1_r2_vertical_strip(const Variables& vars, const Intermediate_values& vals); // Subsection Using the three largest disks
__device__ bool shortcut_r1_r2_vertical_strip(IV la, IV r1, IV r2, IV r3);

// try to use the wall-building argument: either there is a wall or a certain amount of weight in disks below a certain size
__device__ bool   vertical_wall_building_recursion(const Variables& vars, const Intermediate_values& vals); // Subsection Wall building

// maximize the height of a strip of width w covered by two or three disks
__device__ double two_disks_maximize_height(IV r1, IV r2, double ub_w);
struct Max_height_strip_2 {
	Circle c1, c2;
	double lb_height; // for two_disks_maximal_width_strip, width
};
__device__ Max_height_strip_2 two_disks_maximal_height_strip(IV la, IV r1, IV r2);
__device__ Max_height_strip_2 two_disks_maximal_width_strip(IV r1, IV r2);
__device__ bool three_disks_can_cover(IV r1, IV r2, IV r3, double width, double height);
__device__ double three_disks_maximize_height(IV r1, IV r2, IV r3, double ub_w);

struct Max_height_strip_wc_recursion_2 {
	Circle c1, c2;
	double lb_height;
};
__device__ Max_height_strip_wc_recursion_2 two_disks_maximal_height_strip_wc_recursion(IV r1, IV r2, IV R, double ub_w);

// try to decide whether the n disks pointed to by disks can definitely cover a strip of ub_width times ub_height
__device__ bool   nd_can_cover(double ub_width, double ub_height, const IV* disks, int n);

// generally, maximize the length of a strip of "width" w by n disks; 1 <= n <= 6; returns 0 if width is impossible
__device__ double nd_maximize_covered(double ub_width, const IV* disks, int n);

// check whether a ub_w x ub_h rectangle can be recursively covered (without size bound) given lb_weight_remaining weight
__device__ bool can_recurse_worst_case(double lb_weight_remaining, double ub_w, double ub_h);

/**
 * @brief Can we recursively apply our results (i.e., Theorem 1 & Lemma 3/4) to cover a rectangle?
 * Note on why we only take bounds instead of intervals:
 *  - When we check for recursion in our algorithm, we always check whether the preconditions
 *    of our lemmas/theorems apply either directly to the rectangle R we are recursing on,
 *    or to any rectangle R' that contains our rectangle R. This case is important for the
 *    size-bounded results, because they may not directly apply to R but to some R'; this
 *    mainly happens if we have plenty of leftover disk weight for a small rectangle,
 *    but our size bounds do not apply directly.
 *  - This allows us, in the prover, to only check whether we can cover some larger rectangle
 *    R' instead of R, thus we only need a upper bound on width/height of the rectangle
 *    we want to recurse on.
 *  - Obviously, covering becomes easier with more weight because we can simply drop the excess weight,
 *    so an upper bound on the remaining weight is useless.
 *  - The return result is bool instead of Uncertain<bool> because we cannot use an "indeterminate" result
 *    in our proof (we need to be certain recursion works, otherwise we have to assume it doesn't).
 * 
 * @param lb_weight_remaining A lower bound on the total disk weight remaining.
 * @param ub_largest_disk_weight An upper bound on the weight of the largest disk remaining.
 * @param ub_w An upper bound on the width.
 * @param ub_h An upper bound on the height.
 * @return bool true if we are certain that we can recurse; false otherwise. 
 */
__device__ bool can_recurse(double lb_weight_remaining, double ub_largest_disk_weight, double ub_w, double ub_h);

/**
 * @brief Compute an upper bound on the critical ratio for some rectangle.
 * This is mainly used if we want to cover more than one rectangle recursively;
 * we then use the result, together with the largest remaining disk weight,
 * to compute a bound on the amount of weight we lose covering some rectangle.
 * In this case, it is fine to just use upper bounds on the side lengths
 * because the required weight is clearly monotonic in the size of the rectangle.
 * Generally, we return infinity if it cannot verify that recursion works.
 * 
 * @param lb_weight_remaining The amount of weight remaining.
 * @param ub_largest_disk_weight The weight of the largest remaining disk.
 * @param ub_w An upper bound on the width.
 * @param ub_h An upper bound on the height.
 * @return An upper bound (may be infinity) on the critical ratio.
 */
__device__ double bound_critical_ratio(double lb_weight_remaining, double ub_largest_disk_weight, double ub_w, double ub_h);

/**
 * @brief Compute an upper bound on the weight of the largest disk that allows us to recurse.
 * See can_recurse and bound_critical_ratio (similar remarks apply).
 * 
 * @param lb_weight_remaining The total remaining weight.
 * @param ub_w An upper bound on the width of the rectangle.
 * @param ub_h An upper bound on the height of the rectangle.
 * @return double 0 if recursion might be impossible; otherwise, a lower bound on the size of the largest disk.
 */
__device__ double recursion_bound_size(double lb_weight_remaining, double ub_w, double ub_h);

/**
 * @brief Compute a bound on the worst-case covering ratio for a rectangle of width w and height h.
 * 
 * @param w 
 * @param h 
 * @return double The result does not take into account small skews (i.e., below \bar{\lambda}) and is
 *                always at least 195/256.
 */
__device__ double bound_worst_case_ratio(IV w, IV h);

/**
 * @brief Check whether six disks can cover a strip of 
 * 
 * @param vars 
 * @param width 
 * @return __device__ 
 */
//__device__ bool    six_disks_can_cover_width(const Variables& vars, double width);
__device__ double  six_disks_maximize_covered_width(const Variables& vars);

// a placement of two disks and recursion covering a strip at the bottom of the rectangle
struct Bottom_row_2 {
	Point upper_intersection;
	Circle c1, c2;
	double lb_height_border;
};
__device__ Bottom_row_2 compute_bottom_row(IV la, IV larger, IV smaller, IV recursion_weight);

// compute the range of squared radii for which an efficiency of critical_ratio can be achieved when covering a rectangular substrip of a strip of smaller side length h
// and longer side length at least lb_w
__device__ IV compute_efficient_rectangle_cover_weight_range(double lb_w, IV h);

// place r1 in the bottom-left corner and try to cover the remaining space above or to its right using 2-5 explicitly considered disks
__device__ bool   r1_in_corner_explicit_recursion(const Variables& vars, const Intermediate_values& vals); // Subsection Using the four largest disks and Subsection Building a Strip
__device__ bool   r1_in_corner_wall_building_recursion(const Variables& vars, const Intermediate_values& vals); // Subsection Placing r_1 in a corner

// cover a strip at the left side using 2x2 disks
__device__ Two_by_two_cover compute_two_by_two_cover(const IV* disks);

// cover a strip at the left side using 2x2 disks; cover the remaining strip using the two remaining disks and recursion
__device__ bool  two_by_two_cover_with_strip_and_recursion(const Variables& vars, const Intermediate_values& vals); // Subsection Using the four largest disks

// cover an L-shaped region using explicit disks and recurse on the remaining parts
__device__ bool l_shaped_recursion(const Variables& vars, const Intermediate_values& vals); // Subsection Using the three largest disks

// try full coverage by the five largest disks plus recursion on one region
__device__ bool five_disk_full_cover(const Variables& vars, const Intermediate_values& vals); // Subsection Using the five largest disks

// try full coverage by the six largest disks plus (potentially) recursion on one small region
__device__ bool six_disk_full_cover(const Variables& vars, const Intermediate_values& vals); // Subsection Using the six largest disks

// extend the number of explicitly considered disks to 7 and use different strategies for covering
__device__ bool seven_disk_strategies(const Variables& vars, const Intermediate_values& vals);

}
}

#endif

