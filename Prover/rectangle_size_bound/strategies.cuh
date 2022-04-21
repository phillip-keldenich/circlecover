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

/**
 * @brief Compute a bound on the weight difference between two groups resulting from Greedy Splitting.
 * 
 * @param vars The variables (i.e., lambda & disk radii).
 * @param vals Some intermediate values we do not want to recompute (includes R, the remaining weight).
 * @return double An upper bound on the weight difference between the two partitions resulting from Greedy Splitting. 
 */
__device__ double compute_even_split_distance_bound(const Variables& vars, const Intermediate_values& vals);

/**
 * @brief Check whether an 'even' split (i.e., trying to produce two roughly equal partitions by Greedy Splitting) allows recursion.
 * 
 * @param vars 
 * @param vals 
 * @return bool true if recursion based on even greedy splitting definitely works.
 */
__device__ bool   even_split_recursion(const Variables& vars, const Intermediate_values& vals); // Subsection Even and uneven recursive splitting

/**
 * @brief Same as even_split_recursion, but only considers lambda, r1 and r2 (as a shortcut to prevent unnecessary subdivision).
 * @param la Lambda
 * @param r1 Weight of r1.
 * @param r2 Weight of r2.
 * @return bool true if recursion based on even greedy splitting definitely works.
 */
__device__ bool   shortcut_even_split_recursion(IV la, IV r1, IV r2);

/**
 * @brief For a rectangle with minimum side length height_lb,
 * check if the disk with weight r_squared_ub satisfies the Lemma 4-size bound.
 * 
 * @param r_squared_ub 
 * @param height_lb 
 * @return true iff the disk definitely satisfies the size bound.
 */
__device__ bool   disk_satisfies_size_bound(double r_squared_ub, double height_lb);

/**
 * @brief Compute an upper bound on the necessary length of the smaller side if we want to use Lemma 4
 * with a disk of weight at most r_squared_ub.
 * @param r_squared_ub 
 * @return  
 */
__device__ double bound_required_height(double r_squared_ub);

/**
 * @brief Compute a lower bound on the allowed weight for a rectangle
 * with shorter side at least lb_height.
 * @param lb_height 
 * @return 
 */
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

/**
 * @brief Check whether n given disks can definitely cover a strip of given width/height.
 * 
 * @param ub_width An upper bound on the strip's width.
 * @param ub_height An upper bound on the strip's height.
 * @param disks The squared disk radii.
 * @param n The number of disks.
 * @return true if the cover definitely works; false otherwise.
 */
__device__ bool   nd_can_cover(double ub_width, double ub_height, const IV* disks, int n);

/**
 * @brief Maximize the the height of a strip of a given width that can be covered by a given number of disks.
 * 
 * Uses binary search/bisection for the more complex cases.
 * 
 * @param ub_width An upper bound on the strip's width.
 * @param disks The squared disk radii.
 * @param n The number of disks.
 * @return The amount of strip height that can definitely be covered; 0 if no cover may be possible at all.
 */
__device__ double nd_maximize_covered(double ub_width, const IV* disks, int n);

/**
 * @brief Check whether we can recurse using only our worst-case result (Theorem 1).
 * This does not take into account size bounds on the disks; neither does it consider
 * the case \lambda <= \bar{\lambda} (other than by returning 195/256, which is the
 * worst worst-case ratio for that range.)
 * 
 * @param lb_weight_remaining 
 * @param ub_w 
 * @param ub_h 
 * @return true if recursion definitely works solely based on Theorem 1; false otherwise.
 */
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
 * @brief Maximize the width of a substrip of a height-1 strip that can be covered by six given disks.
 * 
 * @param vars The variables (lambda and the squared disk radii).
 * @return double A lower bound on the width of the strip that can be covered with the first six disks.
 */
__device__ double  six_disks_maximize_covered_width(const Variables& vars);

/**
 * @brief Datatype that describes the result of placing two disks such that they cover the
 * bottom side of our rectangle.
 */
/*struct Bottom_row_2 {
	/// The upper intersection point of the two disks.
	Point upper_intersection;
	/// The two disks.
	Circle c1, c2;
	/// A lower bound on the height covered on each side.
	double lb_height_border;
};*/

// compute the range of squared radii for which an efficiency of critical_ratio can be achieved when covering a rectangular substrip of a strip of smaller side length h
// and longer side length at least lb_w
__device__ IV compute_efficient_rectangle_cover_weight_range(double lb_w, IV h);

// place r1 in the bottom-left corner and try to cover the remaining space above or to its right using 2-5 explicitly considered disks
__device__ bool   r1_in_corner_explicit_recursion(const Variables& vars, const Intermediate_values& vals); // Subsection Using the four largest disks and Subsection Building a Strip
__device__ bool   r1_in_corner_wall_building_recursion(const Variables& vars, const Intermediate_values& vals); // Subsection Placing r_1 in a corner

// cover a strip at the left side using 2x2 disks
__device__ Two_by_two_cover compute_two_by_two_cover(const IV* disks);

// cover an L-shaped region using explicit disks and recurse on the remaining parts

__device__ bool l_shaped_recursion(const Variables& vars, const Intermediate_values& vals); // Subsection Using the three largest disks

// cover a strip at the left side using 2x2 disks; cover the remaining strip using the two remaining disks and recursion
/**
 * @brief 
 * 
 * @param vars 
 * @param vals 
 * @return __device__ 
 */
__device__ bool  two_by_two_cover_with_strip_and_recursion(const Variables& vars, const Intermediate_values& vals);

/**
 * @brief Check whether our 5-disk strategies can definitely reach a full cover of the rectangle,
 *        possibly using recursion with the remaining disks.
 * 
 * @param vars The variable values for the given case.
 * @param vals Intemediate values for the given case.
 * @return true if we can definitely cover the rectangle using one of our five-disk strategies.
 */
__device__ bool five_disk_full_cover(const Variables& vars, const Intermediate_values& vals);

/**
 * @brief Check whether our 6-disk strategies can definitely reach a full cover of the rectangle,
 *        possibly using recursion with the remaining disks.
 * 
 * @param vars The variable values for the given case.
 * @param vals Intemediate values for the given case.
 * @return true if we can definitely cover the rectangle using one of our six-disk strategies.
 */
__device__ bool six_disk_full_cover(const Variables& vars, const Intermediate_values& vals);

/**
 * @brief Check whether any of our strategies that consider up to 7 disks explicitly
 *        can definitely handle the given case.
 * 
 * @param vars The variable values (disk radii, lambda, etc).
 * @param vals Intermediate values based on the variables (remaining weight, etc.)
 * @return true if we find a seven-disk strategy that definitely works; false otherwise.
 */
__device__ bool seven_disk_strategies(const Variables& vars, const Intermediate_values& vals);

}
}

#endif
