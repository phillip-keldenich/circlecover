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

/**
 * @brief Everything that is specific to the automated proof of the size-bounded lemma (Lemma 4).
 */
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

/**
 * @brief Try recursion after performing an unbalanced partitioning of all disks into two sets.
 * This is the implementation of the subroutine described in the section
 * "Balanced and Unbalanced Recursive Splitting".
 * 
 * @param vars 
 * @param vals 
 * @return true if an unbalanced split definitely allows us to recurse; false otherwise.
 */
__device__ bool uneven_split_recursion(const Variables& vars, const Intermediate_values& vals);

/**
 * @brief A shortcut version of uneven_split_recursion that only considers lambda, r1 and r2.
 * This is used to avoid unnecessary subdivisions if it is already clear from the
 * first few disks that an uneven split definitely works.
 * 
 * @param la 
 * @param r1 
 * @param r2 
 * @return true if an unbalanced split definitely allows us to recurse; false otherwise.
 */
__device__ bool shortcut_uneven_split_recursion(IV la, IV r1, IV r2);

/**
 * @brief A shortcut version of uneven_split_recursion that only considers lambda, r1, r2 and r3.
 * This is used to avoid unnecessary subdivisions if it is already clear from the
 * first few disks that an uneven split definitely works.
 * 
 * @param la 
 * @param r1 
 * @param r2 
 * @param r3
 * @return true if an unbalanced split definitely allows us to recurse; false otherwise.
 */
__device__ bool shortcut_uneven_split_recursion(IV la, IV r1, IV r2, IV r3);

/**
 * @brief Check whether some (simple) placement of a subset of the first 6 disks
 * allows us to cover a vertical strip of our rectangle such that afterwards,
 * we can recurse on the remaining rectangle. See section "Building a Strip".
 * 
 * @param vars 
 * @param vals 
 * @return true iff a simple placement of disks definitely works.
 */
__device__ bool multi_disk_strip_vertical(const Variables& vars, const Intermediate_values& vals);

/**
 * @brief A more expensive, more thorough version of multi_disk_strip_vertical.
 * It is invoked later in the search than the cheaper multi_disk_strip_vertical to save time.
 * It tries more combinations and subsets; it also allows for the sub-strips to be non-rectangular
 * and relies on circle-circle intersections.
 * 
 * @param vars 
 * @param vals 
 * @return true iff an advanced placement definitely works.
 */
__device__ bool advanced_multi_disk_strip_vertical(const Variables& vars, const Intermediate_values& vals);

// try covering one corner with r1, the opposite corner with r2 and recurse (using both types of recursion) on the rest

/**
 * @brief Check whether we can guarantee success by placing r_1 and r_2 in opposite corners.
 * See section "Placing r_1 and r_2 in Opposite Corners".
 * 
 * @param vars 
 * @param vals 
 * @return true if we can guarantee success with r1 and r2 placed in opposite corners.
 */
__device__ bool r1_r2_opposite_corners_recursion(const Variables& vars, const Intermediate_values& vals); // Subsection Placing r_1 and r_2 in opposing corners

/**
 * @brief A shortcut version of r1_r2_opposite_corners_recursion that only considers
 *        lambda and r1, r2, r3 instead of all disks to avoid unnecessary subdivisions.
 * 
 * @param la
 * @param r1
 * @param r2
 * @param r3
 * @return true if we can guarantee success with r1 and r2 placed in opposite corners.
 */
__device__ bool shortcut_r1_r2_opposite_corners_recursion(IV la, IV r1, IV r2, IV r3);

/**
 * @brief Try several covering routines that are based on covering a vertical substrip of the rectangle
 *        with r1 and r2 and various approaches for the remaining disks/rectangle.
 * See section "Using the Three Largest Disks".
 * 
 * @param vars 
 * @param vals 
 * @return true if we can guarantee success with one of the covering routines. 
 */
__device__ bool r1_r2_vertical_strip(const Variables& vars, const Intermediate_values& vals);

/**
 * @brief A shortcut version of r1_r2_vertical_strip that only considers up to r3.
 * Used to avoid unnecessary subdivisions.
 * See section "Using the Three Largest Disks".
 * 
 * @param la 
 * @param r1
 * @param r2
 * @param r3 
 * @return true if we can guarantee success with one of the covering routines. 
 */
__device__ bool shortcut_r1_r2_vertical_strip(IV la, IV r1, IV r2, IV r3);

/**
 * @brief Check whether we can guarantee coverage by using our wall-building argument:
 * Either we can successfully build a (column of) a wall,
 * or a certain amount of weight is in sufficiently small disks.
 * See section "Wall Building".
 * 
 * @param vars 
 * @param vals 
 * @return true iff the wall-building argument is guaranteed to work. 
 */
__device__ bool vertical_wall_building_recursion(const Variables& vars, const Intermediate_values& vals);

/**
 * @brief Compute the maximum height of a strip of given width that can be covered by two disks.
 * 
 * @param r1 The (squared) radius of the first (larger) disk.
 * @param r2 The (squared) radius of the second (larger) disk.
 * @param ub_w An upper bound on the width of the strip.
 * @return A lower bound on the height that can be covered by r1 and r2;
 * 0 if it is possible that no height can be covered.
 */
__device__ double two_disks_maximize_height(IV r1, IV r2, double ub_w);

/**
 * @brief A datastructure describing a strip of maximal height covered by two disks.
 */
struct Max_height_strip_2 {
	Circle c1, c2; //< The two disk positions.
	double lb_height; //< A lower bound on the covered height.
};

/**
 * @brief Compute a placement of two disks in a strip of width lambda.
 * Maximize the height covered.
 * 
 * @param la 
 * @param r1 
 * @param r2 
 * @return Max_height_strip_2 
 */
__device__ Max_height_strip_2 two_disks_maximal_height_strip(IV la, IV r1, IV r2);
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
