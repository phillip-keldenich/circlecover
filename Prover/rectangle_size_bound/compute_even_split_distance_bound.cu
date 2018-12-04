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

#include "strategies.cuh"

double __device__ circlecover::rectangle_size_bound::compute_even_split_distance_bound(const Variables& vars, const Intermediate_values& vals) {
	// we perform the splitting according to the upper bound of the groups - this gives us one possible split; it does not necessarily have to be the split produced by greedy splitting
	// on the exact disks, but it is a possible split so we can work with its result
	IV g1 = vars.radii[0];
	IV g2 = vars.radii[1] + vars.radii[2];

	for(int i = 3; i < 6; ++i) {
		if(g1.get_ub() <= g2.get_ub()) {
			g1 += vars.radii[i];
		} else {
			g2 += vars.radii[i];
		}
	}

	// upper bound on the distance between the groups
	double dist_ub  = __dadd_ru(g1.get_ub(), -g2.get_lb());
	double dist_ub2 = __dadd_ru(g2.get_ub(), -g1.get_lb());
	if(dist_ub < dist_ub2) {
		dist_ub = dist_ub2;
	}

	// compute an upper bound on the distance between the two groups minus the smallest explicit disk
	double dist_minus_smallest_ub = __dadd_ru(dist_ub, -vars.radii[5].get_lb());

	// if the remaining weight is at least this value, the smallest explicit disk is a bound
	double lb_R = vals.R.get_lb();
	if(dist_minus_smallest_ub <= lb_R) {
		return vars.radii[5].get_ub();
	}

	// otherwise, we add the entire rest to the smaller group
	if(g1.get_ub() <= g2.get_ub()) {
		g1 += vals.R;
	} else {
		g2 += vals.R;
	}

	// compute the new distance upper bound
	dist_ub  = __dadd_ru(g1.get_ub(), -g2.get_lb());
	dist_ub2 = __dadd_ru(g2.get_ub(), -g1.get_lb());
	if(dist_ub < dist_ub2) {
		dist_ub = dist_ub2;
	}

	// the largest disk is an upper bound on the result
	double res_ub = vars.radii[0].get_ub();
	if(dist_ub <= res_ub) {
		res_ub = dist_ub;
	}

	return res_ub;
}

