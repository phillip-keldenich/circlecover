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

/**
 * @file algcuda/macros.hpp Some macros to work with CUDA.
 */

#ifndef ALGCUDA_UTILS_MACROS_HPP_INCLUDED_
#define ALGCUDA_UTILS_MACROS_HPP_INCLUDED_

#ifndef __CUDACC__

/**
 * @brief Make sure __device__ is defined away if we are not using a CUDA-capable compiler.
 */
#ifndef __device__
#define __device__
#endif

/**
 * @brief Make sure __host__ is defined away if we are not using a CUDA-capable compiler.
 */
#ifndef __host__
#define __host__
#endif

/**
 * @brief Make sure __global__ is defined away if we are not using a CUDA-capable compiler.
 */
#ifndef __global__
#define __global__
#endif

#endif

#endif
