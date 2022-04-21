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
 * @file algcuda/exit.cuh Some functions to abort computations from within a CUDA kernel.
 */

#ifndef ALGCUDA_UTILS_EXIT_CUH_INCLUDED_
#define ALGCUDA_UTILS_EXIT_CUH_INCLUDED_

namespace algcuda {
	/**
	 * @brief Cancel execution of the current kernel with an error.
	 */
	inline __device__ void trap() noexcept {
		__threadfence();
		asm("trap;\n");
	}

	/**
	 * @brief Cancel execution of the current kernel, printing an error message.
	 * 
	 * @param expr 
	 * @param file 
	 * @param line 
	 */
	inline __device__ void assert_trap(const char* expr, const char* file, int line) noexcept {
		printf("Assertion '%s' (%s:%d) failed!\n", expr, file, line);
		trap();
	}

	/**
	 * @brief Cancel execution of the current kernel without raising an error.
	 */
	inline __device__ void exit() noexcept {
		asm("exit;\n");
	}
}

#ifdef NDEBUG
#define ALGCUDA_ASSERT(expr) ((void)0)
#else
/**
 * @brief A CUDA assertion macro (traps if false, printing an error).
 * Does nothing if NDEBUG is defined.
 */
#define ALGCUDA_ASSERT(expr) ((void)((expr) ? 0 : (::algcuda::assert_trap( #expr , __FILE__, __LINE__), 0)))
#endif

#endif
