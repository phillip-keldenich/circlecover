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
 * @file algcuda/properties.hpp Utilities to read CUDA device properties.
 */

#ifndef ALGCUDA_PROPERTIES_HPP_INCLUDED_
#define ALGCUDA_PROPERTIES_HPP_INCLUDED_

namespace algcuda {
	namespace device {
		/**
		 * @brief A CUDA device ID (int).
		 */
		using Id = int;

		/**
		 * @brief Get the number of CUDA devices.
		 * @return int 
		 */
		int count();

		/**
		 * @brief Get the current default device.
		 * @return Id The device ID.
		 */
		Id current_default();

		/**
		 * @brief Get the maximum number of threads that can be started per block for the current default device.
		 * @return int 
		 */
		int max_threads_per_block();

		/**
		 * @brief Get the maximum number of threads that can be started per block.
		 * 
		 * @param id The device to get the result for.
		 * @return int 
		 */
		int max_threads_per_block(device::Id id);

		/**
		 * @brief Cached version of the functions above.
		 * Only use once the default device has been (implicitly) set.
		 */
		namespace cached {
			/**
			 * @brief A cached version of the max_threads_per_block function.
			 * @return int 
			 */
			int max_threads_per_block();
		}
	}
}

#endif
