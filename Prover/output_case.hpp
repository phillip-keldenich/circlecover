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

#ifndef CIRCLECOVER_OUTPUT_CASE_HPP_INCLUDED_
#define CIRCLECOVER_OUTPUT_CASE_HPP_INCLUDED_

#include <algorithm>
#include <iostream>
#include <cstring>

namespace circlecover {
namespace detail {
static constexpr std::size_t output_width = 120;
}

/**
 * @brief A utility method for printing the begin of a case.
 * @param message 
 */
inline void output_begin_case(const char* message) {
	std::size_t mlen = std::strlen(message);
	if(mlen >= detail::output_width - 2) {
		std::cout << message << std::endl;
		return;
	}

	std::size_t len_pre  = (detail::output_width - mlen - 2) / 2;
	std::size_t len_post = detail::output_width - mlen - 2 - len_pre;

	std::fill_n(std::ostreambuf_iterator<char>(std::cout), len_pre, '-');
	std::cout << ' ' << message << ' ';
	std::fill_n(std::ostreambuf_iterator<char>(std::cout), len_post, '-');
	std::cout << std::endl;
}

/**
 * @brief A utility method for printing the end of a case.
 */
inline void output_end_case() {
	std::fill_n(std::ostreambuf_iterator<char>(std::cout), detail::output_width, '-');
	std::cout << std::endl;
}
}

#endif

