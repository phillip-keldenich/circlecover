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

#include "interrupt.hpp"
#include <cstdlib>

volatile std::sig_atomic_t circlecover::was_interrupted = 0;

static void (*previous_sigint_handler)(int) = nullptr;

static void our_sigint_handler(int) {
	if(circlecover::was_interrupted) {
		std::_Exit(SIGINT);
	}

	circlecover::was_interrupted = 1;
}

void circlecover::start_interruptible_computation() {
	was_interrupted = 0;
	previous_sigint_handler = std::signal(SIGINT, &our_sigint_handler);
}

void circlecover::stop_interruptible_computation() {
	std::signal(SIGINT, previous_sigint_handler);
	was_interrupted = 0;
}

