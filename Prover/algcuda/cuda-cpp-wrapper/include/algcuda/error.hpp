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
 * @file algcuda/error.hpp CUDA error-related utilities.
 */

#include <system_error>
#include <string>
#include <utility>
#include <sstream>

namespace algcuda {
/**
 * @brief CUDA error condition.
 */
enum class Cuda_binary_error_condition : int {
	success = 0, error = 1
};

/**
 * @brief CUDA error code type (int).
 */
using Cuda_error_code = int;
}

namespace std {
/**
 * @brief Make sure our CUDA errors are recognized as error conditions.
 */
template<> struct is_error_condition_enum<algcuda::Cuda_binary_error_condition> : public true_type {};
}

/**
 * @brief Utilities to work with CUDA.
 */
namespace algcuda {
/**
 * @brief Utilities to work with the last error of CUDA.
 */
namespace last_error {

/**
 * @brief Clear the last CUDA error.
 */
void clear() noexcept;

/**
 * @brief Get the last error code from CUDA, and reset the error.
 * @return int CUDA error code.
 */
int get_and_clear() noexcept;

/**
 * @brief Get the last CUDA error.
 * @return int CUDA error code.
 */
int get() noexcept;

}

/**
 * @brief An error category for CUDA errors.
 */
class Cuda_category : public std::error_category {
public:
	/**
	 * @brief Construct a new Cuda_category object.
	 */
	Cuda_category() = default;

	/**
	 * @brief The name of our error category.
	 * 
	 * @return const char* "CUDA"
	 */
	virtual const char* name() const noexcept override { return "CUDA"; }

	/**
	 * @brief Construct an error condition from a CUDA error code.
	 * @param ev Error code.
	 * @return std::error_condition 
	 */
	virtual std::error_condition default_error_condition(int ev) const noexcept override {
		return ev == 0 ? std::error_condition(Cuda_binary_error_condition::success) : std::error_condition(Cuda_binary_error_condition::error);
	}
	
	/**
	 * @brief Get an error message from an error code.
	 * 
	 * @param ev Error code.
	 * @return std::string Message (from CUDA).
	 */
	virtual std::string message(int ev) const override;

private:
	static const Cuda_category& get_category() {
		static const Cuda_category result;
		return result;
	}

	friend inline const Cuda_category& cuda_category() noexcept;
};

/**
 * @brief Get a default CUDA error category object.
 * @return const Cuda_category& 
 */
inline const Cuda_category& cuda_category() noexcept {
	return Cuda_category::get_category();
}

/**
 * @brief Create an error condition with category CUDA.
 * 
 * @param ec 
 * @return std::error_condition 
 */
inline std::error_condition make_error_condition(Cuda_binary_error_condition ec) {
	return std::error_condition(static_cast<int>(ec), cuda_category());
}

/**
 * @brief Throw a std::system_error if we had a CUDA error.
 * 
 * @param ev The error code from CUDA.
 * @param message A message.
 */
inline void throw_if_cuda_error(int ev, std::string message) noexcept(false) {
	if(ev != 0) {
		last_error::clear();
		throw std::system_error(std::error_code(ev, cuda_category()), std::move(message));
	}
}

/**
 * @brief Throw a std::system_error if we had a CUDA error.
 * 
 * @param ev The error code from CUDA.
 * @param message A message.
 * @param file The file the error occurred in.
 * @param line The line the error occurred in.
 */
inline void throw_if_cuda_error(int ev, const std::string& message, const char* file, int line) noexcept(false) {
	if(ev != 0) {
		last_error::clear();
		std::ostringstream msg;
		msg << '\'' << file << ':' << line << '\'' << ' ' << message;
		throw std::system_error(std::error_code(ev, cuda_category()), msg.str());
	}
}

/**
 * @brief A macro to make using throw_if_cuda_error easier.
 */
#define ALGCUDA_THROW_IF_ERROR(expr, msg) ::algcuda::throw_if_cuda_error((expr), msg, __FILE__, __LINE__)

}
