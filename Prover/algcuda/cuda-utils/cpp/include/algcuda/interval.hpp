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
 * @file algcuda/interval.hpp Interval number types.
 */

#ifndef ALGCUDA_INTERVAL_HPP_INCLUDED_
#define ALGCUDA_INTERVAL_HPP_INCLUDED_

#include <cfloat>
#include <iostream>
#include <iomanip>
#include <algcuda/macros.hpp>

namespace algcuda {
/**
 * @brief An wrapper for uncertain values.
 * @tparam Type Usually bool.
 */
template<typename Type> struct Uncertain {
	/**
	 * @brief Construct a new Uncertain object
	 */
	Uncertain() noexcept = default;
	/**
	 * @brief Construct a new Uncertain object by copying.
	 */
	Uncertain(const Uncertain&) noexcept = default;
	/**
	 * @brief Copy-assign an Uncertain object.
	 * @return Uncertain& *this
	 */
	Uncertain &operator=(const Uncertain&) noexcept = default;

	/**
	 * @brief Create a new Uncertain from its wrapped type.
	 * @param v 
	 */
	explicit __host__ __device__ Uncertain(Type v) noexcept : lb(v), ub(v) {}

	/**
	 * @brief Construct a new Uncertain from a lower and upper bound (wrapped type).
	 * 
	 * @param l 
	 * @param u 
	 */
	__host__ __device__ Uncertain(Type l, Type u) noexcept : lb(l), ub(u) {}

	/**
	 * @brief check if lower and upper bound are equal.
	 * @return bool
	 */
	bool __host__ __device__ is_certain() const noexcept {
		return lb == ub;
	}

	/**
	 * @brief Check if lower and upper bound are not equal.
	 * 
	 * @return bool
	 */
	bool __host__ __device__ is_uncertain() const noexcept {
		return lb != ub;
	}

	/**
	 * @brief Get the lower bound
	 * 
	 * @return Type 
	 */
	Type __host__ __device__ get_lb() const noexcept {
		return lb;
	}

	/**
	 * @brief Get the upper bound
	 * 
	 * @return Type 
	 */
	Type __host__ __device__ get_ub() const noexcept {
		return ub;
	}

	/**
	 * @brief Negate (defined as [!ub, !lb])
	 * 
	 * @return Uncertain 
	 */
	inline Uncertain __host__ __device__ operator!() const noexcept;

private:
	Type lb, ub;
};

/**
 * @brief Check if an uncertain bool is definitely true.
 * @param b 
 * @return bool
 */
static inline __host__ __device__ bool definitely(Uncertain<bool> b) {
	return b.get_lb();
}

/**
 * @brief Check if an uncertain bool is possibly true.
 * @param b 
 * @return bool
 */
static inline __host__ __device__ bool possibly(Uncertain<bool> b) {
	return b.get_ub();
}

/**
 * @brief Check if an uncertain bool is definitely false.
 * @param b 
 * @return bool
 */
static inline __host__ __device__ bool definitely_not(Uncertain<bool> b) {
	return !b.get_ub();
}

/**
 * @brief Check if an uncertain bool is possibly false.
 * @param b 
 * @return bool
 */
static inline __host__ __device__ bool possibly_not(Uncertain<bool> b) {
	return !b.get_lb();
}

static_assert(std::is_pod<Uncertain<bool>>::value, "Uncertain<bool> must be POD!");

template<> inline __host__ __device__ Uncertain<bool> Uncertain<bool>::operator!() const noexcept {
	return {!ub, !lb};
}

/**
 * @brief An interval number type.
 * 
 * @tparam NumType The number type the interval is based on (must be a floating-point type).
 */
template<typename NumType> class Interval {
public:
	static_assert(std::is_floating_point<NumType>::value, "Interval NumType must be floating-point type!");

	/**
	 * @brief Construct a new Interval object
	 */
	Interval() noexcept = default;

	/**
	 * @brief Construct a new Interval object by copying.
	 */
	Interval(const Interval&) noexcept = default;

	/**
	 * @brief Copy-assign a new interval object.
	 * @return Interval& *this
	 */
	Interval& operator=(const Interval&) noexcept = default;

	/**
	 * @brief Create a new interval containing a single value.
	 * @param v 
	 */
	explicit __host__ __device__ Interval(NumType v) noexcept :
		lb(v), ub(v)
	{}

	/**
	 * @brief Create a new interval from a lower and upper bound.
	 * @param l The lower bound.
	 * @param u The upper bound.
	 */
	__host__ __device__ Interval(NumType l, NumType u) noexcept :
		lb(l), ub(u)
	{}

	/**
	 * @brief Get the lower bound
	 * 
	 * @return NumType 
	 */
	NumType __host__ __device__ get_lb() const noexcept {
		return lb;
	}

	/**
	 * @brief Get the upper bound
	 * 
	 * @return NumType 
	 */
	NumType __host__ __device__ get_ub() const noexcept {
		return ub;
	}

	/**
	 * @brief Set the lower bound
	 * 
	 * @param n 
	 */
	void __host__ __device__ set_lb(NumType n) noexcept {
		lb = n;
	}

	/**
	 * @brief Set the upper bound
	 * 
	 * @param n 
	 */
	void __host__ __device__ set_ub(NumType n) noexcept {
		ub = n;
	}

	/**
	 * @brief Check if the interval contains the given number.
	 * 
	 * @param n 
	 * @return true 
	 * @return false 
	 */
	bool __host__ __device__ contains(NumType n) const noexcept {
		return lb <= n && n <= ub;
	}

	/**
	 * @brief Compute the reciprocal interval 1 / *this.
	 * 
	 * @return Interval<NumType> 
	 */
	inline Interval<NumType> reciprocal() const noexcept;
	
	/**
	 * @brief Add another interval to *this.
	 * 
	 * @param o 
	 * @return Interval& *this.
	 */
	inline __device__ Interval &operator+=(const Interval& o) noexcept;

	/**
	 * @brief Add a number to *this.
	 * 
	 * @param n
	 * @return Interval& *this.
	 */
	inline __device__ Interval &operator+=(NumType n) noexcept;

	/**
	 * @brief Subtract another interval from *this.
	 * 
	 * @param o 
	 * @return Interval& *this.
	 */
	inline __device__ Interval &operator-=(const Interval& o) noexcept;

	/**
	 * @brief Subtract a number from *this.
	 * 
	 * @param n
	 * @return Interval& *this.
	 */
	inline __device__ Interval &operator-=(NumType n) noexcept;

	/**
	 * @brief Multiply *this by an interval.
	 * 
	 * @param o
	 * @return Interval& *this.
	 */
	inline __device__ Interval &operator*=(const Interval& o) noexcept;

	/**
	 * @brief Multiply *this by a number.
	 * 
	 * @param n
	 * @return Interval& *this.
	 */
	inline __device__ Interval &operator*=(NumType n) noexcept;

	/**
	 * @brief Divide *this by an interval.
	 * 
	 * @param o
	 * @return Interval& *this.
	 */
	inline __device__ Interval &operator/=(const Interval& o) noexcept;

	/**
	 * @brief Divide *this by a number.
	 * 
	 * @param n
	 * @return Interval& *this.
	 */
	inline __device__ Interval &operator/=(NumType n) noexcept;

	/**
	 * @brief Compute the square root of *this.
	 * @return Interval The square root.
	 */
	inline __device__ Interval sqrt() const noexcept;

	/**
	 * @brief Compute the negation of *this.
	 * @return Interval The negation.
	 */
	Interval __device__  __host__ operator-() const noexcept { return {-ub, -lb}; }

	/**
	 * @brief Compute the square of *this.
	 * @return Interval The square.
	 */
	inline __device__ Interval square() const noexcept;

	/**
	 * @brief Compare two intervals.
	 * 
	 * @param n 
	 * @return Uncertain<bool> 
	 */
	Uncertain<bool> __device__ __host__ operator< (NumType n) const noexcept {
		return { ub < n, lb < n };
	}

	/**
	 * @brief Compare two intervals.
	 * 
	 * @param n 
	 * @return Uncertain<bool> 
	 */
	Uncertain<bool> __device__ __host__ operator> (NumType n) const noexcept {
		return { lb > n, ub > n };
	}

	/**
	 * @brief Compare two intervals.
	 * 
	 * @param n 
	 * @return Uncertain<bool> 
	 */
	Uncertain<bool> __device__ __host__ operator<=(NumType n) const noexcept {
		return { ub <= n, lb <= n };
	}

	/**
	 * @brief Compare two intervals.
	 * 
	 * @param n 
	 * @return Uncertain<bool> 
	 */
	Uncertain<bool> __device__ __host__ operator>=(NumType n) const noexcept {
		return { lb >= n, ub >= n };
	}

	/**
	 * @brief Compare two intervals.
	 * 
	 * @param o
	 * @return Uncertain<bool> 
	 */
	Uncertain<bool> __device__ __host__ operator< (const Interval& o) const noexcept {
		return {
			ub < o.lb, // lb = true iff we are definitely less than o
			lb < o.ub  // ub = true iff we are possibly less than o
		};
	}

	/**
	 * @brief Compare two intervals.
	 * 
	 * @param o
	 * @return Uncertain<bool> 
	 */
	Uncertain<bool> __device__ __host__ operator<=(const Interval& o) const noexcept {
		return {
			ub <= o.lb, // lb = true iff we are definitely less than or equal to o
			lb <= o.ub  // ub = true iff we are possibly   less than or equal to o
		};
	}

	/**
	 * @brief Compare two intervals.
	 * 
	 * @param o
	 * @return Uncertain<bool> 
	 */
	Uncertain<bool> __device__ __host__ operator==(const Interval& o) const noexcept {
		return {lb == ub && o.lb == o.ub && lb == o.lb, !(lb > o.ub) && !(ub < o.lb) };
	}

	/**
	 * @brief Compare two intervals.
	 * 
	 * @param o
	 * @return Uncertain<bool> 
	 */
	Uncertain<bool> __device__ __host__ operator==(NumType n) const noexcept {
		return { lb == ub && lb == n, !(lb > n) && !(ub < n) };
	}

	/**
	 * @brief Compute the intersection of two intervals.
	 * 
	 * @param o 
	 * @return Interval 
	 */
	Interval __device__ __host__ intersect(const Interval& o) const noexcept {
		return Interval(lb < o.lb ? o.lb : lb, ub > o.ub ? o.ub : ub);
	}

	/**
	 * @brief Intersect *this with o and update *this.
	 * 
	 * @param o 
	 * @return Interval& *this
	 */
	Interval& __device__ __host__ do_intersect(const Interval& o) noexcept {
		if(o.lb > lb)
			lb = o.lb;

		if(o.ub < ub)
			ub = o.ub;

		return *this;
	}

	/**
	 * @brief Compute the smallest interval containing the union of two intervals.
	 * 
	 * @param o 
	 * @return Interval 
	 */
	Interval __device__ __host__ join(const Interval& o) const noexcept {
		return Interval(lb > o.lb ? o.lb : lb, ub < o.ub ? o.ub : ub);
	}

	/**
	 * @brief Compute the smallest interval containing the union of *this and o; update *this.
	 * 
	 * @param o 
	 * @return Interval& *this
	 */
	Interval& __device__ __host__ do_join(const Interval& o) noexcept {
		if(o.lb < lb)
			lb = o.lb;

		if(o.ub > ub)
			ub = o.ub;

		return *this;
	}

	/**
	 * @brief Set the upper bound of *this to ub if ub is below the current upper bound.
	 * @param ub 
	 */
	void __device__ __host__ tighten_ub(const NumType ub) noexcept {
		if(ub < this->ub) {
			this->ub = ub;
		}
	}

	/**
	 * @brief Set the lower bound of *this to lb if lb is above the current lower bound.
	 * @param lb 
	 */
	void __device__ __host__ tighten_lb(const NumType lb) noexcept {
		if(this->lb < lb) {
			this->lb = lb;
		}
	}

	/**
	 * @brief Check if *this is empty.
	 * 
	 * @return true 
	 * @return false 
	 */
	bool __device__ __host__ empty() const noexcept {
		return lb > ub;
	}

	/**
	 * @brief Compute an interval for the absolute value of *this.
	 * @return  
	 */
	__device__ __host__ Interval abs() const noexcept {
		NumType l{0}, u{0};

		if(ub > 0) {
			u = ub;
		} else {
			l = -ub;
		}

		if(lb < 0) {
			u = u < -lb ? -lb : u;
		} else {
			l = lb;
		}

		return {l, u};
	}

private:
	NumType lb, ub;

	static inline __device__ NumType mul_rd(NumType n1, NumType n2) noexcept;
	static inline __device__ NumType mul_ru(NumType n1, NumType n2) noexcept;
	static inline __device__ NumType div_rd(NumType n1, NumType n2) noexcept;
	static inline __device__ NumType div_ru(NumType n1, NumType n2) noexcept;
	static inline __device__ NumType infty() noexcept;
};

template<typename NumType> static inline __device__ __host__ Uncertain<bool> operator==(NumType nt, const Interval<NumType>& i) noexcept {
	return i == nt;
}

template<typename NumType> static inline __device__ __host__ Uncertain<bool> operator!=(const Interval<NumType>& i, NumType nt) noexcept {
	return !(i == nt);
}

template<typename NumType> static inline __device__ __host__ Uncertain<bool> operator!=(NumType nt, const Interval<NumType>& i) noexcept {
	return i != nt;
}

template<typename NumType> static inline __device__ __host__ Uncertain<bool> operator!=(const Interval<NumType>& i1, const Interval<NumType>& i2) noexcept {
	return !(i1 == i2);
}

template<typename NumType> static inline __device__ __host__ Uncertain<bool> operator< (NumType nt, const Interval<NumType>& i) noexcept {
	return i > nt;
}

template<typename NumType> static inline __device__ __host__ Uncertain<bool> operator> (NumType nt, const Interval<NumType>& i) noexcept {
	return i < nt;
}

template<typename NumType> static inline __device__ __host__ Uncertain<bool> operator<=(NumType nt, const Interval<NumType>& i) noexcept {
	return i >= nt;
}

template<typename NumType> static inline __device__ __host__ Uncertain<bool> operator>=(NumType nt, const Interval<NumType>& i) noexcept {
	return i <= nt;
}

template<typename NumType> static inline __device__ __host__ Uncertain<bool> operator> (const Interval<NumType>& i1, const Interval<NumType>& i2) noexcept {
	return i2 < i1;
}

template<typename NumType> static inline __device__ __host__ Uncertain<bool> operator>=(const Interval<NumType>& i1, const Interval<NumType>& i2) noexcept {
	return i2 <= i1;
}

static_assert(std::is_pod<Interval<float>>::value,  "Float intervals must be POD!");
static_assert(std::is_pod<Interval<double>>::value, "Double intervals must be POD!");

/**
 * @brief Output an interval to a stream.
 * 
 * @tparam NumType 
 * @param output 
 * @param iv 
 * @return std::ostream& The stream.
 */
template<typename NumType>
static inline __host__ std::ostream& operator<<(std::ostream& output, const Interval<NumType>& iv)
{
	std::ios_base::fmtflags f = output.flags(std::ios::right);
	std::streamsize p = output.precision(19);
	output << '[' << std::setw(26) << iv.get_lb() << ", " << std::setw(26) << iv.get_ub() << ']';
	output.flags(f);
	output.precision(p);
	return output;
}

template<typename CharType, typename NumType>
static inline __host__ std::basic_istream<CharType>& operator>>(std::basic_istream<CharType>& input, Interval<NumType>& iv) {
	CharType c;
	if(!(input >> c)) {
		return input;
	}

	if(c == '[') {
		NumType n1;
		NumType n2;

		CharType comma;
		CharType end;

		if(!(input >> n1) || !(input >> comma)) {
			return input;
		}

		if(comma != ',' && comma != ';') {
			input.setstate(std::ios_base::failbit);
			return input;
		}

		if(!(input >> n2) || !(input >> end)) {
			return input;
		}

		if(end != ']') {
			input.setstate(std::ios_base::failbit);
			return input;
		}

		iv = Interval<NumType>{n1,n2};
	} else {
		if(!input.putback(c)) {
			return input;
		}
		
		NumType result;
		if(!(input >> result)) {
			return input;
		}
		iv = Interval<NumType>{result,result};
	}

	return input;
}

/**
 * @brief Lexicographically compare two intervals (mostly for sorting purposes).
 */
struct Interval_compare {
	/**
	 * @brief Check if interval i1 is lexicographically less than interval i2.
	 * 
	 * @tparam NumType 
	 * @param i1 
	 * @param i2 
	 * @return bool 
	 */
	template<typename NumType> __device__ __host__ bool operator()(const Interval<NumType>& i1, const Interval<NumType>& i2) const noexcept {
		return i1.get_lb() < i2.get_lb() || (i1.get_lb() == i2.get_lb() && i1.get_ub() < i2.get_ub());
	}
};
}

namespace std {
/**
 * @brief Define a hash function for intervals.
 * 
 * @tparam NumType 
 */
template<typename NumType> struct hash<algcuda::Interval<NumType>> {
	std::size_t __host__ __device__ operator()(const algcuda::Interval<NumType>& i) const noexcept {
		std::size_t h1 = std::hash<NumType>{}(i.get_lb());
		std::size_t h2 = std::hash<NumType>{}(i.get_ub());
		return h2 + static_cast<std::size_t>(0x9e3779b97f4a7c15ull) + (h1 << 6) + (h1 << 2);
	}
};
}

#endif
