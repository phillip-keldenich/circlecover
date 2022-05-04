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

#ifndef ALGCUDA_INTERVAL_CUH_INCLUDED_
#define ALGCUDA_INTERVAL_CUH_INCLUDED_

/**
 * @file Implementation of interval arithmetic operations on CUDA devices.
 */

#include <algcuda/interval.hpp>
#include <algcuda/exit.cuh>

namespace algcuda {
	template<> inline  __device__ Interval<float>& Interval<float>::operator+=(const Interval& o) noexcept {
		float l = __fadd_rd(lb, o.lb);
		float u = __fadd_ru(ub, o.ub);
		lb = l;
		ub = u;
		return *this;
	}

	template<> inline __device__ Interval<double>& Interval<double>::operator+=(const Interval& o) noexcept {
		double l = __dadd_rd(lb, o.lb);
		double u = __dadd_ru(ub, o.ub);
		lb = l;
		ub = u;
		return *this;
	}

	template<typename NumType> inline __device__ Interval<NumType>& Interval<NumType>::operator+=(NumType n) noexcept {
		return *this += Interval<NumType>{n,n};
	}

	template<typename NumType> inline __device__ Interval<NumType> operator+(const Interval<NumType>& a, const Interval<NumType>& b) noexcept {
		Interval<NumType> result(a);
		result += b;
		return result;
	}

	template<typename NumType> inline __device__ Interval<NumType> operator+(const Interval<NumType>& a, NumType b) noexcept {
		Interval<NumType> result(a);
		result += b;
		return result;
	}

	template<typename NumType> inline __device__ Interval<NumType> operator+(NumType a, const Interval<NumType>& b) noexcept {
		Interval<NumType> result(b);
		result += a;
		return result;
	}

	template<> inline __device__ Interval<float>& Interval<float>::operator-=(const Interval& o) noexcept {
		float l = __fadd_rd(lb, -o.ub);
		float u = __fadd_ru(ub, -o.lb);
		lb = l;
		ub = u;
		return *this;
	}

	template<> inline __device__ Interval<double>& Interval<double>::operator-=(const Interval& o) noexcept {
		double l = __dadd_rd(lb, -o.ub);
		double u = __dadd_ru(ub, -o.lb);
		lb = l;
		ub = u;
		return *this;
	}

	template<typename NumType> inline __device__ Interval<NumType>& Interval<NumType>::operator-=(NumType n) noexcept {
		return *this -= Interval<NumType>{n,n};
	}

	template<typename NumType> inline __device__ Interval<NumType> operator-(const Interval<NumType>& a, const Interval<NumType>& b) noexcept {
		Interval<NumType> result(a);
		result -= b;
		return result;
	}

	template<typename NumType> inline __device__ Interval<NumType> operator-(const Interval<NumType>& a, NumType b) noexcept {
		Interval<NumType> result(a);
		result -= b;
		return result;
	}

	template<typename NumType> inline __device__ Interval<NumType> operator-(NumType a, const Interval<NumType>& b) noexcept {
		Interval<NumType> result(a,a);
		result -= b;
		return result;
	}

	template<> inline float __device__ Interval<float>::mul_rd(float a, float b) noexcept {
		return __fmul_rd(a,b);
	}

	template<> inline float __device__ Interval<float>::mul_ru(float a, float b) noexcept {
		return __fmul_ru(a,b);
	}

	template<> inline double __device__ Interval<double>::mul_rd(double a, double b) noexcept {
		return __dmul_rd(a,b);
	}

	template<> inline double __device__ Interval<double>::mul_ru(double a, double b) noexcept {
		return __dmul_ru(a,b);
	}

#ifdef ALGCUDA_UTILS_INTERVAL_USE_BRANCHING_MUL
	template<typename NumType> inline __device__ Interval<NumType>& Interval<NumType>::operator*=(const Interval& o) noexcept {
		NumType l, u;

		if(lb >= 0) {
			if(o.lb >= 0) {
				l = mul_rd(lb, o.lb);
				u = mul_ru(ub, o.ub);
			} else if(o.ub <= 0) {
				l = mul_rd(ub, o.lb);
				u = mul_ru(lb, o.ub);
			} else {
				l = mul_rd(ub, o.lb);
				u = mul_ru(ub, o.ub);
			}
		} else if(ub <= 0) {
			if(o.lb >= 0) {
				l = mul_rd(lb, o.ub);
				u = mul_ru(ub, o.lb);
			} else if(o.ub <= 0) {
				l = mul_rd(ub, o.ub);
				u = mul_ru(lb, o.lb);
			} else {
				l = mul_rd(lb, o.ub);
				u = mul_rd(lb, o.lb);
			}
		} else {
			if(o.lb >= 0) {
				l = mul_rd(lb, o.ub);
				u = mul_ru(ub, o.ub);
			} else if(o.ub <= 0) {
				l = mul_rd(ub, o.lb);
				u = mul_rd(lb, o.lb);
			} else {
				NumType l1 = mul_rd(lb, o.ub);
				NumType l2 = mul_rd(ub, o.lb);
				NumType u1 = mul_ru(lb, o.lb);
				NumType u2 = mul_ru(ub, o.ub);

				l = l1 < l2 ? l1 : l2;
				u = u1 > u2 ? u1 : u2;
			}
		}

		lb = l;
		ub = u;
		return *this;
	}
#else
	template<typename NumType> inline __device__ Interval<NumType>& Interval<NumType>::operator*=(const Interval& o) noexcept {
		double l1 = mul_rd(lb, o.lb), l2 = mul_rd(lb, o.ub), l3 = mul_rd(ub, o.lb), l4 = mul_rd(ub, o.ub);
		double u1 = mul_ru(lb, o.lb), u2 = mul_ru(lb, o.ub), u3 = mul_ru(ub, o.lb), u4 = mul_ru(ub, o.ub);
		
		double ll1 = l1 < l2 ? l1 : l2;
		double ll2 = l3 < l4 ? l3 : l4;
		double uu1 = u1 > u2 ? u1 : u2;
		double uu2 = u3 > u4 ? u3 : u4;
		lb = ll1 < ll2 ? ll1 : ll2;
		ub = uu1 > uu2 ? uu1 : uu2;
		return *this;
	}
#endif

	template<typename NumType> inline Interval<NumType>& __device__ Interval<NumType>::operator*=(NumType n) noexcept {
		if(n >= 0) {
			lb = mul_rd(lb, n);
			ub = mul_rd(ub, n);
		} else {
			NumType l = mul_rd(ub, n);
			NumType u = mul_ru(lb, n);
			lb = l;
			ub = u;
		}

		return *this;
	}

	template<typename NumType> inline __device__ Interval<NumType> operator*(const Interval<NumType>& a, const Interval<NumType>& b) noexcept {
		Interval<NumType> result(a);
		result *= b;
		return result;
	}

	template<typename NumType> inline __device__ Interval<NumType> operator*(const Interval<NumType>& a, NumType b) noexcept {
		Interval<NumType> result(a);
		result *= b;
		return result;
	}

	template<typename NumType> inline __device__ Interval<NumType> operator*(NumType a, const Interval<NumType>& b) noexcept {
		Interval<NumType> result(b);
		result *= a;
		return result;
	}

	template<> inline Interval<float> __device__ Interval<float>::reciprocal() const noexcept {
		if(lb > 0) {
			return {__frcp_rd(ub), __frcp_ru(lb)};
		} else if(ub < 0) {
			return {__frcp_rd(lb), __frcp_ru(ub)};
		} else {
			return {-(FLT_MAX * FLT_MAX), FLT_MAX * FLT_MAX};
		}
	}

	template<> inline Interval<double> __device__ Interval<double>::reciprocal() const noexcept {
		if(lb > 0) {
			return {__drcp_rd(ub), __drcp_ru(lb)};
		} else if(ub < 0) {
			return {__drcp_rd(lb), __drcp_ru(ub)};
		} else {
			return {-(DBL_MAX * DBL_MAX), DBL_MAX * DBL_MAX};
		}
	}

	template<> inline __device__ float Interval<float>::infty() noexcept {
		return FLT_MAX * FLT_MAX;
	}

	template<> inline __device__ double Interval<double>::infty() noexcept {
		return DBL_MAX * DBL_MAX;
	}

	template<> inline __device__ float Interval<float>::div_rd(float a, float b) noexcept {
		return __fdiv_rd(a,b);
	}

	template<> inline __device__ double Interval<double>::div_rd(double a, double b) noexcept {
		return __ddiv_rd(a,b);
	}

	template<> inline __device__ float Interval<float>::div_ru(float a, float b) noexcept {
		return __fdiv_ru(a,b);
	}

	template<> inline __device__ double Interval<double>::div_ru(double a, double b) noexcept {
		return __ddiv_ru(a,b);
	}

	template<typename NumType> inline Interval<NumType>& __device__ Interval<NumType>::operator/=(NumType nt) noexcept {
		NumType l,u;

		if(nt > 0) {
			l = div_rd(lb, nt);
			u = div_ru(ub, nt);
		} else if(nt <= 0) {
			l = div_rd(ub, nt);
			u = div_ru(lb, nt);
		}

		lb = l;
		ub = u;
		return *this;
	}

	template<typename NumType> inline Interval<NumType>& __device__ Interval<NumType>::operator/=(const Interval<NumType>& o) noexcept {
		NumType l, u;
		
		if(o.lb > 0) {
			if(lb >= 0) {
				l = div_rd(lb, o.ub);
				u = div_ru(ub, o.lb);
			} else if(ub <= 0) {
				l = div_rd(lb, o.lb);
				u = div_ru(ub, o.ub);
			} else {
				l = div_rd(lb, o.lb);
				u = div_ru(ub, o.lb);
			}
		} else if(o.ub < 0) {
			if(lb >= 0) {
				l = div_rd(ub, o.ub);
				u = div_ru(lb, o.lb);
			} else if(ub <= 0) {
				l = div_rd(ub, o.lb);
				u = div_ru(lb, o.ub);
			} else {
				l = div_rd(ub, o.ub);
				u = div_ru(lb, o.ub);
			}
		} else {
			u = infty();
			l = -u;
		}

		lb = l;
		ub = u;
		return *this;
	}

	template<typename NumType> inline __device__ Interval<NumType> operator/(const Interval<NumType>& a, const Interval<NumType>& b) noexcept {
		Interval<NumType> result(a);
		result /= b;
		return result;
	}

	template<typename NumType> inline __device__ Interval<NumType> operator/(const Interval<NumType>& a, NumType b) noexcept {
		Interval<NumType> result(a);
		result /= b;
		return result;
	}

	template<typename NumType> inline __device__ Interval<NumType> operator/(NumType a, const Interval<NumType>& b) noexcept {
		Interval<NumType> result(a, a);
		result /= b;
		return result;
	}

	template<> inline __device__ Interval<float> Interval<float>::sqrt() const noexcept {
		if(lb < 0.0) {
			printf("Interval with possibly negative value passed to Interval<float>::sqrt(): [%.19g,%.19g]\n", (double)lb, (double)ub);
			trap();
		}

		return { __fsqrt_rd(lb), __fsqrt_ru(ub) };
	}

	template<> inline __device__ Interval<double> Interval<double>::sqrt() const noexcept {
		if(lb < 0.0) {
			printf("Interval with possibly negative value passed to Interval<double>::sqrt(): [%.19g,%.19g]\n", lb, ub);
			trap();
		}

		return { __dsqrt_rd(lb), __dsqrt_ru(ub) };
	}

	template<typename NumType> Interval<NumType> __device__ sqrt(const Interval<NumType>& interval) {
		return interval.sqrt();
	}

	template<typename NumType> Interval<NumType> __device__ Interval<NumType>::square() const noexcept {
		Interval a = abs();
		return {__dmul_rd(a.get_lb(), a.get_lb()), __dmul_ru(a.get_ub(), a.get_ub())};
	}
}

#endif

