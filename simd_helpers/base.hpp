#ifndef _SIMD_HELPERS_BASE_HPP
#define _SIMD_HELPERS_BASE_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

#include "immintrin.h"

#ifndef __AVX2__
#define _mm256_shuffle_epi32(a, imm8)   _mm256_castps_si256(_mm256_permute_ps(_mm256_castsi256_ps(a), imm8))
#define _mm256_blend_epi32(a, b, imm8)  _mm256_castps_si256(_mm256_blend_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b), imm8))
#endif

namespace simd_helpers {
#if 0
}  // pacify emacs c-mode
#endif


// This basic class is used everywhere!
// See extended comment below for more info.
template<typename T, unsigned int S> struct simd_t;


// machine_epsilon<T>(): fractional roundoff error for floating-point type T
template<typename T> inline constexpr T machine_epsilon();
template<> inline constexpr float machine_epsilon()  { return 1.19e-07; }
template<> inline constexpr double machine_epsilon() { return 2.22e-16; }


// Here is a pseudo-declaration of simd_t<T,S>
//
//  struct simd_t<T,S>
//  {
//      using mask_t = int;                         // integer type with same size as T
//
//      simd_t();                                   // default constructor does not initialize 
//      simd_t(__m256 x);                           // construct from low-level simd type (__m256 or similar)
//      simd_t(T x);                                // construct from scalar
//      simd_t(simd_t<T,S/2> x, simd_t<T,S/2> y);   // construct 256-bit simd_t from two 128-bit simd_t's
//
//      static simd_t<T,S> zero();                  // factory function returning all zeros
//      static simd_t<T,S> range();                 // returns [ 0, 1, ..., S-1 ]
//
//      static simd_t<T,S> load(const T *p);
//      static simd_t<T,S> loadu(const T *p);
//
//      void store(T *p);
//      void storeu(T *p);
//
//      // Returns M-th element of the simd_vector.  You may need to use the syntax:
//      //   x.template extract<M> ();
//      template<unsigned int M> inline int extract() const;
//
//      // Arithmetic operators
//      simd_t<T,S> &operator+=(simd_t<T,S> x);
//      simd_t<T,S> &operator-=(simd_t<T,S> x);
//      simd_t<T,S> &operator*=(simd_t<T,S> x);                   // note: multiplication not defined for integer types
//      simd_t<T,S> &operator/=(simd_t<T,S> x);                   // note: division not defined for integer types
//      simd_t<T,S> operator+(simd_t<T,S> x) const;
//      simd_t<T,S> operator-(simd_t<T,S> x) const;
//      simd_t<T,S> operator*(simd_t<T,S> x) const;               // note: multiplication not defined for integer types
//      simd_t<T,S> operator/(simd_t<T,S> x) const;               // note: division not defined for integer types
//
//      simd_t<T,S> bitwise_and(simd_t<mask_t,S> x) const;        // defined for all T
//      simd_t<T,S> bitwise_andnot(simd_t<mask_t,S> x) const;     // defined for all T
//      simd_t<T,S> bitwise_or(simd_t<T,S> x) const;              // defined for integer T
//      simd_t<T,S> bitwise_xor(simd_t<T,S> x) const;             // defined for integer T
//
//      // The output of a comparison operator is -1 (0xff..) for 'true' or 0 for 'false'.
//      // Floating-point comparisons are quiet and ordered (e.g. NaN==NaN evaluates to false).
//      // Note: in the integer case, compare_eq() and compare_gt() may be more efficient than others.
//
//      simd_t<mask_t,S> compare_eq(simd_t<T,S> x) const;
//      simd_t<mask_t,S> compare_ne(simd_t<T,S> x) const;
//      simd_t<mask_t,S> compare_ge(simd_t<T,S> x) const;
//      simd_t<mask_t,S> compare_gt(simd_t<T,S> x) const;
//      simd_t<mask_t,S> compare_le(simd_t<T,S> x) const;
//      simd_t<mask_t,S> compare_lt(simd_t<T,S> x) const;
//
//      simd_t<T,S> abs() const;                 // not defined for int64_t
//      simd_t<T,S> sqrt() const;                // defined for floating-point T
//      simd_t<T,S> rsqrt() const;               // defined for floating-point T
//      simd_t<T,S> min(simd_t<T,S> x) const;
//      simd_t<T,S> max(simd_t<T,S> x) const;
//
//      simd_t<T,S> horizontal_sum() const;      // returns [ t ... t ], where t is the sum of all elements in the simd_t
//      T sum() const;                           // returns t, where t is the sum of all elements in the simd_t
//
//      // Defined only for integer types
//      int testzero_bitwise_and(simd_t<T,S> t) const;       // returns true if all bits in (this & t) are zero
//      int testzero_bitwise_andnot(simd_t<T,S> t) const;    // returns true if all bits in (this & ~t) are zero
//      int is_all_zeros() const;                            // returns true if all bits are zero
//      int is_all_ones() const;                             // returns true if all bits are one
//  };


template<typename T, unsigned int S> inline simd_t<T,S> operator*(T a, simd_t<T,S> b) { return simd_t<T,S>(a) * b; }
template<typename T, unsigned int S> inline simd_t<T,S> operator*(simd_t<T,S> a, T b) { return a * simd_t<T,S>(b); }


}  // namespace simd_helpers

#endif
