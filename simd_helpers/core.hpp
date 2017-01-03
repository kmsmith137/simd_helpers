#ifndef _SIMD_HELPERS_CORE_HPP
#define _SIMD_HELPERS_CORE_HPP

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


// This boilerplate defines mask types, which are returned by comparison operators.
//    smask_t<T> = scalar mask type (signed integer type with same size as T)
//    smask_t<T,S> = simd mask type (same as simd_t<smask_t<T>,S>)

template<typename T, unsigned int S> struct _smask_t { using type = simd_t<typename _smask_t<T,1>::type,S>; };
template<typename T, unsigned int S=1> using smask_t = typename _smask_t<T,S>::type;

template<> struct _smask_t<int,1> { using type = int; };
template<> struct _smask_t<float,1> { using type = int; };
template<> struct _smask_t<int64_t,1> { using type = int64_t; };
template<> struct _smask_t<double,1> { using type = int64_t; };


// machine_epsilon<T>(): fractional roundoff error for floating-point type T
template<typename T> inline constexpr T machine_epsilon();
template<> inline constexpr float machine_epsilon()  { return 1.19e-07; }
template<> inline constexpr double machine_epsilon() { return 2.22e-16; }


// Here is a pseudo-declaration of simd_t<T,S>
//
//  struct simd_t<T,S>
//  {
//      simd_t();                                   // default constructor does not initialize 
//      simd_t(__m256 x);                           // construct from low-level simd type (__m256 or similar)
//      simd_t(T x);                                // construct from scalar (note: for integer-valued types, -1 is faster than +1)
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
//      // Reminder: you may need to use a syntax such as
//      //   x.template extract<M> ();
//
//      template<unsigned int M> inline T extract() const;                       // extracts M-th element of simd vector
//      template<unsigned int M> inline simd_t<T,(S/2)> extract_half() const;    // extracts lower or upper half (only for 256-bit types)
//
//      // Arithmetic operators
//
//      simd_t<T,S> &operator+=(simd_t<T,S> x);
//      simd_t<T,S> &operator-=(simd_t<T,S> x);
//      simd_t<T,S> &operator*=(simd_t<T,S> x);                   // note: multiplication not defined for integer types
//      simd_t<T,S> &operator/=(simd_t<T,S> x);                   // note: division not defined for integer types
//      simd_t<T,S> operator+(simd_t<T,S> x) const;
//      simd_t<T,S> operator-(simd_t<T,S> x) const;
//      simd_t<T,S> operator*(simd_t<T,S> x) const;               // note: multiplication not defined for integer types
//      simd_t<T,S> operator/(simd_t<T,S> x) const;               // note: division not defined for integer types
//      simd_t<T,S> operator-() const;
//
//      simd_t<T,S> abs() const;
//      simd_t<T,S> sqrt() const;                // defined for floating-point types T
//      simd_t<T,S> min(simd_t<T,S> x) const;
//      simd_t<T,S> max(simd_t<T,S> x) const;
//
//      // "Horizontal reducers"
//
//      simd_t<T,S> horizontal_sum() const;      // returns [ t ... t ], where t is the sum of all elements in the simd_t
//      T sum() const;                           // returns t, where t is the sum of all elements in the simd_t
//
//      // "Boolean reducers": these are only defined for integer types, and return either 0 or 1.
//      // Note: it would be possible to define floating-point versions which test positivity (sign bits).
//
//      int is_all_ones() const;                                 // returns true if all bits are one.
//      int is_all_zeros() const;                                // returns true if all bits are zero
//      int is_all_zeros_masked(smask_t<T,S> t) const;           // returns true if all bits in (this & t) are zero
//      int is_all_zeros_inverse_masked(smask_t<T,S> t) const;   // returns true if all bits in (this & ~t) are zero
//
//      // Comparison operators.
//      // The output of a comparison operator is -1 (0xff..) for 'true' or 0 for 'false'.
//      // Floating-point comparisons are quiet and ordered (e.g. NaN==NaN evaluates to false).
//      // Note: in the integer case, eq/gt/lt will be a little more efficient than ne/ge/le.
//
//      smask_t<T,S> compare_eq(simd_t<T,S> x) const;
//      smask_t<T,S> compare_ne(simd_t<T,S> x) const;
//      smask_t<T,S> compare_ge(simd_t<T,S> x) const;
//      smask_t<T,S> compare_gt(simd_t<T,S> x) const;
//      smask_t<T,S> compare_le(simd_t<T,S> x) const;
//      smask_t<T,S> compare_lt(simd_t<T,S> x) const;
//
//      // Masking operators.  These are useful for processing the output of a comparison.
//      // Note: another very useful function is the non-member function
//      //    simd_t<T,S> blendv(smask_t<T,S> mask, simd_t<T,S> a simd_t<T,S> b);   // mask ? a : b
//
//      simd_t<T,S> apply_mask(smask_t<T,S> x) const;           // mask ? x : 0
//      simd_t<T,S> apply_inverse_mask(smask_t<T,S> x) const;   // mask ? 0 : x
//
//      // Bitwise operators, defined for integer types T.
//      simd_t<T,S> bitwise_and(smask_t<T,S> x) const;
//      simd_t<T,S> bitwise_andnot(smask_t<T,S> x) const;   // (this & ~x)   [ not (~this & x) as in the underlying asm instruction ]
//      simd_t<T,S> bitwise_or(simd_t<T,S> x) const;
//      simd_t<T,S> bitwise_xor(simd_t<T,S> x) const;
//      simd_t<T,S> bitwise_not() const;
//
//  };


template<typename T, unsigned int S> inline simd_t<T,S> operator*(T a, simd_t<T,S> b) { return simd_t<T,S>(a) * b; }
template<typename T, unsigned int S> inline simd_t<T,S> operator*(simd_t<T,S> a, T b) { return a * simd_t<T,S>(b); }


}  // namespace simd_helpers

#endif  // _SIMD_HELPERS_CORE_HPP
