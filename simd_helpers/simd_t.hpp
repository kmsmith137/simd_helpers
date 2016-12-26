#ifndef _SIMD_HELPERS_SIMD_T_HPP
#define _SIMD_HELPERS_SIMD_T_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

#include <cmath>
#include <sstream>
#include "immintrin.h"

namespace simd_helpers {
#if 0
}  // pacify emacs c-mode
#endif


template<typename T, unsigned int S> struct simd_t;


template<typename T> inline constexpr T machine_epsilon();
template<> inline constexpr float machine_epsilon()  { return 1.19e-07; }
template<> inline constexpr double machine_epsilon() { return 2.22e-16; }


// -------------------------------------------------------------------------------------------------


template<> struct simd_t<int,4>
{
    __m128i x;

    simd_t() { }
    simd_t(__m128i y) { x = y; }
    simd_t(int y)     { x = _mm_set1_epi32(y); }

    static inline simd_t<int,4> zero()  { return _mm_setzero_si128(); }
    static inline simd_t<int,4> range() { return _mm_set_epi32(3, 2, 1, 0); }

    static inline simd_t<int,4> load(const int *p)  { return _mm_load_si128((const __m128i *) p); }
    static inline simd_t<int,4> loadu(const int *p) { return _mm_loadu_si128((const __m128i *) p); }

    inline void store(int *p) const  { _mm_store_si128((__m128i *)p, x); }
    inline void storeu(int *p) const { _mm_storeu_si128((__m128i *)p, x); }

    // The comparison operators return all ones (0xff..) if "true".
    inline simd_t<int,4> compare_eq(simd_t<int,4> t) const  { return _mm_cmpeq_epi32(x, t.x); }
    inline simd_t<int,4> compare_gt(simd_t<int,4> t) const  { return _mm_cmpgt_epi32(x, t.x); }
    inline simd_t<int,4> compare_lt(simd_t<int,4> t) const  { return _mm_cmplt_epi32(x, t.x); }

    inline simd_t<int,4> min(simd_t<int,4> t) const { return _mm_min_epi32(x, t.x); }
    inline simd_t<int,4> max(simd_t<int,4> t) const { return _mm_max_epi32(x, t.x); }

    inline simd_t<int,4> bitwise_and(simd_t<int,4> t) const     { return _mm_and_si128(x, t.x); }
    inline simd_t<int,4> bitwise_or(simd_t<int,4> t) const      { return _mm_or_si128(x, t.x); }
    inline simd_t<int,4> bitwise_xor(simd_t<int,4> t) const     { return _mm_xor_si128(x, t.x); }
    inline simd_t<int,4> bitwise_andnot(simd_t<int,4> t) const  { return _mm_andnot_si128(x, t.x); }

    inline int testzero_bitwise_and(simd_t<int,4> t) const     { return _mm_testz_si128(x, t.x); }
    inline int testzero_bitwise_andnot(simd_t<int,4> t) const  { return _mm_testc_si128(t.x, x); }
    inline int test_all_zeros() const                          { return _mm_testz_si128(x, x); }
    inline int test_all_ones() const                           { return _mm_test_all_ones(x); }

    // Note: you might need to call this with the weird-looking syntax
    //    x.template extract<M> ();
    template<unsigned int M> inline int extract() const  { return _mm_extract_epi32(x, M); }

    inline simd_t<int,4> horizontal_sum() const
    {
	__m128i y = x + _mm_shuffle_epi32(x, 0xb1);  // (2301)_4 = 0xb1
	return y + _mm_shuffle_epi32(y, 0x4e);       // (1032)_4 = 0x4e
    }

    inline int sum() const { return _mm_extract_epi32(horizontal_sum().x, 0); }
};


// -------------------------------------------------------------------------------------------------


template<> struct simd_t<int,8>
{
    __m256i x;

    simd_t() { }
    simd_t(__m256i y) { x = y; }
    simd_t(int y)     { x = _mm256_set1_epi32(y); }

    static inline simd_t<int,8> zero()  { return _mm256_setzero_si256(); }
    static inline simd_t<int,8> range() { return _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0); }

    static inline simd_t<int,8> load(const int *p)  { return _mm256_load_si256((const __m256i *) p); }
    static inline simd_t<int,8> loadu(const int *p) { return _mm256_loadu_si256((const __m256i *) p); }

    inline void store(int *p) const  { _mm256_store_si256((__m256i *)p, x); }
    inline void storeu(int *p) const { _mm256_storeu_si256((__m256i *)p, x); }

    // The comparison operators return all ones (0xff..) if "true".
    inline simd_t<int,8> compare_eq(simd_t<int,8> t) const  { return _mm256_cmpeq_epi32(x, t.x); }
    inline simd_t<int,8> compare_gt(simd_t<int,8> t) const  { return _mm256_cmpgt_epi32(x, t.x); }

    inline simd_t<int,8> min(simd_t<int,8> t) const { return _mm256_min_epi32(x, t.x); }
    inline simd_t<int,8> max(simd_t<int,8> t) const { return _mm256_max_epi32(x, t.x); }

    inline simd_t<int,8> bitwise_and(simd_t<int,8> t) const     { return _mm256_and_si256(x, t.x); }
    inline simd_t<int,8> bitwise_or(simd_t<int,8> t) const      { return _mm256_or_si256(x, t.x); }
    inline simd_t<int,8> bitwise_xor(simd_t<int,8> t) const     { return _mm256_xor_si256(x, t.x); }
    inline simd_t<int,8> bitwise_andnot(simd_t<int,8> t) const  { return _mm256_andnot_si256(x, t.x); }

    inline int testzero_bitwise_and(simd_t<int,8> t) const     { return _mm256_testz_si256(x, t.x); }
    inline int testzero_bitwise_andnot(simd_t<int,8> t) const  { return _mm256_testc_si256(t.x, x); }
    inline int test_all_zeros() const                          { return _mm256_testz_si256(x, x); }
    inline int test_all_ones() const                           { return _mm256_testc_si256(x, _mm256_set1_epi32(-1)); }

    template<unsigned int M> inline int extract() const  { return _mm256_extract_epi32(x,M); }

    inline simd_t<int,8> horizontal_sum() const
    {
	__m256i y = x + _mm256_shuffle_epi32(x, 0xb1);  // (2301)_4 = 0xb1
	y += _mm256_shuffle_epi32(y, 0x4e);             // (1032)_4 = 0x4e
	return y + _mm256_permute2f128_si256(y, y, 0x01);
    }

    inline int sum() const { return _mm256_extract_epi32(horizontal_sum().x, 0); }
};


// -------------------------------------------------------------------------------------------------


template<> struct simd_t<float,4>
{
    __m128 x;

    simd_t() { }
    simd_t(__m128 y) { x = y; }
    simd_t(float y)  { x = _mm_set1_ps(y); }

    static inline simd_t<float,4> zero()  { return _mm_setzero_ps(); }
    static inline simd_t<float,4> range() { return _mm_set_ps(3., 2., 1., 0.); }

    static inline simd_t<float,4> load(const float *p)  { return _mm_load_ps(p); }
    static inline simd_t<float,4> loadu(const float *p) { return _mm_loadu_ps(p); }

    inline void store(float *p) const  { _mm_store_ps(p,x); }
    inline void storeu(float *p) const { _mm_storeu_ps(p,x); }

    inline simd_t<float,4> sqrt() const { return _mm_sqrt_ps(x); }
    inline simd_t<float,4> rsqrt() const { return _mm_rsqrt_ps(x); }

    // Comparison operators.
    // Note: the output of a comparison is -1 (0xff..) for "true" or 0 for "false".
    // Note: these are quiet ordered comparisons (e.g. NaN==NaN evaluates to "false")

    inline simd_t<int,4> compare_eq(simd_t<float,4> t) const  { return _mm_castps_si128(_mm_cmpeq_ps(x, t.x)); }
    inline simd_t<int,4> compare_ne(simd_t<float,4> t) const  { return _mm_castps_si128(_mm_cmpneq_ps(x, t.x)); }
    inline simd_t<int,4> compare_ge(simd_t<float,4> t) const  { return _mm_castps_si128(_mm_cmpge_ps(x, t.x)); }
    inline simd_t<int,4> compare_gt(simd_t<float,4> t) const  { return _mm_castps_si128(_mm_cmpgt_ps(x, t.x)); }
    inline simd_t<int,4> compare_le(simd_t<float,4> t) const  { return _mm_castps_si128(_mm_cmple_ps(x, t.x)); }
    inline simd_t<int,4> compare_lt(simd_t<float,4> t) const  { return _mm_castps_si128(_mm_cmplt_ps(x, t.x)); }

    inline simd_t<float,4> min(simd_t<float,4> t) const { return _mm_min_ps(x, t.x); }
    inline simd_t<float,4> max(simd_t<float,4> t) const { return _mm_max_ps(x, t.x); }

    // Only makes sense if each 32-bit word in 't' is either 0x0 or 0xff..
    inline simd_t<float,4> bitwise_and(simd_t<int,4> t) const { return _mm_and_ps(x, _mm_castsi128_ps(t.x)); }

    template<unsigned int M> 
    inline float extract() const
    {
	union { int i; float x; } u;
	u.i = _mm_extract_ps(x, M);
	return u.x;
    }

    // Returns the horizontal sum of elements as a constant simd vector [ x ... x ].
    inline simd_t<float,4> horizontal_sum() const
    {
	__m128 y = x + _mm_shuffle_ps(x, x, 0xb1);   // (2301)_4 = 0xb1
	return y + _mm_shuffle_ps(y, y, 0x4e);       // (1032)_4 = 0x4e
    }

    // Returns the horizontal sum of elements as a scalar.
    inline float sum() const
    {
	simd_t<float,4> y = this->horizontal_sum();
	return y.extract<0> ();
    }
};


// -------------------------------------------------------------------------------------------------


template<> struct simd_t<float,8>
{
    __m256 x;

    simd_t() { }
    simd_t(__m256 y) { x = y; }
    simd_t(float y)  { x = _mm256_set1_ps(y); }

    static simd_t<float,8> zero()  { return _mm256_setzero_ps(); }
    static simd_t<float,8> range() { return _mm256_set_ps(7., 6., 5., 4., 3., 2., 1., 0.); }

    static inline simd_t<float,8> load(const float *p)  { return _mm256_load_ps(p); }
    static inline simd_t<float,8> loadu(const float *p) { return _mm256_loadu_ps(p); }

    inline void store(float *p) const  { _mm256_store_ps(p,x); }
    inline void storeu(float *p) const { _mm256_storeu_ps(p,x); }

    inline simd_t<float,8> sqrt() const { return _mm256_sqrt_ps(x); }
    inline simd_t<float,8> rsqrt() const { return _mm256_rsqrt_ps(x); }

    // Comparison operators.
    // Note: the output of a comparison is -1 (0xff..) for "true" or 0 for "false".
    // Note: these are quiet ordered comparisons (e.g. NaN==NaN evaluates to "false")

    inline simd_t<int,8> compare_eq(simd_t<float,8> t) const  { return _mm256_castps_si256(_mm256_cmp_ps(x, t.x, _CMP_EQ_OQ)); }
    inline simd_t<int,8> compare_ne(simd_t<float,8> t) const  { return _mm256_castps_si256(_mm256_cmp_ps(x, t.x, _CMP_NEQ_OQ)); }
    inline simd_t<int,8> compare_ge(simd_t<float,8> t) const  { return _mm256_castps_si256(_mm256_cmp_ps(x, t.x, _CMP_GE_OQ)); }
    inline simd_t<int,8> compare_gt(simd_t<float,8> t) const  { return _mm256_castps_si256(_mm256_cmp_ps(x, t.x, _CMP_GT_OQ)); }
    inline simd_t<int,8> compare_le(simd_t<float,8> t) const  { return _mm256_castps_si256(_mm256_cmp_ps(x, t.x, _CMP_LE_OQ)); }
    inline simd_t<int,8> compare_lt(simd_t<float,8> t) const  { return _mm256_castps_si256(_mm256_cmp_ps(x, t.x, _CMP_LT_OQ)); }

    inline simd_t<float,8> min(simd_t<float,8> t) const { return _mm256_min_ps(x, t.x); }
    inline simd_t<float,8> max(simd_t<float,8> t) const { return _mm256_max_ps(x, t.x); }

    // Only makes sense if each 32-bit word in 't' is either 0x0 or 0xff..
    inline simd_t<float,8> bitwise_and(simd_t<int,8> t) const { return _mm256_and_ps(x, _mm256_castsi256_ps(t.x)); }

    template<unsigned int M> 
    inline float extract() const
    {
	simd_t<float,4> x2 = _mm256_extractf128_ps(x, M/4);
	return x2.extract<M%4> ();
    }

    // Returns the horizontal sum of elements as a constant simd vector [ x ... x ].    
    inline simd_t<float,8> horizontal_sum() const
    {
	__m256 y = x + _mm256_shuffle_ps(x, x, 0xb1);   // (2301)_4 = 0xb1
	y += _mm256_shuffle_ps(y, y, 0x4e);             // (1032)_4 = 0x4e
	return y + _mm256_permute2f128_ps(y, y, 0x01);
    }

    // Returns the horizontal sum of elements as a scalar.
    inline float sum() const
    {
	__m256 y = x + _mm256_permute2f128_ps(x, x, 0x01);

	simd_t<float,4> z = _mm256_extractf128_ps(y, 0);
	return z.sum();
    }
};


template<typename T, unsigned int S> inline simd_t<T,S> &operator+=(simd_t<T,S> &a, simd_t<T,S> b) { a.x += b.x; return a; }
template<typename T, unsigned int S> inline simd_t<T,S> &operator-=(simd_t<T,S> &a, simd_t<T,S> b) { a.x -= b.x; return a; }
template<typename T, unsigned int S> inline simd_t<T,S> &operator*=(simd_t<T,S> &a, simd_t<T,S> b) { a.x *= b.x; return a; }
template<typename T, unsigned int S> inline simd_t<T,S> &operator/=(simd_t<T,S> &a, simd_t<T,S> b) { a.x /= b.x; return a; }

template<typename T, unsigned int S> inline simd_t<T,S> operator+(simd_t<T,S> a, simd_t<T,S> b) { return a.x + b.x; }
template<typename T, unsigned int S> inline simd_t<T,S> operator-(simd_t<T,S> a, simd_t<T,S> b) { return a.x - b.x; }
template<typename T, unsigned int S> inline simd_t<T,S> operator*(simd_t<T,S> a, simd_t<T,S> b) { return a.x * b.x; }
template<typename T, unsigned int S> inline simd_t<T,S> operator/(simd_t<T,S> a, simd_t<T,S> b) { return a.x / b.x; }

template<typename T, unsigned int S> inline simd_t<T,S> operator*(T a, simd_t<T,S> b) { return simd_t<T,S>(a) * b; }
template<typename T, unsigned int S> inline simd_t<T,S> operator*(simd_t<T,S> a, T b) { return a * simd_t<T,S>(b); }


// blendv(mask,a,b) is morally equivalent to (mask ? a : b)
inline simd_t<float,4> blendv(simd_t<int,4> mask, simd_t<float,4> a, simd_t<float,4> b)  { return _mm_blendv_ps(b.x, a.x, _mm_castsi128_ps(mask.x)); }
inline simd_t<float,8> blendv(simd_t<int,8> mask, simd_t<float,8> a, simd_t<float,8> b)  { return _mm256_blendv_ps(b.x, a.x, _mm256_castsi256_ps(mask.x)); }


}  // namespace simd_helpers

#endif // _SIMD_HELPERS_SIMD_T_HPP
