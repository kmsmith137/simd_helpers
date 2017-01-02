#ifndef _SIMD_HELPERS_INT32_HPP
#define _SIMD_HELPERS_INT32_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

#include "base.hpp"

namespace simd_helpers {
#if 0
}  // pacify emacs c-mode
#endif


// -------------------------------------------------------------------------------------------------
//
// simd_t<int,4>
//
// Note: for a "declaration" of the key class simd_t<T,S>,
// see the extended comment in simd_helpers/base.hpp


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

    inline simd_t<int,4> &operator+=(simd_t<int,4> t) { x = _mm_add_epi32(x,t.x); return *this; }
    inline simd_t<int,4> &operator-=(simd_t<int,4> t) { x = _mm_sub_epi32(x,t.x); return *this; }

    inline simd_t<int,4> operator+(simd_t<int,4> t) const { return _mm_add_epi32(x,t.x); }
    inline simd_t<int,4> operator-(simd_t<int,4> t) const { return _mm_sub_epi32(x,t.x); }

    inline simd_t<int,4> abs() const { return _mm_abs_epi32(x); }

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
    inline int is_all_zeros() const                            { return _mm_testz_si128(x, x); }
    inline int is_all_ones() const                             { return _mm_test_all_ones(x); }

    inline simd_t<int,4> horizontal_sum() const
    {
	__m128i y = _mm_add_epi32(x, _mm_shuffle_epi32(x, 0xb1));  // (2301)_4 = 0xb1
	return _mm_add_epi32(y, _mm_shuffle_epi32(y, 0x4e));       // (1032)_4 = 0x4e
    }

    inline int sum() const { return _mm_extract_epi32(horizontal_sum().x, 0); }

    // Note: you might need to call this with the weird-looking syntax
    //    x.template extract<M> ();
    template<unsigned int M> inline int extract() const  { return _mm_extract_epi32(x, M); }
};


// -------------------------------------------------------------------------------------------------
//
// simd_t<int,8>


template<> struct simd_t<int,8>
{
    __m256i x;

    simd_t() { }
    simd_t(__m256i y)                         { x = y; }
    simd_t(int y)                             { x = _mm256_set1_epi32(y); }
    simd_t(simd_t<int,4> y, simd_t<int,4> z)  { x = _mm256_insertf128_si256(_mm256_castsi128_si256(y.x), (z.x), 1); }

    static inline simd_t<int,8> zero()  { return _mm256_setzero_si256(); }
    static inline simd_t<int,8> range() { return _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0); }

    static inline simd_t<int,8> load(const int *p)  { return _mm256_load_si256((const __m256i *) p); }
    static inline simd_t<int,8> loadu(const int *p) { return _mm256_loadu_si256((const __m256i *) p); }

    inline void store(int *p) const  { _mm256_store_si256((__m256i *)p, x); }
    inline void storeu(int *p) const { _mm256_storeu_si256((__m256i *)p, x); }

    inline simd_t<int,8> &operator+=(simd_t<int,8> t) { x = _mm256_add_epi32(x,t.x); return *this; }
    inline simd_t<int,8> &operator-=(simd_t<int,8> t) { x = _mm256_sub_epi32(x,t.x); return *this; }

    inline simd_t<int,8> operator+(simd_t<int,8> t) const { return _mm256_add_epi32(x,t.x); }
    inline simd_t<int,8> operator-(simd_t<int,8> t) const { return _mm256_sub_epi32(x,t.x); }

    inline simd_t<int,8> abs() const
    {
#ifdef __AVX2__
	return _mm256_abs_epi32(x); 
#else
	// Alternate implementation which might work: cast and use _mm256_and_ps()?
	simd_t<int,4> ret0 = _mm_abs_epi32(_mm256_extractf128_si256(x,0));
	simd_t<int,4> ret1 = _mm_abs_epi32(_mm256_extractf128_si256(x,1));
	return simd_t<int,8> (ret0, ret1);
#endif
    }

    // The comparison operators return all ones (0xff..) if "true".
    inline simd_t<int,8> compare_eq(simd_t<int,8> t) const
    {
#ifdef __AVX2__
	return _mm256_cmpeq_epi32(x, t.x); 
#else
	simd_t<int,4> ret0 = _mm_cmpeq_epi32(_mm256_extractf128_si256(x,0), _mm256_extractf128_si256(t.x,0));
	simd_t<int,4> ret1 = _mm_cmpeq_epi32(_mm256_extractf128_si256(x,1), _mm256_extractf128_si256(t.x,1));
	return simd_t<int,8> (ret0, ret1);
#endif
    }

    inline simd_t<int,8> compare_gt(simd_t<int,8> t) const
    {
#ifdef __AVX2__
	return _mm256_cmpgt_epi32(x, t.x); 
#else
	simd_t<int,4> ret0 = _mm_cmpgt_epi32(_mm256_extractf128_si256(x,0), _mm256_extractf128_si256(t.x,0));
	simd_t<int,4> ret1 = _mm_cmpgt_epi32(_mm256_extractf128_si256(x,1), _mm256_extractf128_si256(t.x,1));
	return simd_t<int,8> (ret0, ret1);
#endif
    }

    inline simd_t<int,8> min(simd_t<int,8> t) const
    {
#ifdef __AVX2__
	return _mm256_min_epi32(x, t.x); 
#else
	simd_t<int,4> ret0 = _mm_min_epi32(_mm256_extractf128_si256(x,0), _mm256_extractf128_si256(t.x,0));
	simd_t<int,4> ret1 = _mm_min_epi32(_mm256_extractf128_si256(x,1), _mm256_extractf128_si256(t.x,1));
	return simd_t<int,8> (ret0, ret1);
#endif
    }

    inline simd_t<int,8> max(simd_t<int,8> t) const
    {
#ifdef __AVX2__
	return _mm256_max_epi32(x, t.x); 
#else
	simd_t<int,4> ret0 = _mm_max_epi32(_mm256_extractf128_si256(x,0), _mm256_extractf128_si256(t.x,0));
	simd_t<int,4> ret1 = _mm_max_epi32(_mm256_extractf128_si256(x,1), _mm256_extractf128_si256(t.x,1));
	return simd_t<int,8> (ret0, ret1);
#endif
    }

    inline simd_t<int,8> bitwise_and(simd_t<int,8> t) const
    {
#ifdef __AVX2__
	return _mm256_and_si256(x, t.x); 
#else
	simd_t<int,4> ret0 = _mm_and_si128(_mm256_extractf128_si256(x,0), _mm256_extractf128_si256(t.x,0));
	simd_t<int,4> ret1 = _mm_and_si128(_mm256_extractf128_si256(x,1), _mm256_extractf128_si256(t.x,1));
	return simd_t<int,8> (ret0, ret1);
#endif
    }

    inline simd_t<int,8> bitwise_or(simd_t<int,8> t) const
    {
#ifdef __AVX2__
	return _mm256_or_si256(x, t.x); 
#else
	simd_t<int,4> ret0 = _mm_or_si128(_mm256_extractf128_si256(x,0), _mm256_extractf128_si256(t.x,0));
	simd_t<int,4> ret1 = _mm_or_si128(_mm256_extractf128_si256(x,1), _mm256_extractf128_si256(t.x,1));
	return simd_t<int,8> (ret0, ret1);
#endif
    }

    inline simd_t<int,8> bitwise_xor(simd_t<int,8> t) const
    {
#ifdef __AVX2__
	return _mm256_xor_si256(x, t.x); 
#else
	simd_t<int,4> ret0 = _mm_xor_si128(_mm256_extractf128_si256(x,0), _mm256_extractf128_si256(t.x,0));
	simd_t<int,4> ret1 = _mm_xor_si128(_mm256_extractf128_si256(x,1), _mm256_extractf128_si256(t.x,1));
	return simd_t<int,8> (ret0, ret1);
#endif
    }

    inline simd_t<int,8> bitwise_andnot(simd_t<int,8> t) const
    {
#ifdef __AVX2__
	return _mm256_andnot_si256(x, t.x); 
#else
	simd_t<int,4> ret0 = _mm_andnot_si128(_mm256_extractf128_si256(x,0), _mm256_extractf128_si256(t.x,0));
	simd_t<int,4> ret1 = _mm_andnot_si128(_mm256_extractf128_si256(x,1), _mm256_extractf128_si256(t.x,1));
	return simd_t<int,8> (ret0, ret1);
#endif
    }

    inline int testzero_bitwise_and(simd_t<int,8> t) const     { return _mm256_testz_si256(x, t.x); }
    inline int testzero_bitwise_andnot(simd_t<int,8> t) const  { return _mm256_testc_si256(t.x, x); }
    inline int is_all_zeros() const                            { return _mm256_testz_si256(x, x); }
    inline int is_all_ones() const                             { return _mm256_testc_si256(x, _mm256_set1_epi32(-1)); }

    template<unsigned int M> inline int extract() const                { return _mm256_extract_epi32(x,M); }
    template<unsigned int M> inline simd_t<int,4> extract128() const   { return _mm256_extractf128_si256(x,M); }

    inline simd_t<int,8> horizontal_sum() const
    {
#ifdef __AVX2__
	__m256i y = _mm256_add_epi32(x, _mm256_shuffle_epi32(x, 0xb1));  // (2301)_4 = 0xb1
	y = _mm256_add_epi32(y, _mm256_shuffle_epi32(y, 0x4e));          // (1032)_4 = 0x4e
	return _mm256_add_epi32(y, _mm256_permute2f128_si256(y, y, 0x01));
#else
	simd_t<int,4> x0 = _mm256_extractf128_si256(x,0) + _mm256_extractf128_si256(x,1);
	x0 = x0.horizontal_sum();
	return simd_t<int,8> (x0, x0);
#endif
    }

    inline int sum() const { return _mm256_extract_epi32(horizontal_sum().x, 0); }
};


}  // namespace simd_helpers


#endif // _SIMD_HELPERS_INT32_HPP
