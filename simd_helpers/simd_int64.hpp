#ifndef _SIMD_HELPERS_INT64_HPP
#define _SIMD_HELPERS_INT64_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

#include "core.hpp"

namespace simd_helpers {
#if 0
}  // pacify emacs c-mode
#endif


// -------------------------------------------------------------------------------------------------
//
// simd_t<int64_t,2>
//
// Note: for a "declaration" of the key class simd_t<T,S>,
// see the extended comment in simd_helpers/core.hpp


template<> struct simd_t<int64_t,2>
{
    __m128i x;

    simd_t() { }
    simd_t(__m128i y) { x = y; }
    simd_t(int64_t y) { x = _mm_set1_epi64x(y); }

    static inline simd_t<int64_t,2> zero()  { return _mm_setzero_si128(); }
    static inline simd_t<int64_t,2> range() { return _mm_set_epi64x(1, 0); }

    static inline simd_t<int64_t,2> load(const int64_t *p)  { return _mm_load_si128((const __m128i *) p); }
    static inline simd_t<int64_t,2> loadu(const int64_t *p) { return _mm_loadu_si128((const __m128i *) p); }

    inline void store(int64_t *p) const  { _mm_store_si128((__m128i *)p, x); }
    inline void storeu(int64_t *p) const { _mm_storeu_si128((__m128i *)p, x); }

    template<unsigned int M> inline int64_t extract() const  { return _mm_extract_epi64(x, M); }

    inline simd_t<int64_t,2> &operator+=(simd_t<int64_t,2> t) { x = _mm_add_epi64(x,t.x); return *this; }
    inline simd_t<int64_t,2> &operator-=(simd_t<int64_t,2> t) { x = _mm_sub_epi64(x,t.x); return *this; }

    inline simd_t<int64_t,2> operator+(simd_t<int64_t,2> t) const { return _mm_add_epi64(x,t.x); }
    inline simd_t<int64_t,2> operator-(simd_t<int64_t,2> t) const { return _mm_sub_epi64(x,t.x); }

    inline simd_t<int64_t,2> operator*(simd_t<int64_t,2> t) const
    {
	__m128i y = _mm_mul_epu32(x, t.x);
	__m128i z = _mm_mullo_epi32(x, _mm_shuffle_epi32(t.x, 0xb1));  // 0xb1 = (2301)_4
	z = _mm_add_epi64(z, _mm_shuffle_epi32(z,0xb1));
	z = _mm_blend_epi32(z, _mm_setzero_si128(), 0x5);   // 0x5 = (0101)_2	
	return _mm_add_epi64(y, z);
    }

    inline simd_t<int64_t,2> operator-() const
    {
	__m128i t = _mm_set1_epi16(-1);
	return _mm_xor_si128(_mm_add_epi64(x,t), t);
    }

    inline simd_t<int64_t,2> abs() const
    {
	__m128i t = _mm_set1_epi16(-1);
	__m128i nx = _mm_xor_si128(_mm_add_epi64(x,t), t);
	__m128i pos = _mm_cmpgt_epi64(x, _mm_setzero_si128());
	return _mm_blendv_epi8(nx, x, pos);
    }

    inline simd_t<int64_t,2> min(simd_t<int64_t,2> t) const  { return _mm_blendv_epi8(x, t.x, _mm_cmpgt_epi64(x,t.x)); }
    inline simd_t<int64_t,2> max(simd_t<int64_t,2> t) const  { return _mm_blendv_epi8(t.x, x, _mm_cmpgt_epi64(x,t.x)); }

    inline simd_t<int64_t,2> compare_eq(simd_t<int64_t,2> t) const  { return _mm_cmpeq_epi64(x, t.x); }
    inline simd_t<int64_t,2> compare_gt(simd_t<int64_t,2> t) const  { return _mm_cmpgt_epi64(x, t.x); }
    inline simd_t<int64_t,2> compare_lt(simd_t<int64_t,2> t) const  { return _mm_cmpgt_epi64(t.x, x); }
    inline simd_t<int64_t,2> compare_ne(simd_t<int64_t,2> t) const  { return compare_eq(t).bitwise_not(); }
    inline simd_t<int64_t,2> compare_ge(simd_t<int64_t,2> t) const  { return compare_lt(t).bitwise_not(); }
    inline simd_t<int64_t,2> compare_le(simd_t<int64_t,2> t) const  { return compare_gt(t).bitwise_not(); }

    inline simd_t<int64_t,2> apply_mask(simd_t<int64_t,2> t) const          { return bitwise_and(t); }
    inline simd_t<int64_t,2> apply_inverse_mask(simd_t<int64_t,2> t) const  { return bitwise_andnot(t); }

    inline simd_t<int64_t,2> bitwise_and(simd_t<int64_t,2> t) const     { return _mm_and_si128(x, t.x); }
    inline simd_t<int64_t,2> bitwise_or(simd_t<int64_t,2> t) const      { return _mm_or_si128(x, t.x); }
    inline simd_t<int64_t,2> bitwise_xor(simd_t<int64_t,2> t) const     { return _mm_xor_si128(x, t.x); }
    inline simd_t<int64_t,2> bitwise_andnot(simd_t<int64_t,2> t) const  { return _mm_andnot_si128(t.x, x); }
    inline simd_t<int64_t,2> bitwise_not() const                        { return _mm_xor_si128(x, _mm_set1_epi16(-1)); }

    inline int is_all_ones() const                                        { return _mm_test_all_ones(x); }
    inline int is_all_zeros() const                                       { return _mm_testz_si128(x, x); }
    inline int is_all_zeros_masked(simd_t<int64_t,2> mask) const          { return _mm_testz_si128(x, mask.x); }
    inline int is_all_zeros_inverse_masked(simd_t<int64_t,2> mask) const  { return _mm_testc_si128(mask.x, x); }

    inline simd_t<int64_t,2> horizontal_sum() const  { return _mm_add_epi64(x, _mm_shuffle_epi32(x, 0x4e)); }
    inline int64_t sum() const                       { return _mm_extract_epi64(horizontal_sum().x, 0); }
};


// -------------------------------------------------------------------------------------------------
//
// simd_t<int64_t,4>


template<> struct simd_t<int64_t,4>
{
    __m256i x;

    simd_t() { }
    simd_t(__m256i y)                                 { x = y; }
    simd_t(int64_t y)                                 { x = _mm256_set1_epi64x(y); }
    simd_t(simd_t<int64_t,2> y, simd_t<int64_t,2> z)  { x = _mm256_insertf128_si256(_mm256_castsi128_si256(y.x), (z.x), 1); }

    static inline simd_t<int64_t,4> zero()  { return _mm256_setzero_si256(); }
    static inline simd_t<int64_t,4> range() { return _mm256_set_epi64x(3, 2, 1, 0); }

    static inline simd_t<int64_t,4> load(const int64_t *p)  { return _mm256_load_si256((const __m256i *) p); }
    static inline simd_t<int64_t,4> loadu(const int64_t *p) { return _mm256_loadu_si256((const __m256i *) p); }

    inline void store(int64_t *p) const  { _mm256_store_si256((__m256i *)p, x); }
    inline void storeu(int64_t *p) const { _mm256_storeu_si256((__m256i *)p, x); }

    template<unsigned int M> inline int extract() const                     { return _mm256_extract_epi64(x,M); }
    template<unsigned int M> inline simd_t<int64_t,2> extract_half() const  { return _mm256_extractf128_si256(x,M); }
    
    void split(simd_t<int64_t,2> &x0, simd_t<int64_t,2> &x1) const  { x0 = extract_half<0> (); x1 = extract_half<1> (); }

    inline simd_t<int64_t,4> &operator+=(simd_t<int64_t,4> t) 
    { 
#ifdef __AVX2__
	x = _mm256_add_epi64(x,t.x); 
#else
	simd_t<int,4> ret0 = _mm_add_epi64(_mm256_extractf128_si256(x,0), _mm256_extractf128_si256(t.x,0));
	simd_t<int,4> ret1 = _mm_add_epi64(_mm256_extractf128_si256(x,1), _mm256_extractf128_si256(t.x,1));
	x = _mm256_insertf128_si256(_mm256_castsi128_si256(ret0.x), (ret1.x), 1);

#endif
	return *this; 
    }

    inline simd_t<int64_t,4> &operator-=(simd_t<int64_t,4> t) 
    { 
#ifdef __AVX2__
	x = _mm256_sub_epi64(x,t.x); 
#else
	simd_t<int,4> ret0 = _mm_sub_epi64(_mm256_extractf128_si256(x,0), _mm256_extractf128_si256(t.x,0));
	simd_t<int,4> ret1 = _mm_sub_epi64(_mm256_extractf128_si256(x,1), _mm256_extractf128_si256(t.x,1));
	x = _mm256_insertf128_si256(_mm256_castsi128_si256(ret0.x), (ret1.x), 1);

#endif
	return *this; 
    }

    inline simd_t<int64_t,4> operator+(simd_t<int64_t,4> t) const 
    { 
#ifdef __AVX2__
	return _mm256_add_epi64(x,t.x); 
#else
	simd_t<int,4> ret0 = _mm_add_epi64(_mm256_extractf128_si256(x,0), _mm256_extractf128_si256(t.x,0));
	simd_t<int,4> ret1 = _mm_add_epi64(_mm256_extractf128_si256(x,1), _mm256_extractf128_si256(t.x,1));
	return _mm256_insertf128_si256(_mm256_castsi128_si256(ret0.x), (ret1.x), 1);
#endif
    }

    inline simd_t<int64_t,4> operator-(simd_t<int64_t,4> t) const 
    { 
#ifdef __AVX2__
	return _mm256_sub_epi64(x,t.x); 
#else
	simd_t<int,4> ret0 = _mm_sub_epi64(_mm256_extractf128_si256(x,0), _mm256_extractf128_si256(t.x,0));
	simd_t<int,4> ret1 = _mm_sub_epi64(_mm256_extractf128_si256(x,1), _mm256_extractf128_si256(t.x,1));
	return _mm256_insertf128_si256(_mm256_castsi128_si256(ret0.x), (ret1.x), 1);
#endif
    }

    inline simd_t<int64_t,4> operator*(simd_t<int64_t,4> t) const
    {
#ifdef __AVX2__
	__m256i y = _mm256_mul_epu32(x, t.x);
	__m256i z = _mm256_mullo_epi32(x, _mm256_shuffle_epi32(t.x, 0xb1));  // 0xb1 = (2301)_4
	z = _mm256_add_epi64(z, _mm256_shuffle_epi32(z,0xb1));
	z = _mm256_blend_epi32(z, _mm256_setzero_si256(), 0x55);   // 0x55 = (01010101)_2	
	return _mm256_add_epi64(y, z);
#else
	simd_t<int64_t,2> x0, x1, t0, t1;
	split(x0, x1); 
	t.split(t0, t1);
	return simd_t<int64_t,4> (x0*t0, x1*t1);
#endif
    }

    inline simd_t<int64_t,4> operator-() const
    {
#ifdef __AVX2__
	__m256i t = _mm256_set1_epi16(-1);
	return _mm256_xor_si256(_mm256_add_epi64(x,t), t);
#else
	__m128i t = _mm_set1_epi16(-1);
	simd_t<int64_t,2> ret0 = _mm_xor_si128(_mm_add_epi64(_mm256_extractf128_si256(x,0),t), t);
	simd_t<int64_t,2> ret1 = _mm_xor_si128(_mm_add_epi64(_mm256_extractf128_si256(x,1),t), t);
	return simd_t<int64_t,4> (ret0, ret1);
#endif
    }

    inline simd_t<int64_t,4> abs() const
    {
#ifdef __AVX2__
	__m256i t = _mm256_set1_epi16(-1);
	__m256i nx = _mm256_xor_si256(_mm256_add_epi64(x,t), t);
	__m256i pos = _mm256_cmpgt_epi64(x, _mm256_setzero_si256());
	return _mm256_blendv_epi8(nx, x, pos);
#else
	__m128i z = _mm_setzero_si128();
	__m128i t = _mm_set1_epi16(-1);
	__m128i x0 = _mm256_extractf128_si256(x,0);
	__m128i x1 = _mm256_extractf128_si256(x,1);
	__m128i nx0 = _mm_xor_si128(_mm_add_epi64(x0,t), t);
	__m128i nx1 = _mm_xor_si128(_mm_add_epi64(x1,t), t);
	__m128i pos0 = _mm_cmpgt_epi64(x0, z);
	__m128i pos1 = _mm_cmpgt_epi64(x1, z);
	__m128i ret0 = _mm_blendv_epi8(nx0, x0, pos0);
	__m128i ret1 = _mm_blendv_epi8(nx1, x1, pos1);
	return simd_t<int64_t,4> (ret0, ret1);
#endif
    }

    inline simd_t<int64_t,4> min(simd_t<int64_t,4> t) const
    {
#ifdef __AVX2__
	return _mm256_blendv_epi8(x, t.x, _mm256_cmpgt_epi64(x,t.x));
#else
	__m128i x0 = _mm256_extractf128_si256(x,0);
	__m128i x1 = _mm256_extractf128_si256(x,1);
	__m128i t0 = _mm256_extractf128_si256(t.x,0);
	__m128i t1 = _mm256_extractf128_si256(t.x,1);
	__m128i ret0 = _mm_blendv_epi8(x0, t0, _mm_cmpgt_epi64(x0,t0));
	__m128i ret1 = _mm_blendv_epi8(x1, t1, _mm_cmpgt_epi64(x1,t1));
	return simd_t<int64_t,4> (ret0, ret1);
#endif
    }

    inline simd_t<int64_t,4> max(simd_t<int64_t,4> t) const
    {
#ifdef __AVX2__
	return _mm256_blendv_epi8(t.x, x, _mm256_cmpgt_epi64(x,t.x));
#else
	__m128i x0 = _mm256_extractf128_si256(x,0);
	__m128i x1 = _mm256_extractf128_si256(x,1);
	__m128i t0 = _mm256_extractf128_si256(t.x,0);
	__m128i t1 = _mm256_extractf128_si256(t.x,1);
	__m128i ret0 = _mm_blendv_epi8(t0, x0, _mm_cmpgt_epi64(x0,t0));
	__m128i ret1 = _mm_blendv_epi8(t1, x1, _mm_cmpgt_epi64(x1,t1));
	return simd_t<int64_t,4> (ret0, ret1);
#endif
    }

    inline simd_t<int64_t,4> compare_eq(simd_t<int64_t,4> t) const 
    { 
#ifdef __AVX2__
	return _mm256_cmpeq_epi64(x, t.x); 
#else
	simd_t<int64_t,2> ret0 = _mm_cmpeq_epi64(_mm256_extractf128_si256(x,0), _mm256_extractf128_si256(t.x,0));
	simd_t<int64_t,2> ret1 = _mm_cmpeq_epi64(_mm256_extractf128_si256(x,1), _mm256_extractf128_si256(t.x,1));
	return simd_t<int64_t,4> (ret0, ret1);
#endif
    }

    inline simd_t<int64_t,4> compare_gt(simd_t<int64_t,4> t) const  
    { 
#ifdef __AVX2__
	return _mm256_cmpgt_epi64(x, t.x); 
#else
	simd_t<int64_t,2> ret0 = _mm_cmpgt_epi64(_mm256_extractf128_si256(x,0), _mm256_extractf128_si256(t.x,0));
	simd_t<int64_t,2> ret1 = _mm_cmpgt_epi64(_mm256_extractf128_si256(x,1), _mm256_extractf128_si256(t.x,1));
	return simd_t<int64_t,4> (ret0, ret1);	
#endif
    }

    inline simd_t<int64_t,4> compare_lt(simd_t<int64_t,4> t) const  
    { 
#ifdef __AVX2__
	return _mm256_cmpgt_epi64(t.x, x); 
#else
	simd_t<int64_t,2> ret0 = _mm_cmpgt_epi64(_mm256_extractf128_si256(t.x,0), _mm256_extractf128_si256(x,0));
	simd_t<int64_t,2> ret1 = _mm_cmpgt_epi64(_mm256_extractf128_si256(t.x,1), _mm256_extractf128_si256(x,1));
	return simd_t<int64_t,4> (ret0, ret1);
#endif
    }

    inline simd_t<int64_t,4> compare_ne(simd_t<int64_t,4> t) const  
    { 
#ifdef __AVX2__
	return compare_eq(t).bitwise_not(); 
#else
	simd_t<int64_t,2> ret0 = _mm_cmpeq_epi64(_mm256_extractf128_si256(x,0), _mm256_extractf128_si256(t.x,0));
	simd_t<int64_t,2> ret1 = _mm_cmpeq_epi64(_mm256_extractf128_si256(x,1), _mm256_extractf128_si256(t.x,1));
	return simd_t<int64_t,4> (ret0.bitwise_not(), ret1.bitwise_not());
#endif
    }

    inline simd_t<int64_t,4> compare_ge(simd_t<int64_t,4> t) const  
    { 
#ifdef __AVX2__
	return compare_lt(t).bitwise_not(); 
#else
	simd_t<int64_t,2> ret0 = _mm_cmpgt_epi64(_mm256_extractf128_si256(t.x,0), _mm256_extractf128_si256(x,0));
	simd_t<int64_t,2> ret1 = _mm_cmpgt_epi64(_mm256_extractf128_si256(t.x,1), _mm256_extractf128_si256(x,1));
	return simd_t<int64_t,4> (ret0.bitwise_not(), ret1.bitwise_not());
#endif
    }

    inline simd_t<int64_t,4> compare_le(simd_t<int64_t,4> t) const  
    { 
#ifdef __AVX2__
	return compare_gt(t).bitwise_not(); 
#else
	simd_t<int64_t,2> ret0 = _mm_cmpgt_epi64(_mm256_extractf128_si256(x,0), _mm256_extractf128_si256(t.x,0));
	simd_t<int64_t,2> ret1 = _mm_cmpgt_epi64(_mm256_extractf128_si256(x,1), _mm256_extractf128_si256(t.x,1));
	return simd_t<int64_t,4> (ret0.bitwise_not(), ret1.bitwise_not());
#endif
    }
    
    inline simd_t<int64_t,4> apply_mask(simd_t<int64_t,4> t) const          { return bitwise_and(t); }
    inline simd_t<int64_t,4> apply_inverse_mask(simd_t<int64_t,4> t) const  { return bitwise_andnot(t); }

    inline simd_t<int64_t,4> bitwise_and(simd_t<int64_t,4> t) const
    {
#ifdef __AVX2__
	return _mm256_and_si256(x, t.x); 
#else
	simd_t<int64_t,2> ret0 = _mm_and_si128(_mm256_extractf128_si256(x,0), _mm256_extractf128_si256(t.x,0));
	simd_t<int64_t,2> ret1 = _mm_and_si128(_mm256_extractf128_si256(x,1), _mm256_extractf128_si256(t.x,1));
	return simd_t<int64_t,4> (ret0, ret1);
#endif
    }

    inline simd_t<int64_t,4> bitwise_or(simd_t<int64_t,4> t) const
    {
#ifdef __AVX2__
	return _mm256_or_si256(x, t.x); 
#else
	simd_t<int64_t,2> ret0 = _mm_or_si128(_mm256_extractf128_si256(x,0), _mm256_extractf128_si256(t.x,0));
	simd_t<int64_t,2> ret1 = _mm_or_si128(_mm256_extractf128_si256(x,1), _mm256_extractf128_si256(t.x,1));
	return simd_t<int64_t,4> (ret0, ret1);
#endif
    }

    inline simd_t<int64_t,4> bitwise_xor(simd_t<int64_t,4> t) const
    {
#ifdef __AVX2__
	return _mm256_xor_si256(x, t.x); 
#else
	simd_t<int64_t,2> ret0 = _mm_xor_si128(_mm256_extractf128_si256(x,0), _mm256_extractf128_si256(t.x,0));
	simd_t<int64_t,2> ret1 = _mm_xor_si128(_mm256_extractf128_si256(x,1), _mm256_extractf128_si256(t.x,1));
	return simd_t<int64_t,4> (ret0, ret1);
#endif
    }

    inline simd_t<int64_t,4> bitwise_andnot(simd_t<int64_t,4> t) const
    {
#ifdef __AVX2__
	return _mm256_andnot_si256(t.x, x); 
#else
	simd_t<int64_t,2> ret0 = _mm_andnot_si128(_mm256_extractf128_si256(t.x,0), _mm256_extractf128_si256(x,0));
	simd_t<int64_t,2> ret1 = _mm_andnot_si128(_mm256_extractf128_si256(t.x,1), _mm256_extractf128_si256(x,1));
	return simd_t<int64_t,4> (ret0, ret1);
#endif
    }

    inline simd_t<int64_t,4> bitwise_not() const
    {
#ifdef __AVX2__
	return _mm256_xor_si256(x, _mm256_set1_epi16(-1));
#else
	simd_t<int64_t,2> ret0 = _mm256_extractf128_si256(x,0);
	simd_t<int64_t,2> ret1 = _mm256_extractf128_si256(x,1);
	return simd_t<int64_t,4> (ret0.bitwise_not(), ret1.bitwise_not());
#endif
    }

    inline int is_all_ones() const                                        { return _mm256_testc_si256(x, _mm256_set1_epi32(-1)); }
    inline int is_all_zeros() const                                       { return _mm256_testz_si256(x, x); }
    inline int is_all_zeros_masked(simd_t<int64_t,4> mask) const          { return _mm256_testz_si256(x, mask.x); }
    inline int is_all_zeros_inverse_masked(simd_t<int64_t,4> mask) const  { return _mm256_testc_si256(mask.x, x); }

    inline simd_t<int64_t,4> horizontal_sum() const
    { 
#ifdef __AVX2__
	__m256i y = _mm256_add_epi64(x, _mm256_permute2f128_si256(x, x, 0x01));
	return _mm256_add_epi64(y, _mm256_shuffle_epi32(y, 0x4e));
#else
	__m128i y = _mm_add_epi64(_mm256_extractf128_si256(x,0), _mm256_extractf128_si256(x,1));
	y = _mm_add_epi64(y, _mm_shuffle_epi32(y, 0x4e));
	return _mm256_insertf128_si256(_mm256_castsi128_si256(y), y, 1);
#endif
    }

    inline int64_t sum() const
    {
	simd_t<int64_t,2> y = _mm_add_epi64(_mm256_extractf128_si256(x,0), _mm256_extractf128_si256(x,1));
	return y.sum();
    }
};


// -------------------------------------------------------------------------------------------------
//
// blendv(mask,a,b) is morally equivalent to (mask ? a : b)

inline simd_t<int64_t,2> blendv(simd_t<int64_t,2> mask, simd_t<int64_t,2> a, simd_t<int64_t,2> b)
{ 
    __m128d xmask = _mm_castsi128_pd(mask.x);
    __m128d xa = _mm_castsi128_pd(a.x);
    __m128d xb = _mm_castsi128_pd(b.x);
    __m128d ret = _mm_blendv_pd(xb, xa, xmask);
    return _mm_castpd_si128(ret);
}

inline simd_t<int64_t,4> blendv(simd_t<int64_t,4> mask, simd_t<int64_t,4> a, simd_t<int64_t,4> b)
{ 
    __m256d xmask = _mm256_castsi256_pd(mask.x);
    __m256d xa = _mm256_castsi256_pd(a.x);
    __m256d xb = _mm256_castsi256_pd(b.x);
    __m256d ret = _mm256_blendv_pd(xb, xa, xmask);
    return _mm256_castpd_si256(ret);
}


}  // namespace simd_helpers


#endif // _SIMD_HELPERS_INT64_HPP
