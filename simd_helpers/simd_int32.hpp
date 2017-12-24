#ifndef _SIMD_HELPERS_INT32_HPP
#define _SIMD_HELPERS_INT32_HPP

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
// simd_t<int,4>
//
// The member functions of simd_t<T,S> are mostly pretty intuitive, but for a few comments
// see the extended comment in simd_helpers/core.hpp


template<> struct simd_t<int,4>
{
    __m128i x;

    simd_t() { }
    simd_t(__m128i y) { x = y; }
    simd_t(int y)     { x = _mm_set1_epi32(y); }

    static inline simd_t<int,4> zero()  { return _mm_setzero_si128(); }
    static inline simd_t<int,4> range() { return _mm_set_epi32(3, 2, 1, 0); }

    inline void load(const int *p)  { x = _mm_load_si128((const __m128i *) p); }
    inline void loadu(const int *p) { x = _mm_loadu_si128((const __m128i *) p); }

    inline void store(int *p) const  { _mm_store_si128((__m128i *)p, x); }
    inline void storeu(int *p) const { _mm_storeu_si128((__m128i *)p, x); }
    inline void stores(int *p) const { _mm_stream_si128((__m128i *)p, x); }

    template<unsigned int M> inline int extract() const  { return _mm_extract_epi32(x, M); }

    inline simd_t<int,4> operator+(simd_t<int,4> t) const { return _mm_add_epi32(x,t.x); }
    inline simd_t<int,4> operator-(simd_t<int,4> t) const { return _mm_sub_epi32(x,t.x); }
    inline simd_t<int,4> operator*(simd_t<int,4> t) const { return _mm_mullo_epi32(x,t.x); }
    inline simd_t<int,4> operator&(simd_t<int,4> t) const { return _mm_and_si128(x,t.x); }
    inline simd_t<int,4> operator|(simd_t<int,4> t) const { return _mm_or_si128(x,t.x); }
    inline simd_t<int,4> operator^(simd_t<int,4> t) const { return _mm_xor_si128(x,t.x); }

    // Note: operator>> wraps the "shift in zeros" version of the right-shift operator.
    // FIXME define a wrapper for the "shift in sign bit" version.
    inline simd_t<int,4> operator<<(simd_t<int,4> t) const { return _mm_sllv_epi32(x,t.x); }
    inline simd_t<int,4> operator>>(simd_t<int,4> t) const { return _mm_srlv_epi32(x,t.x); }
    
    // These versions of the shift operators can only be used if 'n' is a compile-time constant.
    inline simd_t<int,4> operator<<(int n) const { return _mm_slli_epi32(x,n); }
    inline simd_t<int,4> operator>>(int n) const { return _mm_srli_epi32(x,n); }

    inline simd_t<int,4> operator-() const
    {
	__m128i t = _mm_set1_epi16(-1);
	return _mm_xor_si128(_mm_add_epi32(x,t), t);
    }

    inline simd_t<int,4> &operator+=(simd_t<int,4> t)   { *this = *this + t; return *this; }
    inline simd_t<int,4> &operator-=(simd_t<int,4> t)   { *this = *this - t; return *this; }
    inline simd_t<int,4> &operator*=(simd_t<int,4> t)   { *this = *this * t; return *this; }
    inline simd_t<int,4> &operator&=(simd_t<int,4> t)   { *this = *this & t; return *this; }
    inline simd_t<int,4> &operator|=(simd_t<int,4> t)   { *this = *this | t; return *this; }
    inline simd_t<int,4> &operator^=(simd_t<int,4> t)   { *this = *this ^ t; return *this; }
    inline simd_t<int,4> &operator<<=(simd_t<int,4> t)  { *this = *this << t; return *this; }
    inline simd_t<int,4> &operator>>=(simd_t<int,4> t)  { *this = *this >> t; return *this; }
    inline simd_t<int,4> &operator<<=(int n)            { *this = *this << n; return *this; }
    inline simd_t<int,4> &operator>>=(int n)            { *this = *this >> n; return *this; }

    inline simd_t<int,4> abs() const { return _mm_abs_epi32(x); }
    inline simd_t<int,4> min(simd_t<int,4> t) const { return _mm_min_epi32(x, t.x); }
    inline simd_t<int,4> max(simd_t<int,4> t) const { return _mm_max_epi32(x, t.x); }

    inline simd_t<int,4> horizontal_sum() const
    {
	__m128i y = _mm_add_epi32(x, _mm_shuffle_epi32(x, 0xb1));  // (2301)_4 = 0xb1
	return _mm_add_epi32(y, _mm_shuffle_epi32(y, 0x4e));       // (1032)_4 = 0x4e
    }

    inline simd_t<int,4> horizontal_max() const
    {
	__m128i y = _mm_max_epi32(x, _mm_shuffle_epi32(x, 0xb1));  // (2301)_4 = 0xb1
	return _mm_max_epi32(y, _mm_shuffle_epi32(y, 0x4e));       // (1032)_4 = 0x4e
    }

    inline int sum() const { return _mm_extract_epi32(horizontal_sum().x, 0); }
    inline int max() const { return _mm_extract_epi32(horizontal_max().x, 0); }

    inline int is_all_ones() const                                    { return _mm_test_all_ones(x); }
    inline int is_all_zeros() const                                   { return _mm_testz_si128(x, x); }
    inline int is_all_zeros_masked(simd_t<int,4> mask) const          { return _mm_testz_si128(x, mask.x); }
    inline int is_all_zeros_inverse_masked(simd_t<int,4> mask) const  { return _mm_testc_si128(mask.x, x); }

    inline simd_t<int,4> compare_eq(simd_t<int,4> t) const  { return _mm_cmpeq_epi32(x, t.x); }
    inline simd_t<int,4> compare_gt(simd_t<int,4> t) const  { return _mm_cmpgt_epi32(x, t.x); }
    inline simd_t<int,4> compare_lt(simd_t<int,4> t) const  { return _mm_cmplt_epi32(x, t.x); }
    inline simd_t<int,4> compare_ne(simd_t<int,4> t) const  { return compare_eq(t).bitwise_not(); }
    inline simd_t<int,4> compare_ge(simd_t<int,4> t) const  { return compare_lt(t).bitwise_not(); }
    inline simd_t<int,4> compare_le(simd_t<int,4> t) const  { return compare_gt(t).bitwise_not(); }

    inline simd_t<int,4> apply_mask(simd_t<int,4> t) const          { return bitwise_and(t); }
    inline simd_t<int,4> apply_inverse_mask(simd_t<int,4> t) const  { return bitwise_andnot(t); }

    inline simd_t<int,4> bitwise_and(simd_t<int,4> t) const     { return _mm_and_si128(x, t.x); }
    inline simd_t<int,4> bitwise_or(simd_t<int,4> t) const      { return _mm_or_si128(x, t.x); }
    inline simd_t<int,4> bitwise_xor(simd_t<int,4> t) const     { return _mm_xor_si128(x, t.x); }
    inline simd_t<int,4> bitwise_andnot(simd_t<int,4> t) const  { return _mm_andnot_si128(t.x, x); }
    inline simd_t<int,4> bitwise_not() const                    { return _mm_xor_si128(x, _mm_set1_epi16(-1)); }
};


// blendv(mask,a,b) is morally equivalent to (mask ? a : b)
inline simd_t<int,4> blendv(simd_t<int,4> mask, simd_t<int,4> a, simd_t<int,4> b)
{ 
    __m128 xmask = _mm_castsi128_ps(mask.x);
    __m128 xa = _mm_castsi128_ps(a.x);
    __m128 xb = _mm_castsi128_ps(b.x);
    __m128 ret = _mm_blendv_ps(xb, xa, xmask);
    return _mm_castps_si128(ret);
}


// -------------------------------------------------------------------------------------------------
//
// simd_t<int,8>


#ifdef __AVX__

template<> struct simd_t<int,8>
{
    __m256i x;

    simd_t() { }
    simd_t(__m256i y)                         { x = y; }
    simd_t(int y)                             { x = _mm256_set1_epi32(y); }
    simd_t(simd_t<int,4> y, simd_t<int,4> z)  { x = _mm256_insertf128_si256(_mm256_castsi128_si256(y.x), (z.x), 1); }

    static inline simd_t<int,8> zero()  { return _mm256_setzero_si256(); }
    static inline simd_t<int,8> range() { return _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0); }

    inline void load(const int *p)  { x = _mm256_load_si256((const __m256i *) p); }
    inline void loadu(const int *p) { x = _mm256_loadu_si256((const __m256i *) p); }

    inline void store(int *p) const  { _mm256_store_si256((__m256i *)p, x); }
    inline void storeu(int *p) const { _mm256_storeu_si256((__m256i *)p, x); }
    inline void stores(int *p) const { _mm256_stream_si256((__m256i *)p, x); }

    template<unsigned int M> inline int extract() const                 { return _mm256_extract_epi32(x,M); }
    template<unsigned int M> inline simd_t<int,4> extract_half() const  { return _mm256_extractf128_si256(x,M); }

    inline simd_t<int,8> operator+(simd_t<int,8> t) const 
    { 
#ifdef __AVX2__
	return _mm256_add_epi32(x,t.x); 
#else
	simd_t<int,4> ret0 = extract_half<0>() + t.extract_half<0> ();
	simd_t<int,4> ret1 = extract_half<1>() + t.extract_half<1> ();
	return simd_t<int,8> (ret0, ret1);
#endif
    }

    inline simd_t<int,8> operator-(simd_t<int,8> t) 
    {
#ifdef __AVX2__
	return _mm256_sub_epi32(x,t.x); 
#else
	simd_t<int,4> ret0 = extract_half<0>() - t.extract_half<0> ();
	simd_t<int,4> ret1 = extract_half<1>() - t.extract_half<1> ();
	return simd_t<int,8> (ret0, ret1);
#endif
    }

    inline simd_t<int,8> operator*(simd_t<int,8> t) const 
    { 
#ifdef __AVX2__
	return _mm256_mullo_epi32(x,t.x); 
#else
	simd_t<int,4> ret0 = extract_half<0>() * t.extract_half<0> ();
	simd_t<int,4> ret1 = extract_half<1>() * t.extract_half<1> ();
	return simd_t<int,8> (ret0, ret1);
#endif
    }

    // Unary minus
    inline simd_t<int,8> operator-() const
    {
#ifdef __AVX2__
	__m256i t = _mm256_set1_epi16(-1);
	return _mm256_xor_si256(_mm256_add_epi32(x,t), t);
#else
	return simd_t<int,8> (-extract_half<0>(), -extract_half<1>());
#endif
    }

    inline simd_t<int,8> operator&(simd_t<int,8> t) const { return _mm256_and_si256(x,t.x); }
    inline simd_t<int,8> operator|(simd_t<int,8> t) const { return _mm256_or_si256(x,t.x); }
    inline simd_t<int,8> operator^(simd_t<int,8> t) const { return _mm256_xor_si256(x,t.x); }

    // Note: operator>> wraps the "shift in zeros" version of the right-shift operator.
    // FIXME define a wrapper for the "shift in sign bit" version.
    inline simd_t<int,8> operator<<(simd_t<int,8> t) const { return _mm256_sllv_epi32(x,t.x); }
    inline simd_t<int,8> operator>>(simd_t<int,8> t) const { return _mm256_srlv_epi32(x,t.x); }
    
    // These versions of the shift operators can only be used if 'n' is a compile-time constant.
    inline simd_t<int,8> operator<<(int n) const { return _mm256_slli_epi32(x,n); }
    inline simd_t<int,8> operator>>(int n) const { return _mm256_srli_epi32(x,n); }

    inline simd_t<int,8> &operator+=(simd_t<int,8> t)   { *this = (*this) + t; return *this; }
    inline simd_t<int,8> &operator-=(simd_t<int,8> t)   { *this = (*this) - t; return *this; }
    inline simd_t<int,8> &operator*=(simd_t<int,8> t)   { *this = (*this) * t; return *this; }
    inline simd_t<int,8> &operator&=(simd_t<int,8> t)   { *this = (*this) & t; return *this; }
    inline simd_t<int,8> &operator|=(simd_t<int,8> t)   { *this = (*this) | t; return *this; }
    inline simd_t<int,8> &operator^=(simd_t<int,8> t)   { *this = (*this) ^ t; return *this; }
    inline simd_t<int,8> &operator<<=(simd_t<int,8> t)  { *this = (*this) << t; return *this; }
    inline simd_t<int,8> &operator>>=(simd_t<int,8> t)  { *this = (*this) >> t; return *this; }
    inline simd_t<int,8> &operator<<=(int n)            { *this = (*this) << n; return *this; }
    inline simd_t<int,8> &operator>>=(int n)            { *this = (*this) >> n; return *this; }
    
    inline simd_t<int,8> abs() const
    {
#ifdef __AVX2__
	return _mm256_abs_epi32(x); 
#else
	simd_t<int,4> x0 = extract_half<0> ();
	simd_t<int,4> x1 = extract_half<1> ();
	return simd_t<int,8> (x0.abs(), x1.abs());
#endif
    }

    inline simd_t<int,8> min(simd_t<int,8> t) const
    {
#ifdef __AVX2__
	return _mm256_min_epi32(x, t.x); 
#else
	simd_t<int,4> x0 = extract_half<0> ();
	simd_t<int,4> x1 = extract_half<1> ();
	simd_t<int,4> t0 = t.extract_half<0> ();
	simd_t<int,4> t1 = t.extract_half<1> ();
	return simd_t<int,8> (x0.min(t0), x1.min(t1));
#endif
    }

    inline simd_t<int,8> max(simd_t<int,8> t) const
    {
#ifdef __AVX2__
	return _mm256_max_epi32(x, t.x); 
#else
	simd_t<int,4> x0 = extract_half<0> ();
	simd_t<int,4> x1 = extract_half<1> ();
	simd_t<int,4> t0 = t.extract_half<0> ();
	simd_t<int,4> t1 = t.extract_half<1> ();
	return simd_t<int,8> (x0.max(t0), x1.max(t1));
#endif
    }

    inline simd_t<int,8> horizontal_sum() const
    {
#ifdef __AVX2__
	__m256i y = _mm256_add_epi32(x, _mm256_shuffle_epi32(x, 0xb1));  // (2301)_4 = 0xb1
	y = _mm256_add_epi32(y, _mm256_shuffle_epi32(y, 0x4e));          // (1032)_4 = 0x4e
	return _mm256_add_epi32(y, _mm256_permute2f128_si256(y, y, 0x01));
#else
	simd_t<int,4> y = extract_half<0>() + extract_half<1>();
	y = y.horizontal_sum();
	return simd_t<int,8> (y, y);
#endif
    }

    inline simd_t<int,8> horizontal_max() const
    {
#ifdef __AVX2__
	__m256i y = _mm256_max_epi32(x, _mm256_shuffle_epi32(x, 0xb1));  // (2301)_4 = 0xb1
	y = _mm256_max_epi32(y, _mm256_shuffle_epi32(y, 0x4e));          // (1032)_4 = 0x4e
	return _mm256_max_epi32(y, _mm256_permute2f128_si256(y, y, 0x01));
#else
	simd_t<int,4> y = extract_half<0>();
	y = y.max(extract_half<1>());
	y = y.horizontal_max();
	return simd_t<int,8> (y, y);
#endif
    }

    inline int sum() const
    { 
	simd_t<int,4> y = extract_half<0>() + extract_half<1>();
	return y.sum();
    }

    inline int max() const
    { 
	simd_t<int,4> y = extract_half<0>();
	y = y.max(extract_half<1>());
	return y.max();
    }

    inline int is_all_ones() const                                    { return _mm256_testc_si256(x, _mm256_set1_epi32(-1)); }
    inline int is_all_zeros() const                                   { return _mm256_testz_si256(x, x); }
    inline int is_all_zeros_masked(simd_t<int,8> mask) const          { return _mm256_testz_si256(x, mask.x); }
    inline int is_all_zeros_inverse_masked(simd_t<int,8> mask) const  { return _mm256_testc_si256(mask.x, x); }

    inline simd_t<int,8> compare_eq(simd_t<int,8> t) const
    {
#ifdef __AVX2__
	return _mm256_cmpeq_epi32(x, t.x); 
#else
	simd_t<int,4> x0 = extract_half<0> ();
	simd_t<int,4> x1 = extract_half<1> ();
	simd_t<int,4> t0 = t.extract_half<0> ();
	simd_t<int,4> t1 = t.extract_half<1> ();
	return simd_t<int,8> (x0.compare_eq(t0), x1.compare_eq(t1));
#endif
    }

    inline simd_t<int,8> compare_gt(simd_t<int,8> t) const
    {
#ifdef __AVX2__
	return _mm256_cmpgt_epi32(x, t.x); 
#else
	simd_t<int,4> x0 = extract_half<0> ();
	simd_t<int,4> x1 = extract_half<1> ();
	simd_t<int,4> t0 = t.extract_half<0> ();
	simd_t<int,4> t1 = t.extract_half<1> ();
	return simd_t<int,8> (x0.compare_gt(t0), x1.compare_gt(t1));
#endif
    }

    inline simd_t<int,8> compare_ge(simd_t<int,8> t) const  
    { 
#ifdef __AVX2__
	return compare_lt(t).bitwise_not(); 
#else
	simd_t<int,4> x0 = extract_half<0> ();
	simd_t<int,4> x1 = extract_half<1> ();
	simd_t<int,4> t0 = t.extract_half<0> ();
	simd_t<int,4> t1 = t.extract_half<1> ();
	return simd_t<int,8> (x0.compare_ge(t0), x1.compare_ge(t1));
#endif
    }

    inline simd_t<int,8> compare_ne(simd_t<int,8> t) const  
    {
#ifdef __AVX2__
	return compare_eq(t).bitwise_not(); 
#else
	simd_t<int,4> x0 = extract_half<0> ();
	simd_t<int,4> x1 = extract_half<1> ();
	simd_t<int,4> t0 = t.extract_half<0> ();
	simd_t<int,4> t1 = t.extract_half<1> ();
	return simd_t<int,8> (x0.compare_ne(t0), x1.compare_ne(t1));
#endif
    }

    inline simd_t<int,8> compare_lt(simd_t<int,8> t) const  { return t.compare_gt(x); }
    inline simd_t<int,8> compare_le(simd_t<int,8> t) const  { return t.compare_ge(x); }

    inline simd_t<int,8> apply_mask(simd_t<int,8> t) const          { return bitwise_and(t); }
    inline simd_t<int,8> apply_inverse_mask(simd_t<int,8> t) const  { return bitwise_andnot(t); }

    inline simd_t<int,8> bitwise_and(simd_t<int,8> t) const     { return _mm256_and_si256(x, t.x); }
    inline simd_t<int,8> bitwise_or(simd_t<int,8> t) const      { return _mm256_or_si256(x, t.x); }
    inline simd_t<int,8> bitwise_xor(simd_t<int,8> t) const     { return _mm256_xor_si256(x, t.x); }
    inline simd_t<int,8> bitwise_andnot(simd_t<int,8> t) const  { return _mm256_andnot_si256(t.x, x); }
    inline simd_t<int,8> bitwise_not() const                    { return _mm256_xor_si256(x, _mm256_set1_epi16(-1)); }
};


// blendv(mask,a,b) is morally equivalent to (mask ? a : b)
inline simd_t<int,8> blendv(simd_t<int,8> mask, simd_t<int,8> a, simd_t<int,8> b)
{ 
    __m256 xmask = _mm256_castsi256_ps(mask.x);
    __m256 xa = _mm256_castsi256_ps(a.x);
    __m256 xb = _mm256_castsi256_ps(b.x);
    __m256 ret = _mm256_blendv_ps(xb, xa, xmask);
    return _mm256_castps_si256(ret);
}

#endif  // __AVX__


}  // namespace simd_helpers


#endif // _SIMD_HELPERS_INT32_HPP
