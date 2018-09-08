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
// The member functions of simd_t<T,S> are mostly pretty intuitive, but for a few comments
// see the extended comment in simd_helpers/core.hpp


template<> struct simd_t<int64_t,2>
{
    using scalar_type = int64_t;
    using iscalar_type = int64_t;

    static constexpr int simd_size = 2;
    static constexpr int total_size = 2;

    __m128i x;

    simd_t() { }
    simd_t(__m128i y) { x = y; }
    simd_t(int64_t y) { x = _mm_set1_epi64x(y); }

    static inline simd_t<int64_t,2> zero()  { return _mm_setzero_si128(); }
    static inline simd_t<int64_t,2> range() { return _mm_set_epi64x(1, 0); }

    inline void load(const int64_t *p)  { x = _mm_load_si128((const __m128i *) p); }
    inline void loadu(const int64_t *p) { x = _mm_loadu_si128((const __m128i *) p); }

    inline void store(int64_t *p) const  { _mm_store_si128((__m128i *)p, x); }
    inline void storeu(int64_t *p) const { _mm_storeu_si128((__m128i *)p, x); }
    inline void stores(int64_t *p) const { _mm_stream_si128((__m128i *)p, x); }

    template<unsigned int M> inline int64_t extract() const  { return _mm_extract_epi64(x, M); }

    inline simd_t<int64_t,2> operator+(simd_t<int64_t,2> t) const { return _mm_add_epi64(x,t.x); }
    inline simd_t<int64_t,2> operator-(simd_t<int64_t,2> t) const { return _mm_sub_epi64(x,t.x); }

    // There's no simd intrinsic for 64-bit integer multiplication, so we emulate it using 32-bit multiplies.
    inline simd_t<int64_t,2> operator*(simd_t<int64_t,2> t) const
    {
	__m128i y = _mm_mul_epu32(x, t.x);
	__m128i z = _mm_mullo_epi32(x, _mm_shuffle_epi32(t.x, 0xb1));  // 0xb1 = (2301)_4
	z = _mm_add_epi64(z, _mm_shuffle_epi32(z,0xb1));
	z = _mm_blend_epi32(z, _mm_setzero_si128(), 0x5);   // 0x5 = (0101)_2	
	return _mm_add_epi64(y, z);
    }

    inline simd_t<int64_t,2> operator&(simd_t<int64_t,2> t) const { return _mm_and_si128(x,t.x); }
    inline simd_t<int64_t,2> operator|(simd_t<int64_t,2> t) const { return _mm_or_si128(x,t.x); }
    inline simd_t<int64_t,2> operator^(simd_t<int64_t,2> t) const { return _mm_xor_si128(x,t.x); }

    // Note: operator>> wraps the "shift in zeros" version of the right-shift operator.
    // FIXME define a wrapper for the "shift in sign bit" version.
    inline simd_t<int64_t,2> operator<<(simd_t<int64_t,2> t) const { return _mm_sllv_epi64(x,t.x); }
    inline simd_t<int64_t,2> operator>>(simd_t<int64_t,2> t) const { return _mm_srlv_epi64(x,t.x); }
    
    // These versions of the shift operators can only be used if 'n' is a compile-time constant.
    inline simd_t<int64_t,2> operator<<(int n) const { return _mm_slli_epi64(x,n); }
    inline simd_t<int64_t,2> operator>>(int n) const { return _mm_srli_epi64(x,n); }

    inline simd_t<int64_t,2> operator-() const
    {
	__m128i t = _mm_set1_epi16(-1);
	return _mm_xor_si128(_mm_add_epi64(x,t), t);
    }

    inline simd_t<int64_t,2> &operator+=(simd_t<int64_t,2> t)   { *this = (*this) + t; return *this; }
    inline simd_t<int64_t,2> &operator-=(simd_t<int64_t,2> t)   { *this = (*this) - t; return *this; }
    inline simd_t<int64_t,2> &operator*=(simd_t<int64_t,2> t)   { *this = (*this) * t; return *this; }
    inline simd_t<int64_t,2> &operator&=(simd_t<int64_t,2> t)   { *this = (*this) & t; return *this; }
    inline simd_t<int64_t,2> &operator|=(simd_t<int64_t,2> t)   { *this = (*this) | t; return *this; }
    inline simd_t<int64_t,2> &operator^=(simd_t<int64_t,2> t)   { *this = (*this) ^ t; return *this; }
    inline simd_t<int64_t,2> &operator<<=(simd_t<int64_t,2> t)  { *this = (*this) << t; return *this; }
    inline simd_t<int64_t,2> &operator>>=(simd_t<int64_t,2> t)  { *this = (*this) >> t; return *this; }
    inline simd_t<int64_t,2> &operator<<=(int n)                { *this = (*this) << n; return *this; }
    inline simd_t<int64_t,2> &operator>>=(int n)                { *this = (*this) >> n; return *this; }

    // FIXME implement operator!=, operator<, operator<=, operator>= (using bit flip)
    inline simd_t<int64_t,2> operator==(simd_t<int64_t,2> t) const { return _mm_cmpeq_epi64(x, t.x); }
    inline simd_t<int64_t,2> operator>(simd_t<int64_t,2> t) const  { return _mm_cmpgt_epi64(x, t.x); }

    inline simd_t<int64_t,2> abs() const
    {
	__m128i t = _mm_set1_epi16(-1);
	__m128i nx = _mm_xor_si128(_mm_add_epi64(x,t), t);
	__m128i pos = _mm_cmpgt_epi64(x, _mm_setzero_si128());
	return _mm_blendv_epi8(nx, x, pos);
    }

    // There's no simd intrinsic for 64-bit integer min/max, so we emulate it using comparison operators and blendv().
    inline simd_t<int64_t,2> min(simd_t<int64_t,2> t) const  { return _mm_blendv_epi8(x, t.x, _mm_cmpgt_epi64(x,t.x)); }
    inline simd_t<int64_t,2> max(simd_t<int64_t,2> t) const  { return _mm_blendv_epi8(t.x, x, _mm_cmpgt_epi64(x,t.x)); }

    inline simd_t<int64_t,2> horizontal_sum() const  { return _mm_add_epi64(x, _mm_shuffle_epi32(x, 0x4e)); }

    inline simd_t<int64_t,2> horizontal_max() const
    {
	simd_t<int64_t,2> y = _mm_shuffle_epi32(x, 0x4e);
	return this->max(y);
    }

    inline int64_t sum() const { return horizontal_sum().extract<0> (); }
    inline int64_t max() const { return horizontal_max().extract<0> (); }

    inline int is_all_ones() const                                        { return _mm_test_all_ones(x); }
    inline int is_all_zeros() const                                       { return _mm_testz_si128(x, x); }
    inline int is_all_zeros_masked(simd_t<int64_t,2> mask) const          { return _mm_testz_si128(x, mask.x); }
    inline int is_all_zeros_inverse_masked(simd_t<int64_t,2> mask) const  { return _mm_testc_si128(mask.x, x); }

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
};


// simd_if(mask,a,b) is morally equivalent to (mask ? a : b)
// Note that there is no x86 blendv() for integer types, need to cast to float64!
inline simd_t<int64_t,2> simd_if(simd_t<int64_t,2> mask, simd_t<int64_t,2> a, simd_t<int64_t,2> b)
{ 
    __m128d xmask = _mm_castsi128_pd(mask.x);
    __m128d xa = _mm_castsi128_pd(a.x);
    __m128d xb = _mm_castsi128_pd(b.x);
    __m128d ret = _mm_blendv_pd(xb, xa, xmask);
    return _mm_castpd_si128(ret);
}


// FIXME deprecated alias for simd_if().
inline simd_t<int64_t,2> blendv(simd_t<int64_t,2> mask, simd_t<int64_t,2> a, simd_t<int64_t,2> b) { return simd_if(mask,a,b); }


// -------------------------------------------------------------------------------------------------
//
// simd_t<int64_t,4>


#ifdef __AVX__

template<> struct simd_t<int64_t,4>
{
    using scalar_type = int64_t;
    using iscalar_type = int64_t;

    static constexpr int simd_size = 4;
    static constexpr int total_size = 4;

    __m256i x;

    simd_t() { }
    simd_t(__m256i y)                                 { x = y; }
    simd_t(int64_t y)                                 { x = _mm256_set1_epi64x(y); }
    simd_t(simd_t<int64_t,2> y, simd_t<int64_t,2> z)  { x = _mm256_insertf128_si256(_mm256_castsi128_si256(y.x), (z.x), 1); }

    static inline simd_t<int64_t,4> zero()  { return _mm256_setzero_si256(); }
    static inline simd_t<int64_t,4> range() { return _mm256_set_epi64x(3, 2, 1, 0); }

    inline void load(const int64_t *p)  { x = _mm256_load_si256((const __m256i *) p); }
    inline void loadu(const int64_t *p) { x = _mm256_loadu_si256((const __m256i *) p); }

    inline void store(int64_t *p) const  { _mm256_store_si256((__m256i *)p, x); }
    inline void storeu(int64_t *p) const { _mm256_storeu_si256((__m256i *)p, x); }
    inline void stores(int64_t *p) const { _mm256_stream_si256((__m256i *)p, x); }

    template<unsigned int M> inline int64_t extract() const                 { return _mm256_extract_epi64(x,M); }
    template<unsigned int M> inline simd_t<int64_t,2> extract_half() const  { return _mm256_extractf128_si256(x,M); }

    inline simd_t<int64_t,4> operator+(simd_t<int64_t,4> t) const 
    { 
#ifdef __AVX2__
	return _mm256_add_epi64(x,t.x); 
#else
	simd_t<int64_t,2> x0 = extract_half<0> ();
	simd_t<int64_t,2> x1 = extract_half<1> ();
	simd_t<int64_t,2> t0 = t.extract_half<0> ();
	simd_t<int64_t,2> t1 = t.extract_half<1> ();
	return simd_t<int64_t,4> (x0+t0, x1+t1);
#endif
    }

    inline simd_t<int64_t,4> operator-(simd_t<int64_t,4> t) const 
    { 
#ifdef __AVX2__
	return _mm256_sub_epi64(x,t.x); 
#else
	simd_t<int64_t,2> x0 = extract_half<0> ();
	simd_t<int64_t,2> x1 = extract_half<1> ();
	simd_t<int64_t,2> t0 = t.extract_half<0> ();
	simd_t<int64_t,2> t1 = t.extract_half<1> ();
	return simd_t<int64_t,4> (x0-t0, x1-t1);
#endif
    }

    // There's no simd intrinsic for 64-bit integer multiplication, so we emulate it using 32-bit multiplies.
    inline simd_t<int64_t,4> operator*(simd_t<int64_t,4> t) const
    {
#ifdef __AVX2__
	__m256i y = _mm256_mul_epu32(x, t.x);
	__m256i z = _mm256_mullo_epi32(x, _mm256_shuffle_epi32(t.x, 0xb1));  // 0xb1 = (2301)_4
	z = _mm256_add_epi64(z, _mm256_shuffle_epi32(z,0xb1));
	z = _mm256_blend_epi32(z, _mm256_setzero_si256(), 0x55);   // 0x55 = (01010101)_2	
	return _mm256_add_epi64(y, z);
#else
	simd_t<int64_t,2> x0 = extract_half<0> ();
	simd_t<int64_t,2> x1 = extract_half<1> ();
	simd_t<int64_t,2> t0 = t.extract_half<0> ();
	simd_t<int64_t,2> t1 = t.extract_half<1> ();
	return simd_t<int64_t,4> (x0*t0, x1*t1);
#endif
    }

    // Unary minus
    inline simd_t<int64_t,4> operator-() const
    {
#ifdef __AVX2__
	__m256i t = _mm256_set1_epi16(-1);
	return _mm256_xor_si256(_mm256_add_epi64(x,t), t);
#else
	return simd_t<int64_t,4> (-extract_half<0>(), -extract_half<1>());
#endif
    }

    inline simd_t<int64_t,4> operator&(simd_t<int64_t,4> t) const  { return _mm256_and_si256(x,t.x); }
    inline simd_t<int64_t,4> operator|(simd_t<int64_t,4> t) const  { return _mm256_or_si256(x,t.x); }
    inline simd_t<int64_t,4> operator^(simd_t<int64_t,4> t) const  { return _mm256_xor_si256(x,t.x); }

    // Note: operator>> wraps the "shift in zeros" version of the right-shift operator.
    // FIXME define a wrapper for the "shift in sign bit" version.
    inline simd_t<int64_t,4> operator<<(simd_t<int64_t,4> t) const { return _mm256_sllv_epi64(x,t.x); }
    inline simd_t<int64_t,4> operator>>(simd_t<int64_t,4> t) const { return _mm256_srlv_epi64(x,t.x); }
    
    // These versions of the shift operators can only be used if 'n' is a compile-time constant.
    inline simd_t<int64_t,4> operator<<(int n) const { return _mm256_slli_epi64(x,n); }
    inline simd_t<int64_t,4> operator>>(int n) const { return _mm256_srli_epi64(x,n); }
    
    inline simd_t<int64_t,4> &operator+=(simd_t<int64_t,4> t)   { *this = (*this) + t; return *this; }
    inline simd_t<int64_t,4> &operator-=(simd_t<int64_t,4> t)   { *this = (*this) - t; return *this; }
    inline simd_t<int64_t,4> &operator*=(simd_t<int64_t,4> t)   { *this = (*this) * t; return *this; }
    inline simd_t<int64_t,4> &operator&=(simd_t<int64_t,4> t)   { *this = (*this) & t; return *this; }
    inline simd_t<int64_t,4> &operator|=(simd_t<int64_t,4> t)   { *this = (*this) | t; return *this; }
    inline simd_t<int64_t,4> &operator^=(simd_t<int64_t,4> t)   { *this = (*this) ^ t; return *this; }
    inline simd_t<int64_t,4> &operator<<=(simd_t<int64_t,4> t)  { *this = (*this) << t; return *this; }
    inline simd_t<int64_t,4> &operator>>=(simd_t<int64_t,4> t)  { *this = (*this) >> t; return *this; }
    inline simd_t<int64_t,4> &operator<<=(int n)                { *this = (*this) << n; return *this; }
    inline simd_t<int64_t,4> &operator>>=(int n)                { *this = (*this) >> n; return *this; }

    // FIXME implement operator!=, operator<, operator<=, operator>= (using bit flip)
    // FIXME this assumes AVX2, write AVX-but-no-AVX2 version.
    inline simd_t<int64_t,4> operator==(simd_t<int64_t,4> t) const { return _mm256_cmpeq_epi64(x, t.x); }
    inline simd_t<int64_t,4> operator>(simd_t<int64_t,4> t) const  { return _mm256_cmpgt_epi64(x, t.x); }

    inline simd_t<int64_t,4> abs() const
    {
#ifdef __AVX2__
	__m256i t = _mm256_set1_epi16(-1);
	__m256i nx = _mm256_xor_si256(_mm256_add_epi64(x,t), t);
	__m256i pos = _mm256_cmpgt_epi64(x, _mm256_setzero_si256());
	return _mm256_blendv_epi8(nx, x, pos);
#else
	simd_t<int64_t,2> x0 = extract_half<0> ();
	simd_t<int64_t,2> x1 = extract_half<1> ();
	return simd_t<int64_t,4> (x0.abs(), x1.abs());
#endif
    }

    // There's no simd intrinsic for 64-bit integer min/max, so we emulate it using comparison operators and blendv().
    inline simd_t<int64_t,4> min(simd_t<int64_t,4> t) const
    {
#ifdef __AVX2__
	return _mm256_blendv_epi8(x, t.x, _mm256_cmpgt_epi64(x,t.x));
#else
	simd_t<int64_t,2> x0 = extract_half<0> ();
	simd_t<int64_t,2> x1 = extract_half<1> ();
	simd_t<int64_t,2> t0 = t.extract_half<0> ();
	simd_t<int64_t,2> t1 = t.extract_half<1> ();
	return simd_t<int64_t,4> (x0.min(t0), x1.min(t1));
#endif
    }

    inline simd_t<int64_t,4> max(simd_t<int64_t,4> t) const
    {
#ifdef __AVX2__
	return _mm256_blendv_epi8(t.x, x, _mm256_cmpgt_epi64(x,t.x));
#else
	simd_t<int64_t,2> x0 = extract_half<0> ();
	simd_t<int64_t,2> x1 = extract_half<1> ();
	simd_t<int64_t,2> t0 = t.extract_half<0> ();
	simd_t<int64_t,2> t1 = t.extract_half<1> ();
	return simd_t<int64_t,4> (x0.max(t0), x1.max(t1));
#endif
    }

    inline simd_t<int64_t,4> horizontal_sum() const
    { 
#ifdef __AVX2__
	__m256i y = _mm256_add_epi64(x, _mm256_permute2f128_si256(x, x, 0x01));
	return _mm256_add_epi64(y, _mm256_shuffle_epi32(y, 0x4e));
#else
	simd_t<int64_t,2> y = extract_half<0> () + extract_half<1> ();
	y = y.horizontal_sum();
	return simd_t<int64_t,4> (y, y);
#endif
    }

    inline simd_t<int64_t,4> horizontal_max() const
    {
#ifdef __AVX2__
	__m256i y = _mm256_permute2f128_si256(x, x, 0x01);
	y = _mm256_blendv_epi8(y, x, _mm256_cmpgt_epi64(x,y));
	__m256i z = _mm256_shuffle_epi32(y, 0x4e);
	return  _mm256_blendv_epi8(z, y, _mm256_cmpgt_epi64(y,z));
#else
	simd_t<int64_t,2> x0 = extract_half<0>();
	simd_t<int64_t,2> x1 = extract_half<1>();
	simd_t<int64_t,2> y = x0.max(x1).horizontal_max();
	return simd_t<int64_t,4> (y,y);
#endif
    }

    inline int64_t sum() const
    {
	simd_t<int64_t,2> y = extract_half<0> () + extract_half<1> ();
	return y.sum();
    }

    inline int64_t max() const
    {
	simd_t<int64_t,2> x0 = extract_half<0>();
	simd_t<int64_t,2> x1 = extract_half<1>();
	return x0.max(x1).max();
    }

    inline int is_all_ones() const                                        { return _mm256_testc_si256(x, _mm256_set1_epi32(-1)); }
    inline int is_all_zeros() const                                       { return _mm256_testz_si256(x, x); }
    inline int is_all_zeros_masked(simd_t<int64_t,4> mask) const          { return _mm256_testz_si256(x, mask.x); }
    inline int is_all_zeros_inverse_masked(simd_t<int64_t,4> mask) const  { return _mm256_testc_si256(mask.x, x); }

    inline simd_t<int64_t,4> compare_eq(simd_t<int64_t,4> t) const 
    { 
#ifdef __AVX2__
	return _mm256_cmpeq_epi64(x, t.x); 
#else
	simd_t<int64_t,2> x0 = extract_half<0> ();
	simd_t<int64_t,2> x1 = extract_half<1> ();
	simd_t<int64_t,2> t0 = t.extract_half<0> ();
	simd_t<int64_t,2> t1 = t.extract_half<1> ();
	return simd_t<int64_t,4> (x0.compare_eq(t0), x1.compare_eq(t1));
#endif
    }

    inline simd_t<int64_t,4> compare_gt(simd_t<int64_t,4> t) const  
    { 
#ifdef __AVX2__
	return _mm256_cmpgt_epi64(x, t.x); 
#else
	simd_t<int64_t,2> x0 = extract_half<0> ();
	simd_t<int64_t,2> x1 = extract_half<1> ();
	simd_t<int64_t,2> t0 = t.extract_half<0> ();
	simd_t<int64_t,2> t1 = t.extract_half<1> ();
	return simd_t<int64_t,4> (x0.compare_gt(t0), x1.compare_gt(t1));
#endif
    }

    inline simd_t<int64_t,4> compare_ge(simd_t<int64_t,4> t) const  
    { 
#ifdef __AVX2__
	return compare_lt(t).bitwise_not(); 
#else
	simd_t<int64_t,2> x0 = extract_half<0> ();
	simd_t<int64_t,2> x1 = extract_half<1> ();
	simd_t<int64_t,2> t0 = t.extract_half<0> ();
	simd_t<int64_t,2> t1 = t.extract_half<1> ();
	return simd_t<int64_t,4> (x0.compare_ge(t0), x1.compare_ge(t1));
#endif
    }

    inline simd_t<int64_t,4> compare_ne(simd_t<int64_t,4> t) const  
    { 
#ifdef __AVX2__
	return compare_eq(t).bitwise_not(); 
#else
	simd_t<int64_t,2> x0 = extract_half<0> ();
	simd_t<int64_t,2> x1 = extract_half<1> ();
	simd_t<int64_t,2> t0 = t.extract_half<0> ();
	simd_t<int64_t,2> t1 = t.extract_half<1> ();
	return simd_t<int64_t,4> (x0.compare_ne(t0), x1.compare_ne(t1));
#endif
    }

    inline simd_t<int64_t,4> compare_le(simd_t<int64_t,4> t) const  { return t.compare_ge(*this); }
    inline simd_t<int64_t,4> compare_lt(simd_t<int64_t,4> t) const  { return t.compare_gt(*this); }
    
    inline simd_t<int64_t,4> apply_mask(simd_t<int64_t,4> t) const          { return bitwise_and(t); }
    inline simd_t<int64_t,4> apply_inverse_mask(simd_t<int64_t,4> t) const  { return bitwise_andnot(t); }

    inline simd_t<int64_t,4> bitwise_and(simd_t<int64_t,4> t) const     { return _mm256_and_si256(x, t.x); }
    inline simd_t<int64_t,4> bitwise_or(simd_t<int64_t,4> t) const      { return _mm256_or_si256(x, t.x);  }
    inline simd_t<int64_t,4> bitwise_xor(simd_t<int64_t,4> t) const     { return _mm256_xor_si256(x, t.x); }
    inline simd_t<int64_t,4> bitwise_andnot(simd_t<int64_t,4> t) const  { return _mm256_andnot_si256(t.x, x); }
    inline simd_t<int64_t,4> bitwise_not() const                        { return _mm256_xor_si256(x, _mm256_set1_epi16(-1)); }
};


// simd_if(mask,a,b) is morally equivalent to (mask ? a : b)
// Note that there is no x86 blendv() for integer types, need to cast to float64!
inline simd_t<int64_t,4> simd_if(simd_t<int64_t,4> mask, simd_t<int64_t,4> a, simd_t<int64_t,4> b)
{ 
    __m256d xmask = _mm256_castsi256_pd(mask.x);
    __m256d xa = _mm256_castsi256_pd(a.x);
    __m256d xb = _mm256_castsi256_pd(b.x);
    __m256d ret = _mm256_blendv_pd(xb, xa, xmask);
    return _mm256_castpd_si256(ret);
}

// FIXME deprecated alias for simd_if().
inline simd_t<int64_t,4> blendv(simd_t<int64_t,4> mask, simd_t<int64_t,4> a, simd_t<int64_t,4> b) { return simd_if(mask,a,b); }


#endif // __AVX__


}  // namespace simd_helpers


#endif // _SIMD_HELPERS_INT64_HPP
