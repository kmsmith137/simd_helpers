#ifndef _SIMD_HELPERS_INT64_HPP
#define _SIMD_HELPERS_INT64_HPP

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
// simd_t<int64_t,2>
//
// Note: for a "declaration" of the key class simd_t<T,S>,
// see the extended comment in simd_helpers/base.hpp


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

    inline simd_t<int64_t,2> compare_eq(simd_t<int64_t,2> t) const  { return _mm_cmpeq_epi64(x, t.x); }
    inline simd_t<int64_t,2> compare_gt(simd_t<int64_t,2> t) const  { return _mm_cmpgt_epi64(x, t.x); }
    inline simd_t<int64_t,2> compare_lt(simd_t<int64_t,2> t) const  { return _mm_cmpgt_epi64(t.x, x); }
    inline simd_t<int64_t,2> compare_ne(simd_t<int64_t,2> t) const  { return compare_eq(t).bitwise_not(); }
    inline simd_t<int64_t,2> compare_ge(simd_t<int64_t,2> t) const  { return compare_lt(t).bitwise_not(); }
    inline simd_t<int64_t,2> compare_le(simd_t<int64_t,2> t) const  { return compare_gt(t).bitwise_not(); }

    inline simd_t<int64_t,2> bitwise_and(simd_t<int64_t,2> t) const     { return _mm_and_si128(x, t.x); }
    inline simd_t<int64_t,2> bitwise_or(simd_t<int64_t,2> t) const      { return _mm_or_si128(x, t.x); }
    inline simd_t<int64_t,2> bitwise_xor(simd_t<int64_t,2> t) const     { return _mm_xor_si128(x, t.x); }
    inline simd_t<int64_t,2> bitwise_andnot(simd_t<int64_t,2> t) const  { return _mm_andnot_si128(x, t.x); }
    inline simd_t<int64_t,2> bitwise_not() const                        { return _mm_xor_si128(x, _mm_set1_epi16(-1)); }

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

    template<unsigned int M> inline int extract() const                    { return _mm256_extract_epi64(x,M); }
    template<unsigned int M> inline simd_t<int64_t,2> extract128() const   { return _mm256_extractf128_si256(x,M); }

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
	return _mm256_andnot_si256(x, t.x); 
#else
	simd_t<int64_t,2> ret0 = _mm_andnot_si128(_mm256_extractf128_si256(x,0), _mm256_extractf128_si256(t.x,0));
	simd_t<int64_t,2> ret1 = _mm_andnot_si128(_mm256_extractf128_si256(x,1), _mm256_extractf128_si256(t.x,1));
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


}  // namespace simd_helpers


#endif // _SIMD_HELPERS_INT64_HPP
