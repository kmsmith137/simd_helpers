#ifndef _SIMD_HELPERS_SIMD_T_IMPLEMENTATIONS_HPP
#define _SIMD_HELPERS_SIMD_T_IMPLEMENTATIONS_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

#include "simd_t.hpp"

namespace simd_helpers {
#if 0
}  // pacify emacs c-mode
#endif


inline simd_t<int,4> simd_t<int,4>::horizontal_sum() const
{
    __m128i y = x + _mm_shuffle_epi32(x, 0xb1);  // (2301)_4 = 0xb1
    return y + _mm_shuffle_epi32(y, 0x4e);       // (1032)_4 = 0x4e
}


inline simd_t<int,8> simd_t<int,8>::abs() const
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


inline simd_t<int,8> simd_t<int,8>::compare_eq(simd_t<int,8> t) const
{ 
#ifdef __AVX2__
    return _mm256_cmpeq_epi32(x, t.x); 
#else
    simd_t<int,4> ret0 = _mm_cmpeq_epi32(_mm256_extractf128_si256(x,0), _mm256_extractf128_si256(t.x,0));
    simd_t<int,4> ret1 = _mm_cmpeq_epi32(_mm256_extractf128_si256(x,1), _mm256_extractf128_si256(t.x,1));
    return simd_t<int,8> (ret0, ret1);
#endif
}


inline simd_t<int,8> simd_t<int,8>::compare_gt(simd_t<int,8> t) const  
{ 
#ifdef __AVX2__
    return _mm256_cmpgt_epi32(x, t.x); 
#else
    simd_t<int,4> ret0 = _mm_cmpgt_epi32(_mm256_extractf128_si256(x,0), _mm256_extractf128_si256(t.x,0));
    simd_t<int,4> ret1 = _mm_cmpgt_epi32(_mm256_extractf128_si256(x,1), _mm256_extractf128_si256(t.x,1));
    return simd_t<int,8> (ret0, ret1);
#endif
}


inline simd_t<int,8> simd_t<int,8>::min(simd_t<int,8> t) const 
{ 
#ifdef __AVX2__
    return _mm256_min_epi32(x, t.x); 
#else
    simd_t<int,4> ret0 = _mm_min_epi32(_mm256_extractf128_si256(x,0), _mm256_extractf128_si256(t.x,0));
    simd_t<int,4> ret1 = _mm_min_epi32(_mm256_extractf128_si256(x,1), _mm256_extractf128_si256(t.x,1));
    return simd_t<int,8> (ret0, ret1);
#endif
}


inline simd_t<int,8> simd_t<int,8>::max(simd_t<int,8> t) const 
{ 
#ifdef __AVX2__
    return _mm256_min_epi32(x, t.x); 
#else
    simd_t<int,4> ret0 = _mm_max_epi32(_mm256_extractf128_si256(x,0), _mm256_extractf128_si256(t.x,0));
    simd_t<int,4> ret1 = _mm_max_epi32(_mm256_extractf128_si256(x,1), _mm256_extractf128_si256(t.x,1));
    return simd_t<int,8> (ret0, ret1);
#endif
}


inline simd_t<int,8> simd_t<int,8>::bitwise_and(simd_t<int,8> t) const
{ 
#ifdef __AVX2__
    return _mm256_and_si256(x, t.x); 
#else
    simd_t<int,4> ret0 = _mm_and_si128(_mm256_extractf128_si256(x,0), _mm256_extractf128_si256(t.x,0));
    simd_t<int,4> ret1 = _mm_and_si128(_mm256_extractf128_si256(x,1), _mm256_extractf128_si256(t.x,1));
    return simd_t<int,8> (ret0, ret1);
#endif
}


inline simd_t<int,8> simd_t<int,8>::bitwise_or(simd_t<int,8> t) const
{ 
#ifdef __AVX2__
    return _mm256_or_si256(x, t.x); 
#else
    simd_t<int,4> ret0 = _mm_or_si128(_mm256_extractf128_si256(x,0), _mm256_extractf128_si256(t.x,0));
    simd_t<int,4> ret1 = _mm_or_si128(_mm256_extractf128_si256(x,1), _mm256_extractf128_si256(t.x,1));
    return simd_t<int,8> (ret0, ret1);
#endif
}


inline simd_t<int,8> simd_t<int,8>::bitwise_xor(simd_t<int,8> t) const
{
#ifdef __AVX2__
    return _mm256_xor_si256(x, t.x); 
#else
    simd_t<int,4> ret0 = _mm_xor_si128(_mm256_extractf128_si256(x,0), _mm256_extractf128_si256(t.x,0));
    simd_t<int,4> ret1 = _mm_xor_si128(_mm256_extractf128_si256(x,1), _mm256_extractf128_si256(t.x,1));
    return simd_t<int,8> (ret0, ret1);
#endif
}


inline simd_t<int,8> simd_t<int,8>::bitwise_andnot(simd_t<int,8> t) const
{
#ifdef __AVX2__
    return _mm256_andnot_si256(x, t.x); 
#else
    simd_t<int,4> ret0 = _mm_andnot_si128(_mm256_extractf128_si256(x,0), _mm256_extractf128_si256(t.x,0));
    simd_t<int,4> ret1 = _mm_andnot_si128(_mm256_extractf128_si256(x,1), _mm256_extractf128_si256(t.x,1));
    return simd_t<int,8> (ret0, ret1);
#endif
}


inline simd_t<int,8> simd_t<int,8>::horizontal_sum() const
{
#ifdef __AVX2__
    __m256i y = x + _mm256_shuffle_epi32(x, 0xb1);  // (2301)_4 = 0xb1
    y += _mm256_shuffle_epi32(y, 0x4e);             // (1032)_4 = 0x4e
    return y + _mm256_permute2f128_si256(y, y, 0x01);
#else
    simd_t<int,4> x0 = _mm256_extractf128_si256(x,0) + _mm256_extractf128_si256(x,1);
    x0 = x0.horizontal_sum();
    return simd_t<int,8> (x0, x0);
#endif
}


inline simd_t<float,4> simd_t<float,4>::horizontal_sum() const
{
    __m128 y = x + _mm_shuffle_ps(x, x, 0xb1);   // (2301)_4 = 0xb1
    return y + _mm_shuffle_ps(y, y, 0x4e);       // (1032)_4 = 0x4e
}


inline float simd_t<float,4>::sum() const
{
    simd_t<float,4> y = this->horizontal_sum();
    return y.extract<0> ();
}


inline simd_t<float,8> simd_t<float,8>::horizontal_sum() const
{
    __m256 y = x + _mm256_shuffle_ps(x, x, 0xb1);   // (2301)_4 = 0xb1
    y += _mm256_shuffle_ps(y, y, 0x4e);             // (1032)_4 = 0x4e
    return y + _mm256_permute2f128_ps(y, y, 0x01);
}


inline float simd_t<float,8>::sum() const
{
    __m256 y = x + _mm256_permute2f128_ps(x, x, 0x01);
    
    simd_t<float,4> z = _mm256_extractf128_ps(y, 0);
    return z.sum();
}


}  // namespace simd_helpers

#endif  // _SIMD_HELPERS_SIMD_T_IMPLEMENTATIONS_HPP
