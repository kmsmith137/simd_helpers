// This header file implements versions of the downsample() routines in which the
// downsampling is done by "taking maximums" instead of "taking sums".
//
// TODO: Currently there are many downsampling-type kernels which are nearly cut-and-paste 
// equivalent.  E.g., downsample(), downsample_max(), downsample_bitwise_or().
// Should clean up by using template magic to eliminate redundancy!


#ifndef _SIMD_HELPERS_DOWNSAMPLE_MAX_HPP
#define _SIMD_HELPERS_DOWNSAMPLE_MAX_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

#include "core.hpp"
#include "simd_float32.hpp"

namespace simd_helpers {
#if 0
}  // pacify emacs c-mode
#endif


// This file defines:
//
//   template<typename T, unsigned int S, unsigned int N>
//   inline simd_t<T,S> downsample_max(const simd_ntuple<T,S,N> &v)
//
// We omit comments from these kernels since they're cut-and-paste versions
// of the "standard" downsampling kernels in udsample.hpp.


// Trivial downsampling

template<typename T, unsigned int S>
inline simd_t<T,S> downsample_max(const simd_ntuple<T,S,1> &src) { return src.x; }


// -------------------------------------------------------------------------------------------------
//
// 128-bit


inline __m128 _kernel128_downsample2_max(__m128 a, __m128 b)
{
    __m128 u = _mm_shuffle_ps(a, b, 0x88);
    __m128 v = _mm_shuffle_ps(a, b, 0xdd);
    return _mm_max_ps(u, v);

}

inline __m128 _kernel128_downsample4_max(__m128 a, __m128 b, __m128 c, __m128 d)
{
    __m128 u = _kernel128_downsample2_max(a, b);
    __m128 v = _kernel128_downsample2_max(c, d);
    return _kernel128_downsample2_max(u, v);
}

inline simd_t<float,4> downsample_max(const simd_ntuple<float,4,2> &t)
{
    return _kernel128_downsample2_max(t.extract<0>().x, 
				      t.extract<1>().x);
}


inline simd_t<float,4> downsample_max(const simd_ntuple<float,4,4> &t)
{
    return _kernel128_downsample4_max(t.extract<0>().x, 
				      t.extract<1>().x, 
				      t.extract<2>().x, 
				      t.extract<3>().x);
}


// -------------------------------------------------------------------------------------------------
//
// 256-bit


#ifdef __AVX__


inline __m256 _kernel256_downsample2_max(__m256 a, __m256 b)
{
    __m256 u = _mm256_shuffle_ps(b, a, 0x88);
    __m256 v = _mm256_shuffle_ps(b, a, 0xdd);
    __m256 w = _mm256_max_ps(u, v);

    __m256 x = _mm256_permute_ps(w, 0x4e);
    __m256 y = _mm256_permute2f128_ps(w, w, 0x01);

    return _mm256_blend_ps(x, y, 0x3c);
}

inline __m256 _kernel256_downsample4_max(__m256 a, __m256 b, __m256 c, __m256 d)
{
    __m256 ac = _mm256_max_ps(_mm256_shuffle_ps(a, c, 0x88), _mm256_shuffle_ps(a, c, 0xdd));
    __m256 bd = _mm256_max_ps(_mm256_shuffle_ps(b, d, 0x88), _mm256_shuffle_ps(b, d, 0xdd));

    __m256 u = _mm256_shuffle_ps(ac, bd, 0x22);
    __m256 v = _mm256_shuffle_ps(ac, bd, 0x77);
    __m256 w = _mm256_max_ps(u, v);

    __m256 x = _mm256_permute_ps(w, 0xb1); 
    __m256 y = _mm256_permute2f128_ps(w, w, 0x01);

    return _mm256_blend_ps(x, y, 0x5a);
}

inline __m256 _kernel256_downsample8a_max(__m256 a, __m256 b, __m256 c, __m256 d)
{
    __m256 ab = _mm256_max_ps(_mm256_shuffle_ps(a, b, 0x88), _mm256_shuffle_ps(a, b, 0xdd));
    __m256 cd = _mm256_max_ps(_mm256_shuffle_ps(c, d, 0x88), _mm256_shuffle_ps(c, d, 0xdd));
    return _mm256_max_ps(_mm256_shuffle_ps(ab, cd, 0x88), _mm256_shuffle_ps(ab, cd, 0xdd));
    
}

inline __m256 _kernel256_downsample8_max(__m256 a, __m256 b, __m256 c, __m256 d, __m256 e, __m256 f, __m256 g, __m256 h)
{
    __m256 abcd = _kernel256_downsample8a_max(a, b, c, d);
    __m256 efgh = _kernel256_downsample8a_max(e, f, g, h);

    __m256 u = _mm256_blend_ps(abcd, efgh, 0xf0);
    __m256 v = _mm256_permute2f128_ps(abcd, efgh, 0x21);
    return _mm256_max_ps(u, v);
}


inline simd_t<float,8> downsample_max(const simd_ntuple<float,8,2> &t)
{
    return _kernel256_downsample2_max(t.extract<0>().x, 
				      t.extract<1>().x);
}


inline simd_t<float,8> downsample_max(const simd_ntuple<float,8,4> &t)
{
    return _kernel256_downsample4_max(t.extract<0>().x, 
				      t.extract<1>().x, 
				      t.extract<2>().x, 
				      t.extract<3>().x);
}


inline simd_t<float,8> downsample_max(const simd_ntuple<float,8,8> &t)
{
    return _kernel256_downsample8_max(t.extract<0>().x, 
				      t.extract<1>().x, 
				      t.extract<2>().x, 
				      t.extract<3>().x,
				      t.extract<4>().x, 
				      t.extract<5>().x, 
				      t.extract<6>().x, 
				      t.extract<7>().x);
}


#endif  // __AVX__


}  // namespace simd_helpers

#endif  // _SIMD_HELPERS_DOWNSAMPLE_MAX_HPP
