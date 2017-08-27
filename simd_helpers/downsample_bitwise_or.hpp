// This header file implements versions of the downsample() routines in which the
// downsampling is done by "bitwise or" instead of "taking sums".
//
// TODO: Currently there are many downsampling-type kernels which are nearly cut-and-paste 
// equivalent.  E.g., downsample(), downsample_bitwise_or(), downsample_bitwise_or().
// Should clean up by using template magic to eliminate redundancy!


#ifndef _SIMD_HELPERS_DOWNSAMPLE_BITWISE_OR_HPP
#define _SIMD_HELPERS_DOWNSAMPLE_BITWISE_OR_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

#include "core.hpp"
#include "simd_int32.hpp"
#include "simd_ntuple.hpp"

namespace simd_helpers {
#if 0
}  // pacify emacs c-mode
#endif


// This file defines:
//
//   template<typename T, int S, int N>
//   inline simd_t<T,S> downsample_bitwise_or(const simd_ntuple<T,S,N> &v)
//
// We omit comments from these kernels since they're cut-and-paste versions
// of the "standard" downsampling kernels in udsample.hpp.


// Trivial downsampling

template<typename T, int S>
inline simd_t<T,S> downsample_bitwise_or(const simd_ntuple<T,S,1> &src) { return src.x; }


// -------------------------------------------------------------------------------------------------
//
// 128-bit


inline __m128i _kernel128_downsample2_bitwise_or(__m128i a, __m128i b)
{
    __m128i u = _mm_xshuffle_epi32(a, b, 0x88);
    __m128i v = _mm_xshuffle_epi32(a, b, 0xdd);
    return _mm_or_si128(u, v);

}

inline __m128i _kernel128_downsample4_bitwise_or(__m128i a, __m128i b, __m128i c, __m128i d)
{
    __m128i u = _kernel128_downsample2_bitwise_or(a, b);
    __m128i v = _kernel128_downsample2_bitwise_or(c, d);
    return _kernel128_downsample2_bitwise_or(u, v);
}

inline simd_t<int,4> downsample_bitwise_or(const simd_ntuple<int,4,2> &t)
{
    return _kernel128_downsample2_bitwise_or(t.extract<0>().x, 
					     t.extract<1>().x);
}


inline simd_t<int,4> downsample_bitwise_or(const simd_ntuple<int,4,4> &t)
{
    return _kernel128_downsample4_bitwise_or(t.extract<0>().x, 
					     t.extract<1>().x, 
					     t.extract<2>().x, 
					     t.extract<3>().x);
}


// -------------------------------------------------------------------------------------------------
//
// 256-bit


#ifdef __AVX__


inline __m256i _kernel256_downsample2_bitwise_or(__m256i a, __m256i b)
{
    __m256i u = _mm256_xshuffle_epi32(b, a, 0x88);
    __m256i v = _mm256_xshuffle_epi32(b, a, 0xdd);
    __m256i w = _mm256_or_si256(u, v);

    __m256i x = _mm256_shuffle_epi32(w, 0x4e);
    __m256i y = _mm256_permute2f128_si256(w, w, 0x01);

    return _mm256_blend_epi32(x, y, 0x3c);
}

inline __m256i _kernel256_downsample4_bitwise_or(__m256i a, __m256i b, __m256i c, __m256i d)
{
    __m256i ac = _mm256_or_si256(_mm256_xshuffle_epi32(a, c, 0x88), _mm256_xshuffle_epi32(a, c, 0xdd));
    __m256i bd = _mm256_or_si256(_mm256_xshuffle_epi32(b, d, 0x88), _mm256_xshuffle_epi32(b, d, 0xdd));

    __m256i u = _mm256_xshuffle_epi32(ac, bd, 0x22);
    __m256i v = _mm256_xshuffle_epi32(ac, bd, 0x77);
    __m256i w = _mm256_or_si256(u, v);

    __m256i x = _mm256_shuffle_epi32(w, 0xb1); 
    __m256i y = _mm256_permute2f128_si256(w, w, 0x01);

    return _mm256_blend_epi32(x, y, 0x5a);
}

inline __m256i _kernel256_downsample8a_bitwise_or(__m256i a, __m256i b, __m256i c, __m256i d)
{
    __m256i ab = _mm256_or_si256(_mm256_xshuffle_epi32(a, b, 0x88), _mm256_xshuffle_epi32(a, b, 0xdd));
    __m256i cd = _mm256_or_si256(_mm256_xshuffle_epi32(c, d, 0x88), _mm256_xshuffle_epi32(c, d, 0xdd));
    return _mm256_or_si256(_mm256_xshuffle_epi32(ab, cd, 0x88), _mm256_xshuffle_epi32(ab, cd, 0xdd));
    
}

inline __m256i _kernel256_downsample8_bitwise_or(__m256i a, __m256i b, __m256i c, __m256i d, __m256i e, __m256i f, __m256i g, __m256i h)
{
    __m256i abcd = _kernel256_downsample8a_bitwise_or(a, b, c, d);
    __m256i efgh = _kernel256_downsample8a_bitwise_or(e, f, g, h);

    __m256i u = _mm256_blend_epi32(abcd, efgh, 0xf0);
    __m256i v = _mm256_permute2f128_si256(abcd, efgh, 0x21);
    return _mm256_or_si256(u, v);
}


inline simd_t<int,8> downsample_bitwise_or(const simd_ntuple<int,8,2> &t)
{
    return _kernel256_downsample2_bitwise_or(t.extract<0>().x, 
					     t.extract<1>().x);
}


inline simd_t<int,8> downsample_bitwise_or(const simd_ntuple<int,8,4> &t)
{
    return _kernel256_downsample4_bitwise_or(t.extract<0>().x, 
					     t.extract<1>().x, 
					     t.extract<2>().x, 
					     t.extract<3>().x);
}


inline simd_t<int,8> downsample_bitwise_or(const simd_ntuple<int,8,8> &t)
{
    return _kernel256_downsample8_bitwise_or(t.extract<0>().x, 
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

#endif  // _SIMD_HELPERS_DOWNSAMPLE_BITWISE_OR_HPP
