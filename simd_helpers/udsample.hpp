#ifndef _SIMD_HELPERS_UDSAMPLE_HPP
#define _SIMD_HELPERS_UDSAMPLE_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

#include "simd_t.hpp"
#include "simd_ntuple.hpp"

namespace simd_helpers {
#if 0
}  // pacify emacs c-mode
#endif


// -------------------------------------------------------------------------------------------------
//
// template<typename T, unsigned int S, unsigned int N>
// inline simd_t<T,S> downsample(const simd_ntuple<T,S,N> &v)
//
// Note: the downsampling kernel does not divide the result by N !!


inline __m128 _kernel128_downsample2(__m128 a, __m128 b)
{
    __m128 u = _mm_shuffle_ps(a, b, 0x88);   // [v0 v2 v4 v6],  0x88 = (2020)_4
    __m128 v = _mm_shuffle_ps(a, b, 0xdd);   // [v1 v3 v5 v7],  0xdd = (3131)_4
    return u + v;

}

inline __m128 _kernel128_downsample4(__m128 a, __m128 b, __m128 c, __m128 d)
{
    // I think this is fastest.
    __m128 u = _kernel128_downsample2(a, b);
    __m128 v = _kernel128_downsample2(c, d);
    return _kernel128_downsample2(u, v);
}


inline simd_t<float,4> downsample(const simd_ntuple<float,4,2> &t)
{
    return _kernel128_downsample2(t.extract<0>().x, t.extract<1>().x);
}

inline simd_t<float,4> downsample(const simd_ntuple<float,4,4> &t)
{
    return _kernel128_downsample4(t.extract<0>().x, t.extract<1>().x, t.extract<2>().x, t.extract<3>().x);
}


}  // namespace simd_helpers

#endif  // _SIMD_HELPERS_UDSAMPLE_HPP
