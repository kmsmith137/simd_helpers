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


inline simd_t<float,4> downsample(const simd_ntuple<float,4,2> &t)
{
    simd_t<float,4> u = _mm_shuffle_ps(t.extract<0>().x, t.extract<1>().x, 0x88);   // [v0 v2 v4 v6],  0x88 = (2020)_4
    simd_t<float,4> v = _mm_shuffle_ps(t.extract<0>().x, t.extract<1>().x, 0xdd);   // [v1 v3 v5 v7],  0xdd = (3131)_4
    return u + v;
}


}  // namespace simd_helpers

#endif  // _SIMD_HELPERS_UDSAMPLE_HPP
