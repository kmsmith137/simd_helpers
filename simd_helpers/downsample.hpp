#ifndef _SIMD_HELPERS_DOWNSAMPLE_HPP
#define _SIMD_HELPERS_DOWNSAMPLE_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

#include "core.hpp"
#include "simd_int32.hpp"
#include "simd_float32.hpp"
#include "simd_ntuple.hpp"

namespace simd_helpers {
#if 0
}  // pacify emacs c-mode
#endif


// -------------------------------------------------------------------------------------------------
//
// Binary operations used in downsampling kernels.


template<typename T, int S> struct simd_add {
    static inline simd_t<T,S> op(simd_t<T,S> x, simd_t<T,S> y) { return x+y; }
};


// -------------------------------------------------------------------------------------------------
//
// simd_downsampler.

template<typename T, int S, int D, typename Op = simd_add<T,S> >
struct simd_downsampler { 
    inline simd_t<T,S> get();

    template<int N>
    inline void put(simd_t<T,S> x);
};


template<typename Tds, typename T, int S, int N, typename std::enable_if<(N==0),int>::type = 0>
inline void _simd_put(Tds &ds, const simd_ntuple<T,S,N> &x) { }

template<typename Tds, typename T, int S, int N, typename std::enable_if<(N>0),int>::type = 0>
inline void _simd_put(Tds &ds, const simd_ntuple<T,S,N> &x)
{
    _simd_put(ds, x.v);
    ds.put<N-1> (x);
}


template<typename T, int S, int N>
inline simd_t<T,S> simd_downsample(const simd_ntuple<T,S,N> &x)
{
    simd_downsampler<T,S,N> ds;
    _simd_put(ds, x);
    return ds.get();
}


// -------------------------------------------------------------------------------------------------


template<>
struct simd_downsampler<float,8,2,simd_add<float,8> > {
    simd_t<float,8> a;

    inline simd_t<float,8> get() { return a; }
    
    template<int N>
    inline void put(simd_t<float,8> x);
};


template<>
inline void simd_downsampler<float,8,2,simd_add<float,8> >::put<0> (simd_t<float,8> a_)
{
    a = a_;
}

#if 0
template<typename Op>
inline void simd_downsampler<float,8,2,Op>::put<1> (simd_t<float,8> b)
{
    // Similar to _kernel128_downsample2(), but with an extra 64-bit swap
    // which in theory helps the subsequent permute/blend block pipeline.
    // (Not sure if this actually helps!)

    simd_t<float,8> u = _mm256_shuffle_ps(b.x, a.x, 0x88);   // [b0 b2 a0 a2],  0x88 = (2020)_4
    simd_t<float,8> v = _mm256_shuffle_ps(b.x, a.x, 0xdd);   // [b1 b3 a1 a3],  0xdd = (3131)_4
    simd_t<float,8> w = Op::op(u,v);

    // Now we have a vector [ w2 w0 w3 w1 ], where w_i is 64 bits
    // and we want to rearrange to [ w0 w1 w2 w3 ].

    __m256 x = _mm256_permute_ps(w.x, 0x4e);          // [ w0 w2 w1 w3 ],  0x4e = (1032)_4
    __m256 y = _mm256_permute2f128_ps(w.x, w.x, 0x01);  // [ w3 w1 w2 w0 ]

    a = _mm256_blend_ps(x, y, 0x3c);   // (00111100)_2
}
#endif


}  // namespace simd_helpers

#endif  // _SIMD_HELPERS_DOWNSAMPLE_HPP
