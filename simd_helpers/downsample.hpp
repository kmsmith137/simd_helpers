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


// This file contains some "weird boilerplate" designed to provide a streamlined
// interface while working around C++ restrictions on partial template specialization.
// The "_simd_downsampler" class is part of the weird boilerplate!
template<typename T, int S, int D, typename Op> struct _simd_downsampler;


// -------------------------------------------------------------------------------------------------
//
// Downsampling API defined in this file.


template<typename T, int S> struct simd_add {
    static inline simd_t<T,S> op(simd_t<T,S> x, simd_t<T,S> y) { return x+y; }
};


template<typename T, int S, int D, typename Op = simd_add<T,S> >
struct simd_downsampler { 
    template<int N> inline void put(simd_t<T,S> x);
    inline simd_t<T,S> get();

    // Weird boilerplate
    _simd_downsampler<T,S,D,Op> _s;
};


template<typename T, int S, typename Op = simd_add<T,S>, int N>
inline simd_t<T,S> simd_downsample(const simd_ntuple<T,S,N> &x);


// -------------------------------------------------------------------------------------------------
//
// Weird boilerplate


template<int N, typename Td, typename Ts, typename std::enable_if<(N==0),int>::type=0> inline void _simd_ds_put(Td &d, Ts x) { d.put0(x); }
template<int N, typename Td, typename Ts, typename std::enable_if<(N==1),int>::type=0> inline void _simd_ds_put(Td &d, Ts x) { d.put1(x); }
template<int N, typename Td, typename Ts, typename std::enable_if<(N==2),int>::type=0> inline void _simd_ds_put(Td &d, Ts x) { d.put2(x); }
template<int N, typename Td, typename Ts, typename std::enable_if<(N==3),int>::type=0> inline void _simd_ds_put(Td &d, Ts x) { d.put3(x); }
template<int N, typename Td, typename Ts, typename std::enable_if<(N==4),int>::type=0> inline void _simd_ds_put(Td &d, Ts x) { d.put4(x); }
template<int N, typename Td, typename Ts, typename std::enable_if<(N==5),int>::type=0> inline void _simd_ds_put(Td &d, Ts x) { d.put5(x); }
template<int N, typename Td, typename Ts, typename std::enable_if<(N==6),int>::type=0> inline void _simd_ds_put(Td &d, Ts x) { d.put6(x); }
template<int N, typename Td, typename Ts, typename std::enable_if<(N==7),int>::type=0> inline void _simd_ds_put(Td &d, Ts x) { d.put7(x); }

template<typename T, int S, int D, typename Op> 
template<int N>
inline void simd_downsampler<T,S,D,Op>::put(simd_t<T,S> x) 
{ 
    _simd_ds_put<N> (_s, x); 
}

template<typename T, int S, int D, typename Op> 
inline simd_t<T,S> simd_downsampler<T,S,D,Op>::get()
{
    return _s.get();
}


template<typename Td, typename T, int S, int N, typename std::enable_if<(N==0),int>::type=0>
inline void _simd_ds_mput(Td &d, const simd_ntuple<T,S,N> &x) { }

template<typename Td, typename T, int S, int N, typename std::enable_if<(N>0),int>::type=0>
inline void _simd_ds_mput(Td &d, const simd_ntuple<T,S,N> &x) 
{ 
    _simd_ds_mput(d, x.v);
    _simd_ds_put<N-1> (d, x.x);
}

template<typename T, int S, typename Op, int N>
inline simd_t<T,S> simd_downsample(const simd_ntuple<T,S,N> &x)
{
    _simd_downsampler<T,S,N,Op> ds;
    _simd_ds_mput(ds, x);
    return ds.get();
}

// -------------------------------------------------------------------------------------------------


template<typename Op>
struct _simd_downsampler<float,8,2,Op>
{
    simd_t<float,8> a;

    inline void put0(simd_t<float,8> a_) { a = a_; }

    inline void put1(simd_t<float,8> b) 
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

    inline simd_t<float,8> get() { return a; }
};


}  // namespace simd_helpers

#endif  // _SIMD_HELPERS_DOWNSAMPLE_HPP
