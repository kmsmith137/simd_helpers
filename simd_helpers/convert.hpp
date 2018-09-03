#ifndef _SIMD_HELPERS_CONVERT_HPP
#define _SIMD_HELPERS_CONVERT_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

#include "core.hpp"
#include "simd_int32.hpp"
#include "simd_int64.hpp"
#include "simd_float32.hpp"
#include "simd_float64.hpp"
#include "simd_ntuple.hpp"

// FIXME: this file is incomplete!  It only contains float <-> double conversions, but many more are possible.

namespace simd_helpers {
#if 0
}  // pacify emacs c-mode
#endif


// ------------------------------------------------------------------------------------------------
//
// convert() can take one of three generic forms.

template<typename T, typename T2, int S>
inline void convert(simd_t<T,S> &dst, simd_t<T2,S> src);

template<typename T, typename T2, int S, int N>
inline void convert(simd_ntuple<T,S,N> &dst, simd_t<T2,S*N> src);

template<typename T, typename T2, int S, int N>
inline void convert(simd_t<T,S*N> &dst, simd_ntuple<T2,S,N> src);


// -------------------------------------------------------------------------------------------------
//
// Trivial T->T conversions


template<typename T, int S> inline void convert(simd_t<T,S> &dst, simd_t<T,S> src)         { dst = src; }
template<typename T, int S> inline void convert(simd_ntuple<T,S,1> &dst, simd_t<T,S> src)  { dst.x = src; }
template<typename T, int S> inline void convert(simd_t<T,S> &dst, simd_ntuple<T,S,1> src)  { dst = src.x; }


// ---------------------------------------------------------------------------------------
//
// Float <-> double conversions


#ifdef __AVX__

template<> inline void convert(simd_t<double,4> &dst, simd_t<float,4> src)         
{ 
    dst = _mm256_cvtps_pd(src.x); 
}

template<> inline void convert(simd_ntuple<double,4,2> &dst, simd_t<float,8> src) 
{
    dst.v.x = _mm256_cvtps_pd(_mm256_extractf128_ps(src.x,0));
    dst.x = _mm256_cvtps_pd(_mm256_extractf128_ps(src.x,1));
}

template<> inline void convert(simd_t<float,4> &dst, simd_t<double,4> src)
{
    dst.x = _mm256_cvtpd_ps(src.x);
}

template<> inline void convert(simd_t<float,8> &dst, simd_ntuple<double,4,2> src)
{
    simd_t<float,4> dst0, dst1;
    convert(dst0, src.v.x);
    convert(dst1, src.x);
    dst = simd_t<float,8> (dst0, dst1);
}

#endif  // __AVX__


// -------------------------------------------------------------------------------------------------
//
// Float <-> int32 conversions


template<> inline void convert(simd_t<int,4> &dst, simd_t<float,4> src)         
{ 
    dst = _mm_cvttps_epi32(src.x);
}

template<> inline void convert(simd_t<float,4> &dst, simd_t<int,4> src)
{
    dst = _mm_cvtepi32_ps(src.x);
}


#ifdef __AVX__

template<> inline void convert(simd_t<int,8> &dst, simd_t<float,8> src)         
{ 
    dst = _mm256_cvttps_epi32(src.x);
}

template<> inline void convert(simd_t<float,8> &dst, simd_t<int,8> src)
{
    dst = _mm256_cvtepi32_ps(src.x);
}

#endif


}  // namespace simd_helpers

#endif
