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

namespace simd_helpers {
#if 0
}  // pacify emacs c-mode
#endif


// ------------------------------------------------------------------------------------------------
//
// convert() can take one of three generic forms, depending on whether the source and destination
// types have the same number of bits.
//
// Note: I haven't tried to exhaustively implement all possible conversions, I've just been
// implementing on an ad hoc basis, as the need for new conversions arises!


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


// -------------------------------------------------------------------------------------------------
//
// Double <-> int64 conversions
//
// There is no SIMD instruction which does this! (pre-AVX512 anyway)
//
// The following amazing code is by Alexander Yee
// (https://stackoverflow.com/questions/41144668/how-to-efficiently-perform-double-int64-conversions-with-sse-avx)
//
// It works up to ~2^52 or so, but silently produces wrong results beyond that.
//
// FIXME: can the silent failure be improved?  If improving it results in extra overhead, we could
// always introduce a "Strict" template argument which defaults to false.


template<> inline void convert(simd_t<int64_t,2> &dst, simd_t<double,2> src)
{
    __m128d t = _mm_set1_pd(0x0018000000000000);
    __m128d x = _mm_add_pd(src.x, t);
    dst = _mm_sub_epi64(_mm_castpd_si128(x), _mm_castpd_si128(t));
}


template<> inline void convert(simd_t<double,2> &dst, simd_t<int64_t,2> src)
{
    __m128d t = _mm_set1_pd(0x0018000000000000);
    __m128i x = _mm_add_epi64(src.x, _mm_castpd_si128(t));
    dst = _mm_sub_pd(_mm_castsi128_pd(x), t);
}

#ifdef __AVX__

template<> inline void convert(simd_t<int64_t,4> &dst, simd_t<double,4> src)
{ 
    __m256d t = _mm256_set1_pd(0x0018000000000000);
    __m256d x = _mm256_add_pd(src.x, t);
    dst.x = _mm256_sub_epi64(_mm256_castpd_si256(x), _mm256_castpd_si256(t));
}

template<> inline void convert(simd_t<double,4> &dst, simd_t<int64_t,4> src)
{
    __m256d t = _mm256_set1_pd(0x0018000000000000);
    __m256i x = _mm256_add_epi64(src.x, _mm256_castpd_si256(t));
    dst = _mm256_sub_pd(_mm256_castsi256_pd(x), t);
}

#endif


}  // namespace simd_helpers

#endif
