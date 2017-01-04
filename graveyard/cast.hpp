#ifndef _SIMD_HELPERS_CAST_HPP
#define _SIMD_HELPERS_CAST_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

#include "core.hpp"
#include "simd_int32.hpp"
#include "simd_int64.hpp"
#include "simd_float32.hpp"
#include "simd_float64.hpp"


namespace simd_helpers {
#if 0
}  // pacify emacs c-mode
#endif


template<typename T, typename T2, unsigned int S>
inline void cast(simd_t<T,S> &dst, simd_t<T2,S> src);


template<typename T, unsigned int S> 
inline void cast(simd_t<T,S> &dst, simd_t<T,S> src) 
{ 
    dst = src; 
}


template<> inline void cast(simd_t<float,4> &dst, simd_t<int,4> src)  { dst.x = _mm_castsi128_ps(x.x); }
template<> inline void cast(simd_t<float,8> &dst, simd_t<int,8> src)  { dst.x = _mm256_castsi256_ps(x.x); }

template<> inline void cast(simd_t<int,4> &dst, simd_t<float,4> src)  { dst.x = _mm_castps_si128(x.x); }
template<> inline void cast(simd_t<int,8> &dst, simd_t<float,8> src)  { dst.x = _mm256_castps_si256(x.x); }

template<> inline void cast(simd_t<double,2> &dst, simd_t<int64_t,2> src)  { dst.x = _mm_castsi128_pd(x.x); }
template<> inline void cast(simd_t<double,4> &dst, simd_t<int64_t,4> src)  { dst.x = _mm256_castsi256_pd(x.x); }

template<> inline void cast(simd_t<int64_t,2> &dst, simd_t<double,2> src)  { dst.x = _mm_castpd_si128(x.x); }
template<> inline void cast(simd_t<int64_t,4> &dst, simd_t<double,4> src)  { dst.x = _mm256_castpd_si256(x.x); }


}  // namespace simd_helpers

#endif
