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


template<typename T, int S, typename T2, int S2>
inline simd_t<T,S> simd_cast(simd_t<T2,S2> x);

template<> inline simd_t<int,4> simd_cast(simd_t<float,4> x)  { return _mm_castps_si128(x.x); }
template<> inline simd_t<float,4> simd_cast(simd_t<int,4> x)  { return _mm_castsi128_ps(x.x); }


#ifdef __AVX__

template<> inline simd_t<int,8> simd_cast(simd_t<float,8> x)  { return _mm256_castps_si256(x.x); }
template<> inline simd_t<float,8> simd_cast(simd_t<int,8> x)  { return _mm256_castsi256_ps(x.x); }

template<> inline simd_t<int64_t,4> simd_cast(simd_t<double,4> x)  { return _mm256_castpd_si256(x.x); }
template<> inline simd_t<double,4> simd_cast(simd_t<int64_t,4> x)  { return _mm256_castsi256_pd(x.x); }

#endif


}  // namespace simd_helpers

#endif  // _SIMD_HELPERS_CAST_HPP
