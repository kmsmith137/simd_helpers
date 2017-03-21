#ifndef _SIMD_HELPERS_FLOAT16_HPP
#define _SIMD_HELPERS_FLOAT16_HPP

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

// Note: float16 is a storage type only, so the only inlines implemented are load/store
// type operations.  We let the pointer type be an arbitrary type (T *), so that the caller
// can use whatever is convenient.


template<bool Aligned=false, typename T>
inline void simd_load_float16(simd_ntuple<float,8,2> &dst, const T *p)
{
#if defined(__F16C__) && defined(__AVX__)
    simd_t<int,8> x = simd_load<int,8,Aligned> ((int *)p);
    simd_t<int,4> x0 = x.template extract_half<0>();
    simd_t<int,4> x1 = x.template extract_half<1>();
    
    dst.template extract<0>() = _mm256_cvtph_ps(x0.x);
    dst.template extract<1>() = _mm256_cvtph_ps(x1.x);
#else
    static_assert(false, "256-bit simd_load_float16() kernel called, AVX and F16C instruction sets are required");
#endif
}


template<bool Aligned=false, bool Streaming=false, typename T>
inline void simd_store_float16(T *p, const simd_ntuple<float,8,2> &src)
{
#if defined(__F16C__) && defined(__AVX__)
    simd_t<float,8> x0 = src.template extract<0> ();
    simd_t<float,8> x1 = src.template extract<1> ();
    simd_t<int,4> y0 = _mm256_cvtps_ph(x0.x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    simd_t<int,4> y1 = _mm256_cvtps_ph(x1.x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    simd_t<int,8> z(y0, y1);
    simd_store<Aligned,Streaming> ((int *)p, z);
#else
    static_assert(false, "256-bit simd_store_float16() kernel called, AVX and F16C instruction sets are required");
#endif
}


}  // namespace simd_helpers

#endif // _SIMD_HELPERS_FLOAT16_HPP
