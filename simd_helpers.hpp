#ifndef _SIMD_HELPERS_HPP
#define _SIMD_HELPERS_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

// Note: for a "declaration" of the key class simd_t<T,S>, 
// see the extended comment in simd_helpers/base.hpp

#include "simd_helpers/base.hpp"
#include "simd_helpers/simd_int32.hpp"       // simd_t<int,S>
#include "simd_helpers/simd_int64.hpp"       // simd_t<int64_t,S>
#include "simd_helpers/simd_float32.hpp"     // simd_t<float,S>
#include "simd_helpers/simd_ntuple.hpp"      // simd_ntuple<T,S,N>
#include "simd_helpers/simd_trimatrix.hpp"   // simd_trimatrix<T,S,N>
#include "simd_helpers/udsample.hpp"         // upsample(), downsample()

// #include "simd_helpers/simd_debug.hpp"    // debug stuff not included by default

#endif
