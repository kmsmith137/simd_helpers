#ifndef _SIMD_HELPERS_HPP
#define _SIMD_HELPERS_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

// Note: for API reference, the place to start is simd_helpers/core.hpp !

#include "simd_helpers/core.hpp"
#include "simd_helpers/simd_int32.hpp"       // simd_t<int,S>
#include "simd_helpers/simd_int64.hpp"       // simd_t<int64_t,S>
#include "simd_helpers/simd_float16.hpp"     // 16-bit floating-point (storage type only)
#include "simd_helpers/simd_float32.hpp"     // simd_t<float,S>
#include "simd_helpers/simd_float64.hpp"     // simd_t<double,S>
#include "simd_helpers/simd_ntuple.hpp"      // simd_ntuple<T,S,N>
#include "simd_helpers/simd_trimatrix.hpp"   // simd_trimatrix<T,S,N>
#include "simd_helpers/align.hpp"            // align()
#include "simd_helpers/convert.hpp"          // convert()
#include "simd_helpers/udsample.hpp"         // upsample(), downsample()
#include "simd_helpers/downsample_max.hpp"   // downsample_max()

// Not included by default, but contains useful debug stuff (e.g. print-routines)
// #include "simd_helpers/simd_debug.hpp"

#endif
