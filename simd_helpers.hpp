#ifndef _SIMD_HELPERS_HPP
#define _SIMD_HELPERS_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

#include "simd_helpers/simd_t.hpp"
#include "simd_helpers/simd_ntuple.hpp"
#include "simd_helpers/simd_trimatrix.hpp"
#include "simd_helpers/udsample.hpp"

// not included by default
// #include "simd_helpers/simd_debug.hpp"

#endif
