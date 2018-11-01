#ifndef _SIMD_HELPERS_LOG_HPP
#define _SIMD_HELPERS_LOG_HPP

#if (__cplusplus < 201103) && !defined(__GXX_LOGERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

#include "core.hpp"
#include "simd_float32.hpp"
#include "simd_float64.hpp"

namespace simd_helpers {
#if 0
}  // pacify emacs c-mode
#endif


// Code in this file is by Erik Schnetter.
// Reference: SLEEF 2.80
//
// FIXME: currently only _simd_log2p_restricted() is implemented, since this suffices
// to implement log_add().  Need general-purpose set of log-type functions.


// _simd_log2_helper(): returns log2(x), as a function of y=(x-1)/(x+1), assuming 1/sqrt(2) < x < sqrt(2).

template<int S>
inline simd_t<float,S> _simd_log2_helper(simd_t<float,S> y)
{
    // Error = 7.09807175879142775648452461821e-8

    simd_t<float,S> y2 = y * y;    
    simd_t<float,S> r = simd_t<float,S> (0.59723611417135718739797302426);
    r = r*y2 + simd_t<float,S> (0.961524413175528426101613434);
    r = r*y2 + simd_t<float,S> (2.88539097665498228703236701);
    
    return r*y;
}


template<int S>
inline simd_t<double,S> _simd_log2_helper(simd_t<double,S> y)
{
    // Error = 2.1410114030383689267772704676e-14

    simd_t<double,S> y2 = y * y;
    simd_t<double,S> r = simd_t<double,S> (0.283751646449323373643963474845);
    r = r*y2 + simd_t<double,S> (0.31983138095551191299118812);
    r = r*y2 + simd_t<double,S> (0.412211603844146279666022);
    r = r*y2 + simd_t<double,S> (0.5770779098948940070516);
    r = r*y2 + simd_t<double,S> (0.961796694295973716912);
    r = r*y2 + simd_t<double,S> (2.885390081777562819196);
    
    return r*y;
}


// Returns log2(1+x), where x is restricted to the range [0,1].
// Intended for use in simd_log2_add().
template<typename T, int S>
inline simd_t<T,S> _simd_log2p_restricted(simd_t<T,S> x)
{
    simd_t<T,S> thresh = 0.41421356237309504880;           // sqrt(2) - 1
    simd_t<T,S> mask = (x >= thresh) & simd_t<T,S>(1.0);   // 1.0 if above threshold, else 0.0
    simd_t<T,S> y = (x - mask) / (x + mask + simd_t<T,S>(2.0));
    
    return _simd_log2_helper(y) + mask;
};


} // namespace simd_helpers

#endif // _SIMD_HELPERS_LOG_HPP
