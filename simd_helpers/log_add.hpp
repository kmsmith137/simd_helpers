#ifndef _SIMD_HELPERS_LOG_ADD_HPP
#define _SIMD_HELPERS_LOG_ADD_HPP

#if (__cplusplus < 201103) && !defined(__GXX_LOG_ADDERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

#include "core.hpp"
#include "simd_float32.hpp"
#include "simd_float64.hpp"
#include "exp.hpp"
#include "log.hpp"

namespace simd_helpers {
#if 0
}  // pacify emacs c-mode
#endif


// log2_add: this is the binary operation (x,y) -> log2(2^x + 2^y).
// ln_add: this is the binary operation (x,y) -> ln(e^x + e^y).

template<typename T, int S>
inline simd_t<T,S> simd_log2_add(simd_t<T,S> x, simd_t<T,S> y)
{
    simd_t<T,S> u = x.max(y);
    simd_t<T,S> v = x.min(y);
    simd_t<T,S> t = v - u;

    // The max(..., -80) takes care of underflow/overflow issues here.
    t = t.max(simd_t<T,S> (-80.0));
    t = simd_exp2_unsafe(t);
    t = _simd_log2p_restricted(t);

    return t + u;
}


template<typename T, int S>
inline simd_t<T,S> simd_ln_add(simd_t<T,S> x, simd_t<T,S> y)
{
    const simd_t<T,S> log2 = 0.69314718055994530942;
    const simd_t<T,S> rlog2 = 1.4426950408889634074;
    
    return log2 * simd_log2_add(rlog2 * x, rlog2 * y);
}


} // namespace simd_helpers

#endif // _SIMD_HELPERS_LOG_ADD_HPP
