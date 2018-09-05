#ifndef _SIMD_HELPERS_EXP_HPP
#define _SIMD_HELPERS_EXP_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

#include "core.hpp"
#include "simd_float32.hpp"
#include "simd_float64.hpp"
#include "cast.hpp"
#include "convert.hpp"

namespace simd_helpers {
#if 0
}  // pacify emacs c-mode
#endif


// Code in this file is by Erik Schnetter.
// Reference: SLEEF 2.80
//
// FIXME: implement underflow/overflow simd_exp2()
// FIXME: implement simd_exp().  (Currently only simd_exp2 is implemented.)


// _simd_exp2_restricted(): returns 2^x, assuming x is restricted to [-0.5, 0.5]

template<int S>
inline simd_t<float,S> _simd_exp2_restricted(simd_t<float,S> x)
{
    // Error = 1.62772721960621336664735896836e-7

    simd_t<float,S> r = 0.00133952915439234389712105060319;
    r = r*x + simd_t<float,S> (0.009670773148229417605024318985);
    r = r*x + simd_t<float,S> (0.055503406540531310853149866446);
    r = r*x + simd_t<float,S> (0.240222115700585316818177639177);
    r = r*x + simd_t<float,S> (0.69314720007380208630542805293);
    r = r*x + simd_t<float,S> (1.00000005230745711373079206024);
    
    return r;
}


template<int S>
inline simd_t<double,S> _simd_exp2_restricted(simd_t<double,S> x)
{
    // Error = 3.74939899823302048807873981077e-14

    simd_t<double,S> r = 1.02072375599725694063203809188e-7;
    r = r*x + simd_t<double,S> (1.32573274434801314145133004073e-6);
    r = r*x + simd_t<double,S> (0.0000152526647170731944840736190013);
    r = r*x + simd_t<double,S> (0.000154034441925859828261898614555);
    r = r*x + simd_t<double,S> (0.00133335582175770747495287552557);
    r = r*x + simd_t<double,S> (0.0096181291794939392517233403183);
    r = r*x + simd_t<double,S> (0.055504108664525029438908798685);
    r = r*x + simd_t<double,S> (0.240226506957026959772247598695);
    r = r*x + simd_t<double,S> (0.6931471805599487321347668143);
    r = r*x + simd_t<double,S> (1.00000000000000942892870993489);
    
    return r;
}


// simd_exp2_unsafe(): returns 2^x, but without underflow/overflow checks!
// (i.e. will return nonsense result if underflow/overflow)

template<typename T, int S>
inline simd_t<T,S> simd_exp2_unsafe(simd_t<T,S> x)
{
    using I = typename simd_t<T,S>::iscalar_type;
    constexpr int mantissa_bits = std::numeric_limits<T>::digits - 1;

    simd_t<T,S> rx = x.round();
    simd_t<I,S> ix;
    convert(ix, rx);
    
    simd_t<T,S> r = _simd_exp2_restricted(x - rx);
    simd_t<I,S> ir = simd_cast<I,S> (r);
    ir += (ix << mantissa_bits);
    
    return simd_cast<T,S> (ir);
}


} // namespace simd_helpers

#endif // _SIMD_HELPERS_EXP_HPP
