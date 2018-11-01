#ifndef _SIMD_HELPERS_MEDIAN_HPP
#define _SIMD_HELPERS_MEDIAN_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

#include <type_traits>
#include "sort.hpp"

namespace simd_helpers {
#if 0
}  // pacify emacs c-mode
#endif


// The median API is really simple: we define inline functions
//
//  simd_t<T,S> simd_median(simd_ntuple<T,S,N> x);
//  simd_t<T,S> simd_median_presorted(const simd_ntuple<T,S,N> &x);
//
// FIXME (easy): unimplemented in the case where T is an integer type, and N is even.
//
// FIXME (super hard): we currently implement simd_median() by running a full sorting 
// network.  The compiler should be able to optimize out a few "dead" min/max 
// operations, but this is still suboptimal.  The design of optimal "median circuits"
// doesn't seem to have gotten much attention in the literature, but here is one
// nice reference from 2004: 
//
//   Lukas Sekanina, "Evolutionary Design Space Exploration for Median Circuits"
//   https://pdfs.semanticscholar.org/c76e/f5ccce202cfbc998399031d71196d7abffa3.pdf
//
// Based on Table 1 of this paper, the current simd_median() implementation is
// suboptimal by a factor ~1.5.  Given the amount of work that would be needed to
// improve this, it will probably never happen!  (Although, optimizing a specific
// case, e.g. N=5,7,9 would not be a crazy undertaking.)


template<typename T, int S, int N, typename std::enable_if<(N % 2 == 1),int>::type = 0>
inline simd_t<T,S> simd_median_presorted(const simd_ntuple<T,S,N> &x)
{
    return x.template extract<(N-1)/2> ();
}

template<typename T, int S, int N, typename std::enable_if<(N % 2 == 0),int>::type = 0>
inline simd_t<T,S> simd_median_presorted(const simd_ntuple<T,S,N> &x)
{
    static_assert(std::is_floating_point<T>::value, "simd_median() is unimplemented in the case where T is an integer type, and N is even");

    simd_t<T,S> a = x.template extract<(N/2)-1> ();
    simd_t<T,S> b = x.template extract<(N/2)> ();
    return simd_t<T,S>(0.5) * (a+b);
}


template<typename T, int S, int N>
inline simd_t<T,S> simd_median(simd_ntuple<T,S,N> x)
{
    simd_sort(x);
    return simd_median_presorted(x);
}


}  // namespace simd_helpers

#endif  // _SIMD_HELPERS_MEDIAN_HPP
