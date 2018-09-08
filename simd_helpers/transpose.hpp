#ifndef _SIMD_HELPERS_TRANSPOSE_HPP
#define _SIMD_HELPERS_TRANSPOSE_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

#include "core.hpp"
#include "simd_float32.hpp"
#include "simd_ntuple.hpp"

namespace simd_helpers {
#if 0
}  // pacify emacs c-mode
#endif

// We define two SIMD transpose operations, "transpose" and "btranspose" (short for
// "block transpose").  They have three parameters: a type T, simd length S, and 
// rank N (= number of simd registers participating in the transpose).
//
// If N < S, then the transpose has spectator indices.  In the "transpose" kernels,
// the spectator indices are slowly varying, whereas in the "btranspose" kernels, the
// spectator indices are rapidly varying.
//
// For example, if S=8 and N=2, then the transpose kernel looks like this:
//
//   x_in = [ x0 x1 x2 x3 x4 x5 x6 x7 ]
//   y_in = [ y0 y1 y2 y3 y4 y5 y6 y7 ]
//
//   x_out = [ x0 y0 x2 y2 x4 y4 x6 y6 ]
//   y_out = [ x1 y1 x3 y3 x5 y5 x7 y7 ]
//
// whereas the btranspose kernel looks like this:
//
//   x_out = [ x0 x1 x2 x3 y0 y1 y2 y3 ]
//   y_out = [ x4 x5 x6 x7 y4 y5 y6 y7 ]
//
// We define three interfaces to the transpose kernels as follows (for (T,S,N)=(float,8,2)):
// 
//    // low-level kernel with __m256 args
//    _transpose2_ps256(__m256 x, __m256 y)
//
//    // simd_t<T,S> wrapper 
//    simd_transpose(simd_t<float,8> x, simd_t<float,8> y)
//
//    // simd_ntuple<T,S,N> wrapper
//    simd_transpose(simd_t<float,8,2> x)
//
// FIXME: only a few (T,S,N) combinations are currently implemented!
// FIXME: assumes AVX2!


#ifdef __AVX2__


inline void _btranspose2_ps256(__m256 &x, __m256 &y)
{
#if 0
    // Slightly slower.
    __m256 out_x = _mm256_permute2f128_ps(x, y, 0x20);  // [ x0 y0 ]
    __m256 out_y = _mm256_permute2f128_ps(x, y, 0x31);  // [ x1 y1 ]
    x = out_x;
    y = out_y;
#else
    // Slightly faster.
    __m256 z = _mm256_permute2f128_ps(x, y, 0x21);  // [ x1 y0 ]
    x = _mm256_blend_ps(x, z, 0xf0);  // [ x0 y0 ]
    y = _mm256_blend_ps(y, z, 0x0f);  // [ x1 y1 ]
#endif    
}


inline void _transpose4_ps256(__m256 &a, __m256 &b, __m256 &c, __m256 &d)
{
    __m256 w = _mm256_shuffle_ps(a, c, 0x44);  // (1010)_4 -> [ a0 a1 c0 c1 ]
    __m256 x = _mm256_shuffle_ps(b, d, 0x11);  // (0101)_4 -> [ b1 b0 d1 d0 ]
    __m256 y = _mm256_shuffle_ps(a, c, 0xee);  // (3232)_4 -> [ a2 a3 c2 c3 ]
    __m256 z = _mm256_shuffle_ps(b, d, 0xbb);  // (2323)_4 -> [ b3 b2 d3 d2 ]

    a = _mm256_blend_ps(w, x, 0xaa);  // (10101010)_2 -> [ a0 b0 c0 d0 ]
    b = _mm256_blend_ps(w, x, 0x55);  // (01010101)_2 -> [ b1 a1 d1 c1 ]
    b = _mm256_permute_ps(b, 0xb1);   // (2301)_4 -> [ a1 b1 c1 d1 ]

    c = _mm256_blend_ps(y, z, 0xaa);  // (10101010)_2 -> [ a2 b2 c2 d2 ]
    d = _mm256_blend_ps(y, z, 0x55);  // (01010101)_2 -> [ b3 a3 d3 c3 ]
    d = _mm256_permute_ps(d, 0xb1);   // (2301)_4 -> [ a3 b3 c3 d3 ]
}


inline void _transpose8_ps256(__m256 &a, __m256 &b, __m256 &c, __m256 &d, __m256 &e, __m256 &f, __m256 &g, __m256 &h)
{
    _btranspose2_ps256(a, e);
    _btranspose2_ps256(b, f);
    _btranspose2_ps256(c, g);
    _btranspose2_ps256(d, h);

    _transpose4_ps256(a, b, c, d);
    _transpose4_ps256(e, f, g, h);
}


inline void simd_btranspose(simd_t<float,8> &a, simd_t<float,8> &b)
{
    _btranspose2_ps256(a.x, b.x);
}

inline void simd_transpose(simd_t<float,8> &a, simd_t<float,8> &b, simd_t<float,8> &c, simd_t<float,8> &d)
{
    _transpose4_ps256(a.x, b.x, c.x, d.x);
}

inline void simd_transpose(simd_t<float,8> &a, simd_t<float,8> &b, simd_t<float,8> &c, simd_t<float,8> &d, simd_t<float,8> &e, simd_t<float,8> &f, simd_t<float,8> &g, simd_t<float,8> &h)
{
    _transpose8_ps256(a.x, b.x, c.x, d.x, e.x, f.x, g.x, h.x);
}


#endif  // __AVX2__


// -------------------------------------------------------------------------------------------------
//
// The boilerplate below arranges things so that
//
//   simd_transpose(simd_ntuple<T,S,N> &x)
//   simd_btranspose(simd_ntuple<T,S,N> &x)
//
// are aliases for
//
//   simd_transpose(x.extract<0>(), ..., x.extract<N-1>())
//   simd_btranspose(x.extract<0>(), ..., x.extract<N-1>())


template<typename T, int S, int N, typename... Args, typename std::enable_if<(N==0),int>::type = 0>
inline void _simd_transpose(simd_ntuple<T,S,N> &x, Args& ... a)
{ 
    simd_transpose(a...);
}

template<typename T, int S, int N, typename... Args, typename std::enable_if<(N>0),int>::type = 0>
inline void _simd_transpose(simd_ntuple<T,S,N> &x, Args& ... a)
{
    _simd_transpose(x.v, x.x, a...);
}

template<typename T, int S, int N, typename... Args, typename std::enable_if<(N==0),int>::type = 0>
inline void _simd_btranspose(simd_ntuple<T,S,N> &x, Args& ... a)
{
    simd_btranspose(a...);
}

template<typename T, int S, int N, typename... Args, typename std::enable_if<(N>0),int>::type = 0>
inline void _simd_btranspose(simd_ntuple<T,S,N> &x, Args& ... a)
{
    _simd_btranspose(x.v, x.x, a...);
}


template<typename T, int S, int N>
inline void simd_transpose(simd_ntuple<T,S,N> &x)
{
    _simd_transpose(x);
}

template<typename T, int S, int N>
inline void simd_btranspose(simd_ntuple<T,S,N> &x)
{
    _simd_btranspose(x);
}


} // namespace simd_helpers

#endif // _SIMD_HELPERS_TRANSPOSE_HPP
