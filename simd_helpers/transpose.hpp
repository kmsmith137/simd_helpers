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


// The transpose API is:
//
//   simd_transpose<T,S,N,M=1> (simd_ntuple<T,S,N> &x);
//   simd_transpose<M=1> (simd_t<T,S> &x1, simd_t<T,S> &x2, ..., simd_t<T,S> &xN);
//
// Each transpose kernel is labeled by its "size" N and "multiplicity" M.
// The size N is the number of simd_t's which participate in the transpose.
// The multiplicity M determines which bits inside the simd_t participate in the transpose.
//
// Formally, the simd_ntuple<T,S,N> is viewed as a shape (N,M,N,S/(MN))-array, and the
// two length-N axes are exchanged.  To write this out in detail, if N=2, then there
// are three transpose kernels with M=1,2,4:
//
//   x_in = [ x0 x1 x2 x3 x4 x5 x6 x7 ]
//   y_in = [ y0 y1 y2 y3 y4 y5 y6 y7 ]
//
//   x_out_M1 = [ x0 x1 x2 x3 y0 y1 y2 y3 ]
//   y_out_M1 = [ x4 x5 x6 x7 y4 y5 y6 y7 ]
//
//   x_out_M2 = [ x0 x1 y0 y1 x4 x5 y4 y5 ]
//   y_out_M2 = [ x2 x3 y2 y3 x6 x7 y6 y7 ]
//
//   x_out_M4 = [ x0 y0 x2 y2 x4 y4 x6 y6 ]
//   y_out_M4 = [ x1 y1 x3 y3 x5 y5 x7 y7 ]
//
// The multiplicity M is a power of 2, satisfying 1 <= M <= (S/N).
// (In particular, if N=S, then M=1 is the only possibility.)
//
// FIXME: could use static_asserts to check for bad (N,M) combinations (need to write some constexpr inline functions first)
// FIXME: only a few (T,S,N) combinations are currently implemented!
// FIXME: assumes AVX2!


#ifdef __AVX2__


// -------------------------------------------------------------------------------------------------
//
// These versions of the kernels can be called directly on the low-level simd types (e.g. __m256)


// Size 2, multiplicity 1.
inline void _transpose2_m1_ps256(__m256 &x, __m256 &y)
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

// Size 2, multiplicity 2.
inline void _transpose2_m2_ps256(__m256 &a, __m256 &b)
{
    __m256 anew = _mm256_shuffle_ps(a, b, 0x44);  // (1010)_4
    __m256 bnew = _mm256_shuffle_ps(a, b, 0xee);  // (3232)_4

    a = anew;
    b = bnew;
}

// Size 2, multiplicity 4.
inline void _transpose2_m4_ps256(__m256 &a, __m256 &b)
{
    __m256 ta = _mm256_permute_ps(a, 0xb1);  // (2301)_4 -> [ a1 a0 a3 a2 a5 a4 a7 a6 ]
    __m256 tb = _mm256_permute_ps(b, 0xb1);  // (2301)_4 -> [ b1 b0 b3 b2 b5 b4 b7 b6 ]

    a = _mm256_blend_ps(a, tb, 0xaa);  // (10101010)_2 -> [ a0 b0 a2 b2 a4 b4 a6 b6 ]
    b = _mm256_blend_ps(ta, b, 0xaa);  // (10101010)_2 -> [ a1 b1 a3 b3 a5 b5 a7 b7 ]
}

// Size 4, multiplicity 1.
inline void _transpose4_m1_ps256(__m256 &a, __m256 &b, __m256 &c, __m256 &d)
{
    // Input
    //   a = [ a0 a1 a2 a3 ]
    //   b = [ b0 b1 b2 b3 ]
    //   c = [ c0 c1 c2 c3 ]
    //   d = [ d0 d1 d2 d3 ]
    
    _transpose2_m2_ps256(a, b);
    _transpose2_m2_ps256(c, d);

    // After _mtranspose2_256()
    //   a = [ a0 b0 a2 b2 ]
    //   b = [ a1 b1 a3 b2 ]
    //   c = [ c0 d0 c2 d2 ]
    //   d = [ c1 d1 c3 d3 ]

    _transpose2_m1_ps256(a, c);
    _transpose2_m1_ps256(b, d);
    
    // After _btranspose2_ps256()
    //   a = [ a0 b0 c0 d0 ]
    //   b = [ a1 b1 c1 d1 ]
    //   c = [ a2 b2 c2 d2 ]
    //   d = [ a3 b3 c3 d3 ]
}

// Size 4, multiplicity 2.
inline void _transpose4_m2_ps256(__m256 &a, __m256 &b, __m256 &c, __m256 &d)
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

// Size 8, multiplicity 1.
inline void _transpose8_m1_ps256(__m256 &a, __m256 &b, __m256 &c, __m256 &d, __m256 &e, __m256 &f, __m256 &g, __m256 &h)
{
    _transpose2_m1_ps256(a, e);
    _transpose2_m1_ps256(b, f);
    _transpose2_m1_ps256(c, g);
    _transpose2_m1_ps256(d, h);

    _transpose4_m2_ps256(a, b, c, d);
    _transpose4_m2_ps256(e, f, g, h);
}


// -------------------------------------------------------------------------------------------------
//
// First simd_helpers transpose API:
//
//   simd_transpose<M=1> (simd_t<T,S> &x1, simd_t<T,S> &x2, ..., simd_t<T,S> &xN);


template<int M=1, typename std::enable_if<(M==1),int>::type = 0>
inline void simd_transpose(simd_t<float,8> &a, simd_t<float,8> &b)
{
    _transpose2_m1_ps256(a.x, b.x);
}

template<int M=1, typename std::enable_if<(M==2),int>::type = 0>
inline void simd_transpose(simd_t<float,8> &a, simd_t<float,8> &b)
{
    _transpose2_m2_ps256(a.x, b.x);
}

template<int M=1, typename std::enable_if<(M==4),int>::type = 0>
inline void simd_transpose(simd_t<float,8> &a, simd_t<float,8> &b)
{
    _transpose2_m4_ps256(a.x, b.x);
}

template<int M=1, typename std::enable_if<(M==1),int>::type = 0>
inline void simd_transpose(simd_t<float,8> &a, simd_t<float,8> &b, simd_t<float,8> &c, simd_t<float,8> &d)
{
    _transpose4_m1_ps256(a.x, b.x, c.x, d.x);
}

template<int M=1, typename std::enable_if<(M==2),int>::type = 0>
inline void simd_transpose(simd_t<float,8> &a, simd_t<float,8> &b, simd_t<float,8> &c, simd_t<float,8> &d)
{
    _transpose4_m2_ps256(a.x, b.x, c.x, d.x);
}

template<int M=1, typename std::enable_if<(M==1),int>::type = 0>
inline void simd_transpose(simd_t<float,8> &a, simd_t<float,8> &b, simd_t<float,8> &c, simd_t<float,8> &d, simd_t<float,8> &e, simd_t<float,8> &f, simd_t<float,8> &g, simd_t<float,8> &h)
{
    _transpose8_m1_ps256(a.x, b.x, c.x, d.x, e.x, f.x, g.x, h.x);
}


#endif  // __AVX2__


// -------------------------------------------------------------------------------------------------
//
// The boilerplate below defines the second simd_helpers transpose API:
//
//   simd_transpose<T,S,N,M=1> (simd_ntuple<T,S,N> &x);


template<int M, typename T, int S, int N, typename... Args, typename std::enable_if<(N==0),int>::type = 0>
inline void _simd_transpose(simd_ntuple<T,S,N> &x, Args& ... a)
{ 
    simd_transpose<M> (a...);
}

template<int M, typename T, int S, int N, typename... Args, typename std::enable_if<(N>0),int>::type = 0>
inline void _simd_transpose(simd_ntuple<T,S,N> &x, Args& ... a)
{
    _simd_transpose<M> (x.v, x.x, a...);
}

template<typename T, int S, int N, int M=1>
inline void simd_transpose(simd_ntuple<T,S,N> &x)
{
    _simd_transpose<M> (x);
}


} // namespace simd_helpers

#endif // _SIMD_HELPERS_TRANSPOSE_HPP
