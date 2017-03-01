#ifndef _SIMD_HELPERS_ALIGN_HPP
#define _SIMD_HELPERS_ALIGN_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

#include "core.hpp"
#include "simd_ntuple.hpp"

namespace simd_helpers {
#if 0
}  // pacify emacs c-mode
#endif

// This file defines:
//
//   simd_t<T,S> align<A> (simd_t<T,S> x, simd_t<T,S> y);
//
//   void align<A> (const simd_ntuple<T,S,N> &x, const simd_ntuple<T,S,N> &y);



// Implementation notes:
// 
//   - The align intrinsics are
//        __m128i _mm_alignr_epi8(__m128i, __m128i, imm8)
//        __m256i _mm256_alignr_epi8(__m256i, __m256i, imm8)
//
//   - The last argument is the align count in bytes (not words), and the first 
//     two arguments are swapped relative to their "natural" ordering.
// 
//   - In the 256-bit case, the alignment takes place _within_ 128-bit
//     subregisters, which is usually not what's wanted.  One can work
//     around this with a call to _mm256_permute2f128_ps(), see below.
//
//   - The intrinsics are defined only for integer SIMD types, but in the
//     case of a floating-point type, it's harmless to cast.


// -------------------------------------------------------------------------------------------------
//
// "Thin" wrappers around _mm_alignr_epi8() or __m256_alignr_epi8().
//
//    __m128i _align128b<Abytes> (__m128i, __m128i)
//    __m256i _align256b<Abytes> (__m256i, __m256i)
//
// The template parameter Abytes is the align count in bytes.


template<unsigned int Abytes, typename std::enable_if<(Abytes==0),int>::type = 0>
inline __m128i _align128b(__m128i x, __m128i y) { return x; }

template<unsigned int Abytes, typename std::enable_if<(Abytes==16),int>::type = 0>
inline __m128i _align128b(__m128i x, __m128i y) { return y; }

template<unsigned int Abytes, typename std::enable_if<((Abytes > 0) && (Abytes < 16)),int>::type = 0>
inline __m128i _align128b(__m128i x, __m128i y) 
{ 
    return _mm_alignr_epi8(y, x, Abytes);
}


template<unsigned int Abytes, typename std::enable_if<(Abytes==0),int>::type = 0>
inline __m256i _align256b(__m256i x, __m256i y) { return x; }

template<unsigned int Abytes, typename std::enable_if<(Abytes==32),int>::type = 0>
inline __m256i _align256b(__m256i x, __m256i y) { return y; }

template<unsigned int Abytes, typename std::enable_if<((Abytes > 0) && (Abytes < 16)),int>::type = 0>
inline __m256i _align256b(__m256i x, __m256i y) 
{ 
    __m256i t = _mm256_permute2f128_ps(x, y, 0x21);
    return _mm256_alignr_epi8(t, x, Abytes);
}

template<unsigned int Abytes, typename std::enable_if<(Abytes==16),int>::type = 0>
inline __m256i _align256b(__m256i x, __m256i y) 
{ 
    return _mm256_permute2f128_ps(x, y, 0x21);
}

template<unsigned int Abytes, typename std::enable_if<((Abytes > 16) && (Abytes < 32)),int>::type = 0>
inline __m256i _align256b(__m256i x, __m256i y) 
{ 
    __m256i t = _mm256_permute2f128_ps(x, y, 0x21);
    return _mm256_alignr_epi8(y, t, Abytes-16);
}


// -------------------------------------------------------------------------------------------------


template<unsigned int A, typename T, unsigned int S, typename std::enable_if<((A <= S) && (S*sizeof(T)==16)),int>::type = 0>
inline simd_t<T,S> align(simd_t<T,S> x, simd_t<T,S> y)
{
    constexpr unsigned int Abytes = A * sizeof(T);
    __m128i ret = _align128b<Abytes> ((__m128i) x.x, (__m128i) y.x);
    return static_cast<decltype(x.x)> (ret);
}

template<unsigned int A, typename T, unsigned int S, typename std::enable_if<((A <= S) && (S*sizeof(T)==32)),int>::type = 0>
inline simd_t<T,S> align(simd_t<T,S> x, simd_t<T,S> y)
{
    constexpr unsigned int Abytes = A * sizeof(T);
    __m256i ret = _align256b<Abytes> ((__m256i) x.x, (__m256i) y.x);
    return static_cast<decltype(x.x)> (ret);
}


// -------------------------------------------------------------------------------------------------


template<unsigned int A, typename T, unsigned int S, unsigned int N, typename std::enable_if<((N==0) && (A<=S)),int>::type = 0>
inline void align(simd_ntuple<T,S,N> &dst, const simd_ntuple<T,S,N> &x, const simd_ntuple<T,S,N> &y) 
{ }


template<unsigned int A, typename T, unsigned int S, unsigned int N, typename std::enable_if<((N>0) && (A<=S)),int>::type = 0>
inline void align(simd_ntuple<T,S,N> &dst, const simd_ntuple<T,S,N> &x, const simd_ntuple<T,S,N> &y)
{
    align<A> (dst.v, x.v, y.v);
    dst.x = align<A> (x.x, y.x);
}


}  // namespace simd_helpers

#endif  // _SIMD_HELPERS_ALIGN_HPP
