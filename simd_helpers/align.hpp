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
// 128-bit


template<int Abytes, typename std::enable_if<(Abytes==0),int>::type = 0>
inline __m128i _align128b(__m128i x, __m128i y) { return x; }

template<int Abytes, typename std::enable_if<(Abytes==16),int>::type = 0>
inline __m128i _align128b(__m128i x, __m128i y) { return y; }

template<int Abytes, typename std::enable_if<((Abytes > 0) && (Abytes < 16)),int>::type = 0>
inline __m128i _align128b(__m128i x, __m128i y) 
{ 
    return _mm_alignr_epi8(y, x, Abytes);
}

template<int A, typename T, int S, typename std::enable_if<((A <= S) && (S*sizeof(T)==16)),int>::type = 0>
inline simd_t<T,S> align(simd_t<T,S> x, simd_t<T,S> y)
{
    constexpr int Abytes = A * sizeof(T);
    __m128i ret = _align128b<Abytes> ((__m128i) x.x, (__m128i) y.x);
    return reinterpret_cast<decltype(x.x)> (ret);
}


// -------------------------------------------------------------------------------------------------
//
// 256-bit


#ifdef __AVX__


template<int Abytes, typename std::enable_if<(Abytes==0),int>::type = 0>
inline __m256i _align256b(__m256i x, __m256i y) { return x; }

template<int Abytes, typename std::enable_if<(Abytes==32),int>::type = 0>
inline __m256i _align256b(__m256i x, __m256i y) { return y; }

template<int Abytes, typename std::enable_if<((Abytes > 0) && (Abytes < 16)),int>::type = 0>
inline __m256i _align256b(__m256i x, __m256i y) 
{ 
#ifdef __AVX2__
    __m256i t = _mm256_permute2f128_si256(x, y, 0x21);
    return _mm256_alignr_epi8(t, x, Abytes);
#else
    __m128i x0 = _mm256_extractf128_si256(x, 0);
    __m128i x1 = _mm256_extractf128_si256(x, 1);
    __m128i y0 = _mm256_extractf128_si256(y, 0);
    __m128i z0 = _mm_alignr_epi8(x1, x0, Abytes);
    __m128i z1 = _mm_alignr_epi8(y0, x1, Abytes);
    return _mm256_insertf128_si256(_mm256_castsi128_si256(z0), z1, 1);
#endif
}

template<int Abytes, typename std::enable_if<(Abytes==16),int>::type = 0>
inline __m256i _align256b(__m256i x, __m256i y) 
{ 
    return _mm256_permute2f128_si256(x, y, 0x21);
}

template<int Abytes, typename std::enable_if<((Abytes > 16) && (Abytes < 32)),int>::type = 0>
inline __m256i _align256b(__m256i x, __m256i y) 
{ 
#ifdef __AVX2__
    __m256i t = _mm256_permute2f128_si256(x, y, 0x21);
    return _mm256_alignr_epi8(y, t, Abytes-16);
#else
    __m128i x1 = _mm256_extractf128_si256(x, 1);
    __m128i y0 = _mm256_extractf128_si256(y, 0);
    __m128i y1 = _mm256_extractf128_si256(y, 1);
    __m128i z0 = _mm_alignr_epi8(y0, x1, Abytes-16);
    __m128i z1 = _mm_alignr_epi8(y1, y0, Abytes-16);
    return _mm256_insertf128_si256(_mm256_castsi128_si256(z0), z1, 1);
#endif
}


template<int A, typename T, int S, typename std::enable_if<((A <= S) && (S*sizeof(T)==32)),int>::type = 0>
inline simd_t<T,S> align(simd_t<T,S> x, simd_t<T,S> y)
{
    constexpr int Abytes = A * sizeof(T);
    __m256i ret = _align256b<Abytes> ((__m256i) x.x, (__m256i) y.x);
    return reinterpret_cast<decltype(x.x)> (ret);
}


#endif  // __AVX__


// -------------------------------------------------------------------------------------------------


template<int A, typename T, int S, int N, typename std::enable_if<((N==0) && (A<=S)),int>::type = 0>
inline void align(simd_ntuple<T,S,N> &dst, const simd_ntuple<T,S,N> &x, const simd_ntuple<T,S,N> &y) 
{ }


template<int A, typename T, int S, int N, typename std::enable_if<((N>0) && (A<=S)),int>::type = 0>
inline void align(simd_ntuple<T,S,N> &dst, const simd_ntuple<T,S,N> &x, const simd_ntuple<T,S,N> &y)
{
    align<A> (dst.v, x.v, y.v);
    dst.x = align<A> (x.x, y.x);
}


}  // namespace simd_helpers

#endif  // _SIMD_HELPERS_ALIGN_HPP
