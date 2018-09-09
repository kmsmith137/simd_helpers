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
//
//   - The 256-bit AVX-but-not-AVX2 kernels may be a little suboptimal,
//     but this type of CPU is starting to become prehistoric anyway.
//
//   - Implementation depends on dead code elimination!



template<typename T, int S, int Nbytes = S*sizeof(T)>
struct simd_align_helper;


template<int A, typename T, int S>
inline simd_t<T,S> simd_align(simd_t<T,S> x, simd_t<T,S> y)
{
    simd_align_helper<T,S> h(x,y);
    return h.template align<A> ();
}


// FIXME deprecated alias for simd_align().
template<int A, typename T, int S>
inline simd_t<T,S> align(simd_t<T,S> x, simd_t<T,S> y)
{
    return simd_align<A> (x,y);
}


// -------------------------------------------------------------------------------------------------
//
// 128-bit


template<int Abytes, typename std::enable_if<(Abytes==0),int>::type = 0>
inline __m128i _simd_align128b(__m128i x, __m128i y) { return x; }

template<int Abytes, typename std::enable_if<(Abytes==16),int>::type = 0>
inline __m128i _simd_align128b(__m128i x, __m128i y) { return y; }

template<int Abytes, typename std::enable_if<((Abytes > 0) && (Abytes < 16)),int>::type = 0>
inline __m128i _simd_align128b(__m128i x, __m128i y) 
{ 
    return _mm_alignr_epi8(y, x, Abytes);
}


template<typename T, int S>
struct simd_align_helper<T,S,16>
{
    const simd_t<T,S> x, y;

    simd_align_helper(simd_t<T,S> x_, simd_t<T,S> y_) : x(x_), y(y_) { }
    
    template<int A> simd_t<T,S> align() const
    {
	constexpr int Abytes = A * sizeof(T);
	__m128i ret = _simd_align128b<Abytes> ((__m128i) x.x, (__m128i) y.x);
	return reinterpret_cast<decltype(x.x)> (ret);
    }
};



// -------------------------------------------------------------------------------------------------
//
// 256-bit


#if defined(__AVX2__)


template<int Abytes, typename std::enable_if<(Abytes==0),int>::type = 0>
inline __m256i _simd_align256b(__m256i x, __m256i xy, __m256i y)
{
    return x;
}

template<int Abytes, typename std::enable_if<((Abytes > 0) && (Abytes < 16)),int>::type = 0>
inline __m256i _simd_align256b(__m256i x, __m256i xy, __m256i y)
{
    return _mm256_alignr_epi8(xy, x, Abytes);    
}

template<int Abytes, typename std::enable_if<(Abytes==16),int>::type = 0>
inline __m256i _simd_align256b(__m256i x, __m256i xy, __m256i y)
{
    return xy;
}

template<int Abytes, typename std::enable_if<((Abytes > 16) && (Abytes < 32)),int>::type = 0>
inline __m256i _simd_align256b(__m256i x, __m256i xy, __m256i y)
{
    return _mm256_alignr_epi8(y, xy, Abytes-16);
}

template<int Abytes, typename std::enable_if<(Abytes==32),int>::type = 0>
inline __m256i _simd_align256b(__m256i x, __m256i xy, __m256i y)
{
    return y;
}


template<typename T, int S>
struct simd_align_helper<T,S,32>
{
    simd_t<T,S> x, y;
    __m256i xy;

    simd_align_helper(simd_t<T,S> x_, simd_t<T,S> y_) :
	x(x_), y(y_),
	xy(_mm256_permute2f128_si256((__m256i)x_.x, (__m256i)y_.x, 0x21))
    { }

    template<int A> simd_t<T,S> align() const
    {
	constexpr int Abytes = A * sizeof(T);
	__m256i ret = _simd_align256b<Abytes> ((__m256i) x.x, xy, (__m256i) y.x);
	return reinterpret_cast<decltype(x.x)> (ret);	
    }
};


#elif defined(__AVX__)   // 256-bit AVX-but-no-AVX2 kernels


template<int Abytes, typename std::enable_if<(Abytes==0),int>::type = 0>
inline __m256i _simd_align256b(__m256i x, __m256i y, __m128i x0, __m128i x1, __m128i y0, __m128i y1)
{
    return x;
}

template<int Abytes, typename std::enable_if<((Abytes > 0) && (Abytes < 16)),int>::type = 0>
inline __m256i _simd_align256b(__m256i x, __m256i y, __m128i x0, __m128i x1, __m128i y0, __m128i y1)
{
    __m128i ret0 = _simd_align128b<Abytes> (x0, x1);
    __m128i ret1 = _simd_align128b<Abytes> (x1, y0);
    return _mm256_insertf128_si256(_mm256_castsi128_si256(ret0), ret1, 1);
}

template<int Abytes, typename std::enable_if<(Abytes==16),int>::type = 0>
inline __m256i _simd_align256b(__m256i x, __m256i y, __m128i x0, __m128i x1, __m128i y0, __m128i y1)
{
    return _mm256_insertf128_si256(_mm256_castsi128_si256(x1), y0, 1);
}

template<int Abytes, typename std::enable_if<((Abytes > 16) && (Abytes < 32)),int>::type = 0>
inline __m256i _simd_align256b(__m256i x, __m256i y, __m128i x0, __m128i x1, __m128i y0, __m128i y1)
{
    __m128i ret0 = _simd_align128b<Abytes> (x0, x1);
    __m128i ret1 = _simd_align128b<Abytes> (x1, y0);
    return _mm256_insertf128_si256(_mm256_castsi128_si256(ret0), ret1, 1);
}

template<int Abytes, typename std::enable_if<(Abytes==32),int>::type = 0>
inline __m256i _simd_align256b(__m256i x, __m256i y, __m128i x0, __m128i x1, __m128i y0, __m128i y1)
{
    return y;
}


template<typename T, int S>
struct simd_align_helper<T,S,32>
{
    const __m256i x, y;
    const __m128i x0, x1, y0, y1;

    simd_align_helper(simd_t<T,S> x_, simd_t<T,S> y_) :
	x((__m256i) x_.x),
	y((__m256i) t_.x),
	x0(_mm256_extractf128_si256(x,0)),
	x1(_mm256_extractf128_si256(x,1)),
	y0(_mm256_extractf128_si256(y,0)),
	y1(_mm256_extractf128_si256(y,1))
    { }

    template<int A> simd_t<T,S> align() const
    {
	__m256i ret = _simd_align256b<(A*sizeof(T))> (x, y, x0, x1, y0, y1);
	return reinterpret_cast<decltype(x.x)> (ret);	
    }
};
    

#endif  // 256-bit kernels end here


// -------------------------------------------------------------------------------------------------
//
// ntuple version of simd_align().
//
// FIXME it would make sense to define 'struct simd_ntuple_align_helper'.



template<int A, typename T, int S, int N, typename std::enable_if<((N==0) && (A<=S)),int>::type = 0>
inline void simd_align(simd_ntuple<T,S,N> &dst, const simd_ntuple<T,S,N> &x, const simd_ntuple<T,S,N> &y) 
{ }


template<int A, typename T, int S, int N, typename std::enable_if<((N>0) && (A<=S)),int>::type = 0>
inline void simd_align(simd_ntuple<T,S,N> &dst, const simd_ntuple<T,S,N> &x, const simd_ntuple<T,S,N> &y)
{
    simd_align<A> (dst.v, x.v, y.v);
    dst.x = simd_align<A> (x.x, y.x);
}


// FIXME deprecated alias for simd_align().
template<int A, typename T, int S, int N>
inline void align(simd_ntuple<T,S,N> &dst, const simd_ntuple<T,S,N> &x, const simd_ntuple<T,S,N> &y)
{
    simd_align<A> (dst, x, y);
}


}  // namespace simd_helpers

#endif  // _SIMD_HELPERS_ALIGN_HPP
