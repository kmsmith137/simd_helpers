#ifndef _SIMD_HELPERS_QUANTIZE_HPP
#define _SIMD_HELPERS_QUANTIZE_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

#include "cast.hpp"
#include "downsample.hpp"

namespace simd_helpers {
#if 0
}  // pacify emacs c-mode
#endif


// -------------------------------------------------------------------------------------------------
//
// Quantization API:
//
//   using T = float;
//   constexpr int S = 8;   // simd size
//   constexpr int B = 1;   // bit depth
//   constexpr int K = (sizeof(T) * 8) / B;   // length of quantization kernel, in units sizeof(simd_t<T,S>)
//
//   const T *p;
//   simd_quantizer<T,S,B> q;
//
//   Option 1:
//      q.template put<0> (simd_load<T,S> (p));
//      q.template put<1> (simd_load<T,S> (p+S));
//         ...
//      q.template put<K-1> (simd_load<T,S> (p+(K-1)*S));
//      simd_t<int,S> out = q.get();
//
//   Option 2:
//      q.template mput<0,K> (p);
//      simd_t<int,S> out = q.get();
//
//   Option 3:
//      simd_t<int,S> out = q.quantize(p);


template<typename T, int S, int B>
struct simd_quantizer;


// -------------------------------------------------------------------------------------------------
//
// Only 1-bit quantization is implemented for now!
//
// A detail: we define 1-bit quantization by
//   q(x) = (x > 0.0) ? 1 : 0
//
// where we use ">" instead of ">=" so that applying 1-bit quantization to a "weights" array
// gives the bitmask (not an all-ones array).


// _get_qmask(): returns a particular bitmask used by the quantization kernels
template<int S> inline simd_t<int,S> _get_qmask();
template<> inline simd_t<int,4> _get_qmask() { return _mm_set_epi32(1U<<3, 1U<<2, 1U<<1, 1U); }
template<> inline simd_t<int,8> _get_qmask() { return _mm256_set_epi32(1U<<7, 1U<<6, 1U<<5, 1U<<4, 1U<<3, 1U<<2, 1U<<1, 1U); }


template<int S>
struct simd_quantizer<float,S,1> {
    simd_downsampler<int,S,32,simd_bitwise_or<int,S>> ds;
    const simd_t<float,S> z;
    const simd_t<int,S> c;
    
    simd_quantizer() : 
	z(0.0f),
	c(_get_qmask<S>())
    { }

    template<int M>
    inline void put(simd_t<float,S> x)
    {
	// Amount of left shifting to apply
	constexpr int LS = (M % (32/S)) * S;
	
	simd_t<float,S> fmask = (x > z);
	simd_t<int,S> imask = simd_cast<int,S> (fmask);
	ds.template put<M> ((imask & c) << LS);
    }

    // Defined below.
    template<int M, int N, bool Aligned=false>
    inline void mput(const float *p);

    inline simd_t<int,S> get() const
    {
	return ds.get();
    }

    inline simd_t<int,S> quantize(const float *p)
    {
	this->template mput<0,32> (p);
	return get();
    }
};


// -------------------------------------------------------------------------------------------------
//
// We implement simd_quantizer::mput() by delegating to an inline function _simd_quantizer_mput(), 
// so that we can use std::enable_if.


template<int M, int N, bool Aligned, typename T, int S, int B, typename std::enable_if<(N==0),int>::type=0>
inline void _simd_quantizer_mput(simd_quantizer<T,S,B> &q, const T *p) { }

template<int M, int N, bool Aligned, typename T, int S, int B, typename std::enable_if<(N>0),int>::type=0>
inline void _simd_quantizer_mput(simd_quantizer<T,S,B> &q, const T *p) 
{
    _simd_quantizer_mput<M,N-1,Aligned> (q, p);
    q.template put<M+N-1> (simd_load<T,S> (p + (N-1)*S));
}

template<int S>
template<int M, int N, bool Aligned>
inline void simd_quantizer<float,S,1>::mput(const float *p)
{
    _simd_quantizer_mput<M,N,Aligned> (*this, p);
}


}  // namespace simd_helpers

#endif  // _SIMD_HELPERS_QUANTIZE_HPP
