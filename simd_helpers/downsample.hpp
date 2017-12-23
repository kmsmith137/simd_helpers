#ifndef _SIMD_HELPERS_DOWNSAMPLE_HPP
#define _SIMD_HELPERS_DOWNSAMPLE_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

#include "core.hpp"
#include "simd_int32.hpp"
#include "simd_float32.hpp"
#include "simd_ntuple.hpp"

namespace simd_helpers {
#if 0
}  // pacify emacs c-mode
#endif


// FIXME use more C++ template-ology to eliminate cut-and-paste between int, float kernels!


// This file contains some "weird boilerplate" designed to provide a streamlined
// interface while working around C++ restrictions on partial template specialization.
// The "_simd_downsampler" class is part of the weird boilerplate!
template<typename T, int S, int D, typename Op> struct _simd_downsampler;


// -------------------------------------------------------------------------------------------------
//
// Binary operators which can be passed as template parameters to simd_downsample(r).
//
// FIXME: should define operators systematically (these are just the ones I happen to have needed so far).


template<typename T, int S>
struct simd_add {
    static inline simd_t<T,S> op(simd_t<T,S> x, simd_t<T,S> y) { return x + y; }
};

template<typename T, int S>
struct simd_max {
    static inline simd_t<T,S> op(simd_t<T,S> x, simd_t<T,S> y) { return x.max(y); }
};

template<typename T, int S>
struct simd_bitwise_or {
    static inline simd_t<T,S> op(simd_t<T,S> x, simd_t<T,S> y) { return x | y; }
};


// -------------------------------------------------------------------------------------------------
//
// Downsampling API defined in this file


template<typename T, int S, int D, typename Op = simd_add<T,S>, bool two_stage = (D > S)>
struct simd_downsampler;


// One-stage case (D <= S)
template<typename T, int S, int D, typename Op>
struct simd_downsampler<T,S,D,Op,false> {
    static constexpr bool valid = (D >= 1) && (D <= S) && (D & (D-1)) == 0;
    static_assert(valid, "simd_downsampler: downsampling factor D must be either a power of two, or an integer multiple of simd size S");

    template<int N> inline void put(simd_t<T,S> x);
    inline simd_t<T,S> get() const;

    // Weird boilerplate
    _simd_downsampler<T,S,D,Op> _s;
};


// Two-stage case (D > S)
template<typename T, int S, int D, typename Op>
struct simd_downsampler<T,S,D,Op,true> {
    static constexpr bool valid = (D > S) && (D % S) == 0;
    static_assert(valid, "simd_downsampler: downsampling factor D must be either a power of two, or an integer multiple of simd size S");

    template<int N> inline void put(simd_t<T,S> x);
    inline simd_t<T,S> get() const;

    simd_downsampler<T,S,S,Op> _sd;
    simd_t<T,S> _acc;
};


// N-tuple interface
template<typename T, int S, typename Op = simd_add<T,S>, int N>
inline simd_t<T,S> simd_downsample(const simd_ntuple<T,S,N> &x);


// -------------------------------------------------------------------------------------------------
//
// Downsampling kernels follow.
// Trivial case: downsampling-by-one.


template<typename T, int S, typename Op>
struct _simd_downsampler<T,S,1,Op>
{
    simd_t<T,S> x;

    inline void put0(simd_t<T,S> x_) { x = x_; }
    inline simd_t<T,S> get() const { return x; }
};


// -------------------------------------------------------------------------------------------------
//
// 128-bit downsampling by two


template<typename Op>
struct _simd_downsampler<float,4,2,Op>
{
    simd_t<float,4> a;

    inline void put0(simd_t<float,4> a_) { a = a_; }

    inline void put1(simd_t<float,4> b)
    {
	simd_t<float,4> u = _mm_shuffle_ps(a.x, b.x, 0x88);   // [a0 a2 b0 b2],  0x88 = (2020)_4
	simd_t<float,4> v = _mm_shuffle_ps(a.x, b.x, 0xdd);   // [a1 a3 b1 b3],  0xdd = (3131)_4
	a = Op::op(u,v);
    }

    inline simd_t<float,4> get() const { return a; }
};


template<typename Op>
struct _simd_downsampler<int,4,2,Op>
{
    simd_t<int,4> a;

    inline void put0(simd_t<int,4> a_) { a = a_; }

    inline void put1(simd_t<int,4> b)
    {
	simd_t<int,4> u = _mm_xshuffle_epi32(a.x, b.x, 0x88);
	simd_t<int,4> v = _mm_xshuffle_epi32(a.x, b.x, 0xdd);
	a = Op::op(u,v);
    }

    inline simd_t<int,4> get() const { return a; }
};


// -------------------------------------------------------------------------------------------------
//
// 128-bit downsampling by four


template<typename Op>
struct _simd_downsampler<float,4,4,Op>
{
    simd_t<float,4> x, y;
    
    static inline simd_t<float,4> _ds2(simd_t<float,4> a, simd_t<float,4> b)
    {
	__m128 u = _mm_shuffle_ps(a.x, b.x, 0x88);
	__m128 v = _mm_shuffle_ps(a.x, b.x, 0xdd);
	return Op::op(u,v);
    }
    
    // I think this is fastest.
    inline void put0(simd_t<float,4> t) { x = t; }
    inline void put1(simd_t<float,4> t) { x = _ds2(x,t); }
    inline void put2(simd_t<float,4> t) { y = t; }
    inline void put3(simd_t<float,4> t) { y = _ds2(x,_ds2(y,t)); }

    inline simd_t<float,4> get() const { return y; }
};


template<typename Op>
struct _simd_downsampler<int,4,4,Op>
{
    simd_t<int,4> x, y;
    
    static inline simd_t<int,4> _ds2(simd_t<int,4> a, simd_t<int,4> b)
    {
	__m128i u = _mm_xshuffle_epi32(a.x, b.x, 0x88);
	__m128i v = _mm_xshuffle_epi32(a.x, b.x, 0xdd);
	return Op::op(u,v);
    }
    
    inline void put0(simd_t<int,4> t) { x = t; }
    inline void put1(simd_t<int,4> t) { x = _ds2(x,t); }
    inline void put2(simd_t<int,4> t) { y = t; }
    inline void put3(simd_t<int,4> t) { y = _ds2(x,_ds2(y,t)); }

    inline simd_t<int,4> get() const { return y; }
};


#ifdef __AVX__
// 256-bit kernels start here


// -------------------------------------------------------------------------------------------------
//
// 256-bit downsampling by two


template<typename Op>
struct _simd_downsampler<float,8,2,Op>
{
    simd_t<float,8> a;

    inline void put0(simd_t<float,8> a_) { a = a_; }

    inline void put1(simd_t<float,8> b) 
    { 
	// Similar to _kernel128_downsample2(), but with an extra 64-bit swap
	// which in theory helps the subsequent permute/blend block pipeline.
	// (Not sure if this actually helps!)

	simd_t<float,8> u = _mm256_shuffle_ps(b.x, a.x, 0x88);   // [b0 b2 a0 a2],  0x88 = (2020)_4
	simd_t<float,8> v = _mm256_shuffle_ps(b.x, a.x, 0xdd);   // [b1 b3 a1 a3],  0xdd = (3131)_4
	simd_t<float,8> w = Op::op(u,v);

	// Now we have a vector [ w2 w0 w3 w1 ], where w_i is 64 bits
	// and we want to rearrange to [ w0 w1 w2 w3 ].
	
	__m256 x = _mm256_permute_ps(w.x, 0x4e);          // [ w0 w2 w1 w3 ],  0x4e = (1032)_4
	__m256 y = _mm256_permute2f128_ps(w.x, w.x, 0x01);  // [ w3 w1 w2 w0 ]
	
	a = _mm256_blend_ps(x, y, 0x3c);   // (00111100)_2
    }

    inline simd_t<float,8> get() const { return a; }
};


template<typename Op>
struct _simd_downsampler<int,8,2,Op>
{
    simd_t<int,8> a;

    inline void put0(simd_t<int,8> a_) { a = a_; }

    inline void put1(simd_t<int,8> b) 
    { 
	simd_t<int,8> u = _mm256_xshuffle_epi32(b.x, a.x, 0x88);
	simd_t<int,8> v = _mm256_xshuffle_epi32(b.x, a.x, 0xdd);
	simd_t<int,8> w = Op::op(u,v);
	
	__m256i x = _mm256_shuffle_epi32(w.x, 0x4e);
	__m256i y = _mm256_permute2f128_si256(w.x, w.x, 0x01);
	
	a = _mm256_blend_epi32(x, y, 0x3c);
    }

    inline simd_t<int,8> get() const { return a; }
};


// -------------------------------------------------------------------------------------------------
//
// 256-bit downsampling by four


template<typename Op>
struct _simd_downsampler<float,8,4,Op>
{
    simd_t<float,8> ac;
    simd_t<float,8> bd;

    // Until the very end of put3(), all operations are happening
    // in sync on the two 128-bit halves of the 256-bit register.  
    // Thus we denote a = [a0 a1 a2 a3], and analogously for b,c,d.

    // The _ds2() helper combines a and c, to obtain [ a01 a23 c01 c23 ]
    // The logic here is the same as _simd_downsampler<float,4,2>.
    
    static inline simd_t<float,8> _ds2(simd_t<float,8> a, simd_t<float,8> c)
    {
	simd_t<float,8> u = _mm256_shuffle_ps(a.x, c.x, 0x88);
	simd_t<float,8> v = _mm256_shuffle_ps(a.x, c.x, 0xdd);
	return Op::op(u, v);
    }

    inline void put0(simd_t<float,8> a) { ac = a; }
    inline void put1(simd_t<float,8> b) { bd = b; }
    inline void put2(simd_t<float,8> c) { ac = _ds2(ac, c); }

    inline void put3(simd_t<float,8> d)
    {
	bd = _ds2(bd, d);

	// At this point in the code, we have ac = [ a01 a23 c01 c23 ]
	// and bd = [ b01 b23 d01 d23 ].
	
	simd_t<float,8> u = _mm256_shuffle_ps(ac.x, bd.x, 0x22);  // [ c01 a01 d01 b01 ],  0x22 = (0202)_4
	simd_t<float,8> v = _mm256_shuffle_ps(ac.x, bd.x, 0x77);  // [ c23 a23 d23 b23 ],  0x77 = (1313)_4
	simd_t<float,8> w = Op::op(u, v);                         // [ c a d b ]

	// The 256-bit vector w is the output we want, but it has ordering [ w4 w0 w6 w2 w5 w1 w7 w3 ].

	__m256 x = _mm256_permute_ps(w.x, 0xb1);            // [ w0 w4 w2 w6 w1 w5 w3 w7 ],  0xb1 = (2301)_4
	__m256 y = _mm256_permute2f128_ps(w.x, w.x, 0x01);  // [ w5 w1 w7 w3 w4 w0 w6 w2 ]

	bd = _mm256_blend_ps(x, y, 0x5a);   // (01011010)_2
    }

    inline simd_t<float,8> get() const { return bd; }
};


template<typename Op>
struct _simd_downsampler<int,8,4,Op>
{
    simd_t<int,8> ac;
    simd_t<int,8> bd;
    
    static inline simd_t<int,8> _ds2(simd_t<int,8> a, simd_t<int,8> c)
    {
	simd_t<int,8> u = _mm256_xshuffle_epi32(a.x, c.x, 0x88);
	simd_t<int,8> v = _mm256_xshuffle_epi32(a.x, c.x, 0xdd);
	return Op::op(u, v);
    }

    inline void put0(simd_t<int,8> a) { ac = a; }
    inline void put1(simd_t<int,8> b) { bd = b; }
    inline void put2(simd_t<int,8> c) { ac = _ds2(ac, c); }

    inline void put3(simd_t<int,8> d)
    {
	bd = _ds2(bd, d);
	
	simd_t<int,8> u = _mm256_xshuffle_epi32(ac.x, bd.x, 0x22);
	simd_t<int,8> v = _mm256_xshuffle_epi32(ac.x, bd.x, 0x77);
	simd_t<int,8> w = Op::op(u, v);
	
	__m256i x = _mm256_shuffle_epi32(w.x, 0xb1);
	__m256i y = _mm256_permute2f128_si256(w.x, w.x, 0x01);

	bd = _mm256_blend_epi32(x, y, 0x5a);
    }

    inline simd_t<int,8> get() const { return bd; }
};


// -------------------------------------------------------------------------------------------------
//
// 256-bit downsampling by 8.


template<typename Op>
struct _simd_downsampler<float,8,8,Op>
{
    simd_t<float,8> x0;
    simd_t<float,8> x1;
    simd_t<float,8> x2;

    // _ds2(): downsamples by a factor 2 within each 128-bit "half"
    // of the 256-bit AVX register, i.e. the return value is:
    //
    //  [ a01 a23 b01 b23 a45 a67 b45 b67 ]
    
    static inline simd_t<float,8> _ds2(simd_t<float,8> a, simd_t<float,8> b)
    {
	simd_t<float,8> u = _mm256_shuffle_ps(a.x, b.x, 0x88);
	simd_t<float,8> v = _mm256_shuffle_ps(a.x, b.x, 0xdd);
	return Op::op(u,v);
    }

    inline void put0(simd_t<float,8> a) { x0 = a; }
    inline void put1(simd_t<float,8> b) { x0 = _ds2(x0,b); }
    inline void put2(simd_t<float,8> c) { x1 = c; }
    inline void put3(simd_t<float,8> d) { x0 = _ds2(x0,_ds2(x1,d)); }
    inline void put4(simd_t<float,8> e) { x1 = e; }
    inline void put5(simd_t<float,8> f) { x1 = _ds2(x1,f); }
    inline void put6(simd_t<float,8> g) { x2 = g; }

    inline void put7(simd_t<float,8> h)
    {
	x1 = _ds2(x1, _ds2(x2,h));

	// At this point in the code the original inputs have been downsampled
	// by a factor of 4, but nothing has been exchanged between 128-bit halves
	// of the 256-bit AVX register.  Thus we can write:
	//
	//   x0 = [ a0 b0 c0 d0 a1 b1 c1 d1 ]
	//   x1 = [ e0 f0 g0 h0 e1 f1 g1 h1 ]

	__m256 u = _mm256_blend_ps(x0.x, x1.x, 0xf0);         // [ a0 b0 c0 d0 e1 f1 g1 h1 ],  0xf0 = (11110000)_2
	__m256 v = _mm256_permute2f128_ps(x0.x, x1.x, 0x21);  // [ a1 b1 c1 d1 e0 f0 g0 h0 ]

	x2 = Op::op(u, v);
    }

    inline simd_t<float,8> get() const { return x2; }
};


template<typename Op>
struct _simd_downsampler<int,8,8,Op>
{
    simd_t<int,8> x0;
    simd_t<int,8> x1;
    simd_t<int,8> x2;
    
    static inline simd_t<int,8> _ds2(simd_t<int,8> a, simd_t<int,8> b)
    {
	simd_t<int,8> u = _mm256_xshuffle_epi32(a.x, b.x, 0x88);
	simd_t<int,8> v = _mm256_xshuffle_epi32(a.x, b.x, 0xdd);
	return Op::op(u,v);
    }

    inline void put0(simd_t<int,8> a) { x0 = a; }
    inline void put1(simd_t<int,8> b) { x0 = _ds2(x0,b); }
    inline void put2(simd_t<int,8> c) { x1 = c; }
    inline void put3(simd_t<int,8> d) { x0 = _ds2(x0,_ds2(x1,d)); }
    inline void put4(simd_t<int,8> e) { x1 = e; }
    inline void put5(simd_t<int,8> f) { x1 = _ds2(x1,f); }
    inline void put6(simd_t<int,8> g) { x2 = g; }

    inline void put7(simd_t<int,8> h)
    {
	x1 = _ds2(x1, _ds2(x2,h));

	__m256i u = _mm256_blend_epi32(x0.x, x1.x, 0xf0);         // [ a0 b0 c0 d0 e1 f1 g1 h1 ],  0xf0 = (11110000)_2
	__m256i v = _mm256_permute2f128_si256(x0.x, x1.x, 0x21);  // [ a1 b1 c1 d1 e0 f0 g0 h0 ]

	x2 = Op::op(u, v);
    }

    inline simd_t<int,8> get() const { return x2; }
};


// 256-bit kernels end here
#endif // __AVX__


// -------------------------------------------------------------------------------------------------
//
// Weird template boilerplate: single-stage case (D <= S).


template<int N, typename Td, typename Ts, typename std::enable_if<(N==0),int>::type=0> inline void _simd_ds_put(Td &d, Ts x) { d.put0(x); }
template<int N, typename Td, typename Ts, typename std::enable_if<(N==1),int>::type=0> inline void _simd_ds_put(Td &d, Ts x) { d.put1(x); }
template<int N, typename Td, typename Ts, typename std::enable_if<(N==2),int>::type=0> inline void _simd_ds_put(Td &d, Ts x) { d.put2(x); }
template<int N, typename Td, typename Ts, typename std::enable_if<(N==3),int>::type=0> inline void _simd_ds_put(Td &d, Ts x) { d.put3(x); }
template<int N, typename Td, typename Ts, typename std::enable_if<(N==4),int>::type=0> inline void _simd_ds_put(Td &d, Ts x) { d.put4(x); }
template<int N, typename Td, typename Ts, typename std::enable_if<(N==5),int>::type=0> inline void _simd_ds_put(Td &d, Ts x) { d.put5(x); }
template<int N, typename Td, typename Ts, typename std::enable_if<(N==6),int>::type=0> inline void _simd_ds_put(Td &d, Ts x) { d.put6(x); }
template<int N, typename Td, typename Ts, typename std::enable_if<(N==7),int>::type=0> inline void _simd_ds_put(Td &d, Ts x) { d.put7(x); }

template<typename T, int S, int D, typename Op>
template<int N>
inline void simd_downsampler<T,S,D,Op,false>::put(simd_t<T,S> x)
{ 
    _simd_ds_put<N> (_s, x); 
}

template<typename T, int S, int D, typename Op> 
inline simd_t<T,S> simd_downsampler<T,S,D,Op,false>::get() const
{
    return _s.get();
}


// -------------------------------------------------------------------------------------------------
//
// Weird template boilerplate: two-stage case (D > S).


template<int N, int R, typename Op, typename T, int S, typename std::enable_if<(N % R == 0),int>::type=0>
inline void _simd_ds_stage1(simd_t<T,S> &acc, simd_t<T,S> x)
{
    acc = x;
}

template<int N, int R, typename Op, typename T, int S, typename std::enable_if<(N % R > 0),int>::type=0>
inline void _simd_ds_stage1(simd_t<T,S> &acc, simd_t<T,S> x)
{
    acc = Op::op(acc, x);
}

template<int N, int R, typename Op, typename T, int S, typename std::enable_if<(N % R != R-1),int>::type=0>
inline void _simd_ds_stage2(simd_downsampler<T,S,S,Op> &sd, simd_t<T,S> acc)
{
    // Do nothing.
}

template<int N, int R, typename Op, typename T, int S, typename std::enable_if<(N % R == R-1),int>::type=0>
inline void _simd_ds_stage2(simd_downsampler<T,S,S,Op> &sd, simd_t<T,S> acc)
{
    sd.template put<(N/R)> (acc);
}

template<typename T, int S, int D, typename Op>
template<int N>
inline void simd_downsampler<T,S,D,Op,true>::put(simd_t<T,S> x)
{
    constexpr int R = D / S;

    _simd_ds_stage1<N,R,Op> (_acc, x);
    _simd_ds_stage2<N,R,Op> (_sd, _acc);
}

template<typename T, int S, int D, typename Op> 
inline simd_t<T,S> simd_downsampler<T,S,D,Op,true>::get() const
{
    return _sd.get();
}


// -------------------------------------------------------------------------------------------------
//
// Weird template boilerplate: N-tuple interface


template<typename Td, typename T, int S, int N, typename std::enable_if<(N==0),int>::type=0>
inline void _simd_ds_ntuple(Td &d, const simd_ntuple<T,S,N> &x) { }

template<typename Td, typename T, int S, int N, typename std::enable_if<(N>0),int>::type=0>
inline void _simd_ds_ntuple(Td &d, const simd_ntuple<T,S,N> &x) 
{ 
    _simd_ds_ntuple(d, x.v);
    d.template put<N-1> (x.x);
}

template<typename T, int S, typename Op, int N>
inline simd_t<T,S> simd_downsample(const simd_ntuple<T,S,N> &x)
{
    simd_downsampler<T,S,N,Op> ds;
    _simd_ds_ntuple(ds, x);
    return ds.get();
}


}  // namespace simd_helpers

#endif  // _SIMD_HELPERS_DOWNSAMPLE_HPP
