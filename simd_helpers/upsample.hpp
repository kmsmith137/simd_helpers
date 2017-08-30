#ifndef _SIMD_HELPERS_UPSAMPLE_HPP
#define _SIMD_HELPERS_UPSAMPLE_HPP

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


// -------------------------------------------------------------------------------------------------
//
// Upsampling API defined in this file.


template<typename T, int S, int D>
struct simd_upsampler {
    explicit simd_upsampler(simd_t<T,S> x);
    template<int N> inline simd_t<T,S> get() const;
};


template<typename T, int S, int N>
inline void simd_upsample(simd_ntuple<T,S,N> &dst, simd_t<T,S> src);



// -------------------------------------------------------------------------------------------------
//
// Trivial case: upsampling-by-one


template<typename T, int S>
struct simd_upsampler<T,S,1>
{
    simd_t<T,S> x;

    simd_upsampler(simd_t<T,S> x_) { x = x_; }

    template<int N> inline simd_t<T,S> get() const { return x; }
};


// -------------------------------------------------------------------------------------------------
//
// 128-bit upsample by two


template<>
struct simd_upsampler<float,4,2> {
    simd_t<float,4> x;

    simd_upsampler(simd_t<float,4> t) { x = t; }

    template<int N> inline simd_t<float,4> get() const;
};

template<> inline simd_t<float,4> simd_upsampler<float,4,2>::get<0> () const { return _mm_permute_ps(x.x, 0x50); }  // (1100)_4
template<> inline simd_t<float,4> simd_upsampler<float,4,2>::get<1> () const { return _mm_permute_ps(x.x, 0xfa); }  // (3322)_4


template<>
struct simd_upsampler<int,4,2> {
    simd_t<int,4> x;

    simd_upsampler(simd_t<int,4> t) { x = t; }

    template<int N> inline simd_t<int,4> get() const;
};

template<> inline simd_t<int,4> simd_upsampler<int,4,2>::get<0> () const { return _mm_shuffle_epi32(x.x, 0x50); }  // (1100)_4
template<> inline simd_t<int,4> simd_upsampler<int,4,2>::get<1> () const { return _mm_shuffle_epi32(x.x, 0xfa); }  // (3322)_4


// -------------------------------------------------------------------------------------------------
//
// 128-bit upsample by four


template<>
struct simd_upsampler<float,4,4> {
    simd_t<float,4> x;

    simd_upsampler(simd_t<float,4> t) { x = t; }

    template<int N> inline simd_t<float,4> get() const;
};

template<> inline simd_t<float,4> simd_upsampler<float,4,4>::get<0> () const { return _mm_permute_ps(x.x, 0x00); }  // (0000)_4
template<> inline simd_t<float,4> simd_upsampler<float,4,4>::get<1> () const { return _mm_permute_ps(x.x, 0x55); }  // (1111)_4
template<> inline simd_t<float,4> simd_upsampler<float,4,4>::get<2> () const { return _mm_permute_ps(x.x, 0xaa); }  // (2222)_4
template<> inline simd_t<float,4> simd_upsampler<float,4,4>::get<3> () const { return _mm_permute_ps(x.x, 0xff); }  // (3333)_4


template<>
struct simd_upsampler<int,4,4> {
    simd_t<int,4> x;

    simd_upsampler(simd_t<int,4> t) { x = t; }

    template<int N> inline simd_t<int,4> get() const;
};

template<> inline simd_t<int,4> simd_upsampler<int,4,4>::get<0> () const { return _mm_shuffle_epi32(x.x, 0x00); }  // (0000)_4
template<> inline simd_t<int,4> simd_upsampler<int,4,4>::get<1> () const { return _mm_shuffle_epi32(x.x, 0x55); }  // (1111)_4
template<> inline simd_t<int,4> simd_upsampler<int,4,4>::get<2> () const { return _mm_shuffle_epi32(x.x, 0xaa); }  // (2222)_4
template<> inline simd_t<int,4> simd_upsampler<int,4,4>::get<3> () const { return _mm_shuffle_epi32(x.x, 0xff); }  // (3333)_4


// -------------------------------------------------------------------------------------------------
//
// 256-bit upsample by two


template<>
struct simd_upsampler<float,8,2> {
    simd_t<float,8> a;
    simd_t<float,8> b;

    simd_upsampler(simd_t<float,8> t)
    {
	__m256 u = _mm256_permute_ps(t.x, 0x50);   // [ t0 t0 t1 t1 t4 t4 t5 t5 ]
	__m256 v = _mm256_permute_ps(t.x, 0xfa);   // [ t2 t2 t3 t3 t6 t6 t7 t7 ]

	a = _mm256_permute2f128_ps(u, v, 0x20);  // [ t0 t0 t1 t1 t2 t2 t3 t3 ]
	b = _mm256_permute2f128_ps(u, v, 0x31);  // [ t4 t4 t5 t5 t6 t6 t7 t7 ]
    }

    template<int N> inline simd_t<float,8> get() const;
};

template<> inline simd_t<float,8> simd_upsampler<float,8,2>::get<0> () const { return a; }
template<> inline simd_t<float,8> simd_upsampler<float,8,2>::get<1> () const { return b; }


template<>
struct simd_upsampler<int,8,2> {
    simd_t<int,8> a;
    simd_t<int,8> b;

    simd_upsampler(simd_t<int,8> t)
    {
	__m256i u = _mm256_shuffle_epi32(t.x, 0x50);   // [ t0 t0 t1 t1 t4 t4 t5 t5 ]
	__m256i v = _mm256_shuffle_epi32(t.x, 0xfa);   // [ t2 t2 t3 t3 t6 t6 t7 t7 ]

	a = _mm256_permute2f128_si256(u, v, 0x20);  // [ t0 t0 t1 t1 t2 t2 t3 t3 ]
	b = _mm256_permute2f128_si256(u, v, 0x31);  // [ t4 t4 t5 t5 t6 t6 t7 t7 ]
    }

    template<int N> inline simd_t<int,8> get() const;
};

template<> inline simd_t<int,8> simd_upsampler<int,8,2>::get<0> () const { return a; }
template<> inline simd_t<int,8> simd_upsampler<int,8,2>::get<1> () const { return b; }



// -------------------------------------------------------------------------------------------------
//
// 256-bit upsample by four


template<>
struct simd_upsampler<float,8,4> {
    simd_t<float,8> w;

    simd_upsampler(simd_t<float,8> t) 
    {
	__m256 u = _mm256_permute_ps(t.x, 0xb1);            // [ t1 t0 t3 t2 t5 t4 t7 t6 ],  0xb1 = (2301)_4
	__m256 v = _mm256_permute2f128_ps(t.x, t.x, 0x01);  // [ t4 t5 t6 t7 t0 t1 t2 t3 ]
	w = _mm256_blend_ps(u, v, 0xa5);                    // [ t4 t0 t6 t2 t5 t1 t7 t3 ],  0xa5 = (10100101)_2
    }

    template<int N> inline simd_t<float,8> get() const;
};

template<> inline simd_t<float,8> simd_upsampler<float,8,4>::get<0> () const { return _mm256_permute_ps(w.x, 0x55); }  // [ t0 t0 t0 t0 t1 t1 t1 t1 ],  (1111)_4
template<> inline simd_t<float,8> simd_upsampler<float,8,4>::get<1> () const { return _mm256_permute_ps(w.x, 0xff); }  // [ t2 t2 t2 t2 t3 t3 t3 t3 ],  (3333)_4
template<> inline simd_t<float,8> simd_upsampler<float,8,4>::get<2> () const { return _mm256_permute_ps(w.x, 0x00); }  // [ t4 t4 t4 t4 t5 t5 t5 t5 ],  (0000)_4
template<> inline simd_t<float,8> simd_upsampler<float,8,4>::get<3> () const { return _mm256_permute_ps(w.x, 0xaa); }  // [ t6 t6 t6 t6 t7 t7 t7 t7 ],  (2222)_4


template<>
struct simd_upsampler<int,8,4> {
    simd_t<int,8> w;

    simd_upsampler(simd_t<int,8> t) 
    {
	__m256i u = _mm256_shuffle_epi32(t.x, 0xb1);            // [ t1 t0 t3 t2 t5 t4 t7 t6 ],  0xb1 = (2301)_4
	__m256i v = _mm256_permute2f128_si256(t.x, t.x, 0x01);  // [ t4 t5 t6 t7 t0 t1 t2 t3 ]
	w = _mm256_blend_epi32(u, v, 0xa5);                     // [ t4 t0 t6 t2 t5 t1 t7 t3 ],  0xa5 = (10100101)_2
    }

    template<int N> inline simd_t<int,8> get() const;
};

template<> inline simd_t<int,8> simd_upsampler<int,8,4>::get<0> () const { return _mm256_shuffle_epi32(w.x, 0x55); }  // [ t0 t0 t0 t0 t1 t1 t1 t1 ],  (1111)_4
template<> inline simd_t<int,8> simd_upsampler<int,8,4>::get<1> () const { return _mm256_shuffle_epi32(w.x, 0xff); }  // [ t2 t2 t2 t2 t3 t3 t3 t3 ],  (3333)_4
template<> inline simd_t<int,8> simd_upsampler<int,8,4>::get<2> () const { return _mm256_shuffle_epi32(w.x, 0x00); }  // [ t4 t4 t4 t4 t5 t5 t5 t5 ],  (0000)_4
template<> inline simd_t<int,8> simd_upsampler<int,8,4>::get<3> () const { return _mm256_shuffle_epi32(w.x, 0xaa); }  // [ t6 t6 t6 t6 t7 t7 t7 t7 ],  (2222)_4


// -------------------------------------------------------------------------------------------------
//
// 256-bit upsample by eight


template<>
struct simd_upsampler<float,8,8> {
    simd_t<float,8> u;
    simd_t<float,8> v;

    simd_upsampler(simd_t<float,8> t) 
    {
	__m256 r = _mm256_permute2f128_ps(t.x, t.x, 0x01);  // [t1 t0]

	u = _mm256_blend_ps(t.x, r, 0xf0);  // [t0 t0]
	v = _mm256_blend_ps(t.x, r, 0x0f);  // [t1 t1]
    }

    template<int N> inline simd_t<float,8> get() const;
};

template<> inline simd_t<float,8> simd_upsampler<float,8,8>::get<0> () const { return _mm256_permute_ps(u.x, 0x00); }  // (0000)_4
template<> inline simd_t<float,8> simd_upsampler<float,8,8>::get<1> () const { return _mm256_permute_ps(u.x, 0x55); }  // (1111)_4
template<> inline simd_t<float,8> simd_upsampler<float,8,8>::get<2> () const { return _mm256_permute_ps(u.x, 0xaa); }  // (2222)_4
template<> inline simd_t<float,8> simd_upsampler<float,8,8>::get<3> () const { return _mm256_permute_ps(u.x, 0xff); }  // (3333)_4
template<> inline simd_t<float,8> simd_upsampler<float,8,8>::get<4> () const { return _mm256_permute_ps(v.x, 0x00); }
template<> inline simd_t<float,8> simd_upsampler<float,8,8>::get<5> () const { return _mm256_permute_ps(v.x, 0x55); }
template<> inline simd_t<float,8> simd_upsampler<float,8,8>::get<6> () const { return _mm256_permute_ps(v.x, 0xaa); }
template<> inline simd_t<float,8> simd_upsampler<float,8,8>::get<7> () const { return _mm256_permute_ps(v.x, 0xff); }


template<>
struct simd_upsampler<int,8,8> {
    simd_t<int,8> u;
    simd_t<int,8> v;

    simd_upsampler(simd_t<int,8> t) 
    {
#if 1
	// Fastest
	u = _mm256_permute2f128_si256(t.x, t.x, 0x00);
	v = _mm256_permute2f128_si256(t.x, t.x, 0x11);
#else
	// Equivalent but slightly slower
	__m256i r = _mm256_permute2f128_si256(t.x, t.x, 0x01);  // [t1 t0]
	u = _mm256_blend_epi32(t.x, r, 0xf0);  // [t0 t0]
	v = _mm256_blend_epi32(t.x, r, 0x0f);  // [t1 t1]
#endif
    }

    template<int N> inline simd_t<int,8> get() const;
};

template<> inline simd_t<int,8> simd_upsampler<int,8,8>::get<0> () const { return _mm256_shuffle_epi32(u.x, 0x00); }  // (0000)_4
template<> inline simd_t<int,8> simd_upsampler<int,8,8>::get<1> () const { return _mm256_shuffle_epi32(u.x, 0x55); }  // (1111)_4
template<> inline simd_t<int,8> simd_upsampler<int,8,8>::get<2> () const { return _mm256_shuffle_epi32(u.x, 0xaa); }  // (2222)_4
template<> inline simd_t<int,8> simd_upsampler<int,8,8>::get<3> () const { return _mm256_shuffle_epi32(u.x, 0xff); }  // (3333)_4
template<> inline simd_t<int,8> simd_upsampler<int,8,8>::get<4> () const { return _mm256_shuffle_epi32(v.x, 0x00); }
template<> inline simd_t<int,8> simd_upsampler<int,8,8>::get<5> () const { return _mm256_shuffle_epi32(v.x, 0x55); }
template<> inline simd_t<int,8> simd_upsampler<int,8,8>::get<6> () const { return _mm256_shuffle_epi32(v.x, 0xaa); }
template<> inline simd_t<int,8> simd_upsampler<int,8,8>::get<7> () const { return _mm256_shuffle_epi32(v.x, 0xff); }


// -------------------------------------------------------------------------------------------------
//
// Boilerplate


template<typename T, int S, int N, int D, typename std::enable_if<(N==0),int>::type = 0>
inline void _simd_upsample(simd_ntuple<T,S,N> &dst, const simd_upsampler<T,S,D> &src) { }

template<typename T, int S, int N, int D, typename std::enable_if<(N>0),int>::type = 0>
inline void _simd_upsample(simd_ntuple<T,S,N> &dst, const simd_upsampler<T,S,D> &src)
{
    _simd_upsample(dst.v, src);
    dst.x = src.template get<(N-1)> ();
}

template<typename T, int S, int N>
inline void simd_upsample(simd_ntuple<T,S,N> &dst, simd_t<T,S> x)
{
    _simd_upsample(dst, simd_upsampler<T,S,N>(x));
}
    

}  // namespace simd_helpers

#endif  // _SIMD_HELPERS_UPSAMPLE_HPP
