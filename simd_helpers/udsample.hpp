#ifndef _SIMD_HELPERS_UDSAMPLE_HPP
#define _SIMD_HELPERS_UDSAMPLE_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

#include "simd_t.hpp"
#include "simd_ntuple.hpp"

namespace simd_helpers {
#if 0
}  // pacify emacs c-mode
#endif


// -------------------------------------------------------------------------------------------------
//
// template<typename T, unsigned int S, unsigned int N>
// inline simd_t<T,S> downsample(const simd_ntuple<T,S,N> &v)
//
// Note: the downsampling kernel does not divide the result by N !!


inline __m128 _kernel128_downsample2(__m128 a, __m128 b)
{
    __m128 u = _mm_shuffle_ps(a, b, 0x88);   // [a0 a2 b0 b2],  0x88 = (2020)_4
    __m128 v = _mm_shuffle_ps(a, b, 0xdd);   // [a1 a3 b1 b3],  0xdd = (3131)_4
    return u + v;

}

inline __m128 _kernel128_downsample4(__m128 a, __m128 b, __m128 c, __m128 d)
{
    // I think this is fastest.
    __m128 u = _kernel128_downsample2(a, b);
    __m128 v = _kernel128_downsample2(c, d);
    return _kernel128_downsample2(u, v);
}

inline __m256 _kernel256_downsample2(__m256 a, __m256 b)
{
    // Similar to _kernel128_downsample2(), but with an extra 64-bit swap
    // which in theory helps the subsequent permute/blend block pipeline.
    // (Not sure if this actually helps!)

    __m256 u = _mm256_shuffle_ps(b, a, 0x88);   // [b0 b2 a0 a2],  0x88 = (2020)_4
    __m256 v = _mm256_shuffle_ps(b, a, 0xdd);   // [b1 b3 a1 a3],  0xdd = (3131)_4
    __m256 w = u + v;

    // Now we have a vector [ w2 w0 w3 w1 ], where w_i is 64 bits
    // and we want to rearrange to [ w0 w1 w2 w3 ].

    __m256 x = _mm256_permute_ps(w, 0x4e);          // [ w0 w2 w1 w3 ],  0x4e = (1032)_4
    __m256 y = _mm256_permute2f128_ps(w, w, 0x01);  // [ w3 w1 w2 w0 ]

    return _mm256_blend_ps(x, y, 0x3c);   // (00111100)_2
}

inline __m256 _kernel256_downsample4(__m256 a, __m256 b, __m256 c, __m256 d)
{
    // Until the very end of this routine, all operations are happening
    // in sync on the two 128-bit halves of the 256-bit register.  
    // Thus we denote a = [a0 a1 a2 a3], and analogously for b,c,d.

    // Combine a and c, to obtain [ a01 a23 c01 c23 ]
    // The logic here is the same as _kernel128_downsample2().
    __m256 ac = _mm256_shuffle_ps(a, c, 0x88) + _mm256_shuffle_ps(a, c, 0xdd);

    // Combine b and d, to obtain [ b01 b23 d01 d23 ]
    __m256 bd = _mm256_shuffle_ps(b, d, 0x88) + _mm256_shuffle_ps(b, d, 0xdd);

    __m256 u = _mm256_shuffle_ps(ac, bd, 0x22);  // [ c01 a01 d01 b01 ],  0x22 = (0202)_4
    __m256 v = _mm256_shuffle_ps(ac, bd, 0x77);  // [ c23 a23 d23 b23 ],  0x77 = (1313)_4
    __m256 w = u + v;                            // [ c a d b ]

    // The 256-bit vector w is the output we want, but it has ordering [ w4 w0 w6 w2 w5 w1 w7 w3 ].

    __m256 x = _mm256_permute_ps(w, 0xb1);           // [ w0 w4 w2 w6 w1 w5 w3 w7 ],  0xb1 = (2301)_4
    __m256 y = _mm256_permute2f128_ps(w, w, 0x01);   // [ w5 w1 w7 w3 w4 w0 w6 w2 ]
    
    return _mm256_blend_ps(x, y, 0x5a);   // (01011010)_2
}

// Helper for _kernel256_downsample8().
inline __m256 _kernel256_downsample8a(__m256 a, __m256 b, __m256 c, __m256 d)
{
    __m256 ab = _mm256_shuffle_ps(a, b, 0x88) + _mm256_shuffle_ps(a, b, 0xdd);
    __m256 cd = _mm256_shuffle_ps(c, d, 0x88) + _mm256_shuffle_ps(c, d, 0xdd);
    return _mm256_shuffle_ps(ab, cd, 0x88) + _mm256_shuffle_ps(ab, cd, 0xdd);
    
}

inline __m256 _kernel256_downsample8(__m256 a, __m256 b, __m256 c, __m256 d, __m256 e, __m256 f, __m256 g, __m256 h)
{
    __m256 abcd = _kernel256_downsample8a(a, b, c, d);    // [ a0 b0 c0 d0 a1 b1 c1 d1 ]
    __m256 efgh = _kernel256_downsample8a(e, f, g, h);    // [ e0 f0 g0 h0 e1 f1 g1 h1 ]

    __m256 u = _mm256_blend_ps(abcd, efgh, 0xf0);         // [ a0 b0 c0 d0 e1 f1 g1 h1 ],  0xf0 = (11110000)_2
    __m256 v = _mm256_permute2f128_ps(abcd, efgh, 0x21);  // [ a1 b1 c1 d1 e0 f0 g0 h0 ]
    return u + v;
}


inline void _kernel128_upsample2(__m128 &a, __m128 &b, __m128 t)
{
    a = _mm_permute_ps(t, 0x50);  // (1100)_4
    b = _mm_permute_ps(t, 0xfa);  // (3322)_4
}

inline void _kernel128_upsample4(__m128 &a, __m128 &b, __m128 &c, __m128 &d, __m128 t)
{
    a = _mm_permute_ps(t, 0x00);  // (0000)_4
    b = _mm_permute_ps(t, 0x55);  // (1111)_4
    c = _mm_permute_ps(t, 0xaa);  // (2222)_4
    d = _mm_permute_ps(t, 0xff);  // (3333)_4
}

inline void _kernel256_upsample2(__m256 &a, __m256 &b, __m256 t)
{
    // Is this fastest?
    __m256 u = _mm256_permute_ps(t, 0x50);   // [ t0 t0 t1 t1 t4 t4 t5 t5 ]
    __m256 v = _mm256_permute_ps(t, 0xfa);   // [ t2 t2 t3 t3 t6 t6 t7 t7 ]

    a = _mm256_permute2f128_ps(u, v, 0x20);  // [ t0 t0 t1 t1 t2 t2 t3 t3 ]
    b = _mm256_permute2f128_ps(u, v, 0x31);  // [ t4 t4 t5 t5 t6 t6 t7 t7 ]
}

inline void _kernel256_upsample4(__m256 &a, __m256 &b, __m256 &c, __m256 &d, __m256 t)
{
    __m256 u = _mm256_permute_ps(t, 0xb1);          // [ t1 t0 t3 t2 t5 t4 t7 t6 ],  0xb1 = (2301)_4
    __m256 v = _mm256_permute2f128_ps(t, t, 0x01);  // [ t4 t5 t6 t7 t0 t1 t2 t3 ]
    __m256 w = _mm256_blend_ps(u, v, 0xa5);         // [ t4 t0 t6 t2 t5 t1 t7 t3 ],  0xa5 = (10100101)_2

    a = _mm256_permute_ps(w, 0x55);  // [ t0 t0 t0 t0 t1 t1 t1 t1 ],  (1111)_4
    b = _mm256_permute_ps(w, 0xff);  // [ t2 t2 t2 t2 t3 t3 t3 t3 ],  (3333)_4
    c = _mm256_permute_ps(w, 0x00);  // [ t4 t4 t4 t4 t5 t5 t5 t5 ],  (0000)_4
    d = _mm256_permute_ps(w, 0xaa);  // [ t6 t6 t6 t6 t7 t7 t7 t7 ],  (2222)_4
}

inline void _kernel256_upsample8(__m256 &a, __m256 &b, __m256 &c, __m256 &d, __m256 &e, __m256 &f, __m256 &g, __m256 &h, __m256 t)
{
    __m256 u = _mm256_permute2f128_ps(t, t, 0x00);
    __m256 v = _mm256_permute2f128_ps(t, t, 0x11);

    a = _mm256_permute_ps(u, 0x00);  // (0000)_4
    b = _mm256_permute_ps(u, 0x55);  // (1111)_4
    c = _mm256_permute_ps(u, 0xaa);  // (2222)_4
    d = _mm256_permute_ps(u, 0xff);  // (3333)_4

    e = _mm256_permute_ps(v, 0x00);
    f = _mm256_permute_ps(v, 0x55);
    g = _mm256_permute_ps(v, 0xaa);
    h = _mm256_permute_ps(v, 0xff);
}


// -------------------------------------------------------------------------------------------------


inline simd_t<float,4> downsample(const simd_ntuple<float,4,2> &t)
{
    return _kernel128_downsample2(t.extract<0>().x, 
				  t.extract<1>().x);
}


inline simd_t<float,4> downsample(const simd_ntuple<float,4,4> &t)
{
    return _kernel128_downsample4(t.extract<0>().x, 
				  t.extract<1>().x, 
				  t.extract<2>().x, 
				  t.extract<3>().x);
}


inline simd_t<float,8> downsample(const simd_ntuple<float,8,2> &t)
{
    return _kernel256_downsample2(t.extract<0>().x, 
				  t.extract<1>().x);
}


inline simd_t<float,8> downsample(const simd_ntuple<float,8,4> &t)
{
    return _kernel256_downsample4(t.extract<0>().x, 
				  t.extract<1>().x, 
				  t.extract<2>().x, 
				  t.extract<3>().x);
}


inline simd_t<float,8> downsample(const simd_ntuple<float,8,8> &t)
{
    return _kernel256_downsample8(t.extract<0>().x, 
				  t.extract<1>().x, 
				  t.extract<2>().x, 
				  t.extract<3>().x,
				  t.extract<4>().x, 
				  t.extract<5>().x, 
				  t.extract<6>().x, 
				  t.extract<7>().x);
}


inline void upsample(simd_ntuple<float,4,2> &out, simd_t<float,4> t)
{
    _kernel128_upsample2(out.extract<0>().x, 
			 out.extract<1>().x, 
			 t.x);
}


inline void upsample(simd_ntuple<float,4,4> &out, simd_t<float,4> t)
{
    _kernel128_upsample4(out.extract<0>().x, 
			 out.extract<1>().x, 
			 out.extract<2>().x, 
			 out.extract<3>().x, t.x);
}


inline void upsample(simd_ntuple<float,8,2> &out, simd_t<float,8> t)
{
    _kernel256_upsample2(out.extract<0>().x, 
			 out.extract<1>().x, t.x);
}


inline void upsample(simd_ntuple<float,8,4> &out, simd_t<float,8> t)
{
    _kernel256_upsample4(out.extract<0>().x, 
			 out.extract<1>().x, 
			 out.extract<2>().x, 
			 out.extract<3>().x, t.x);
}


inline void upsample(simd_ntuple<float,8,8> &out, simd_t<float,8> t)
{
    _kernel256_upsample8(out.extract<0>().x, 
			 out.extract<1>().x, 
			 out.extract<2>().x, 
			 out.extract<3>().x,
			 out.extract<4>().x, 
			 out.extract<5>().x, 
			 out.extract<6>().x, 
			 out.extract<7>().x, t.x);
}


// -------------------------------------------------------------------------------------------------
//
// Integer versions of the upsample-type routines.


inline void _kernel128i_upsample2(__m128i &a, __m128i &b, __m128i t)
{
    a = _mm_shuffle_epi32(t, 0x50);  // (1100)_4
    b = _mm_shuffle_epi32(t, 0xfa);  // (3322)_4
}

inline void _kernel128i_upsample4(__m128i &a, __m128i &b, __m128i &c, __m128i &d, __m128i t)
{
    a = _mm_shuffle_epi32(t, 0x00);  // (0000)_4
    b = _mm_shuffle_epi32(t, 0x55);  // (1111)_4
    c = _mm_shuffle_epi32(t, 0xaa);  // (2222)_4
    d = _mm_shuffle_epi32(t, 0xff);  // (3333)_4
}

inline void _kernel256i_upsample2(__m256i &a, __m256i &b, __m256i t)
{
    __m256i u = _mm256_shuffle_epi32(t, 0x50);   // [ t0 t0 t1 t1 t4 t4 t5 t5 ]
    __m256i v = _mm256_shuffle_epi32(t, 0xfa);   // [ t2 t2 t3 t3 t6 t6 t7 t7 ]

    a = _mm256_permute2f128_si256(u, v, 0x20);  // [ t0 t0 t1 t1 t2 t2 t3 t3 ]
    b = _mm256_permute2f128_si256(u, v, 0x31);  // [ t4 t4 t5 t5 t6 t6 t7 t7 ]
}

inline void _kernel256i_upsample4(__m256i &a, __m256i &b, __m256i &c, __m256i &d, __m256i t)
{
    __m256i u = _mm256_shuffle_epi32(t, 0xb1);          // [ t1 t0 t3 t2 t5 t4 t7 t6 ],  0xb1 = (2301)_4
    __m256i v = _mm256_permute2f128_si256(t, t, 0x01);  // [ t4 t5 t6 t7 t0 t1 t2 t3 ]
    __m256i w = _mm256_blend_epi32(u, v, 0xa5);         // [ t4 t0 t6 t2 t5 t1 t7 t3 ],  0xa5 = (10100101)_2

    a = _mm256_shuffle_epi32(w, 0x55);  // [ t0 t0 t0 t0 t1 t1 t1 t1 ],  (1111)_4
    b = _mm256_shuffle_epi32(w, 0xff);  // [ t2 t2 t2 t2 t3 t3 t3 t3 ],  (3333)_4
    c = _mm256_shuffle_epi32(w, 0x00);  // [ t4 t4 t4 t4 t5 t5 t5 t5 ],  (0000)_4
    d = _mm256_shuffle_epi32(w, 0xaa);  // [ t6 t6 t6 t6 t7 t7 t7 t7 ],  (2222)_4
}

inline void _kernel256i_upsample8(__m256i &a, __m256i &b, __m256i &c, __m256i &d, __m256i &e, __m256i &f, __m256i &g, __m256i &h, __m256i t)
{
    __m256i u = _mm256_permute2f128_si256(t, t, 0x00);
    __m256i v = _mm256_permute2f128_si256(t, t, 0x11);

    a = _mm256_shuffle_epi32(u, 0x00);  // (0000)_4
    b = _mm256_shuffle_epi32(u, 0x55);  // (1111)_4
    c = _mm256_shuffle_epi32(u, 0xaa);  // (2222)_4
    d = _mm256_shuffle_epi32(u, 0xff);  // (3333)_4

    e = _mm256_shuffle_epi32(v, 0x00);
    f = _mm256_shuffle_epi32(v, 0x55);
    g = _mm256_shuffle_epi32(v, 0xaa);
    h = _mm256_shuffle_epi32(v, 0xff);
}


inline void upsample(simd_ntuple<int,4,2> &out, simd_t<int,4> t)
{
    _kernel128i_upsample2(out.extract<0>().x, 
			  out.extract<1>().x, 
			  t.x);
}


inline void upsample(simd_ntuple<int,4,4> &out, simd_t<int,4> t)
{
    _kernel128i_upsample4(out.extract<0>().x, 
			  out.extract<1>().x, 
			  out.extract<2>().x, 
			  out.extract<3>().x, t.x);
}


inline void upsample(simd_ntuple<int,8,2> &out, simd_t<int,8> t)
{
    _kernel256i_upsample2(out.extract<0>().x, 
			  out.extract<1>().x, t.x);
}


inline void upsample(simd_ntuple<int,8,4> &out, simd_t<int,8> t)
{
    _kernel256i_upsample4(out.extract<0>().x, 
			  out.extract<1>().x, 
			  out.extract<2>().x, 
			  out.extract<3>().x, t.x);
}


inline void upsample(simd_ntuple<int,8,8> &out, simd_t<int,8> t)
{
    _kernel256i_upsample8(out.extract<0>().x, 
			  out.extract<1>().x, 
			  out.extract<2>().x, 
			  out.extract<3>().x,
			  out.extract<4>().x, 
			  out.extract<5>().x, 
			  out.extract<6>().x, 
			  out.extract<7>().x, t.x);
}


}  // namespace simd_helpers

#endif  // _SIMD_HELPERS_UDSAMPLE_HPP
