#ifndef _SIMD_HELPERS_INT64_HPP
#define _SIMD_HELPERS_INT64_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

#include "base.hpp"

namespace simd_helpers {
#if 0
}  // pacify emacs c-mode
#endif


// -------------------------------------------------------------------------------------------------
//
// simd_t<int64_t,2>
//
// Note: for a "declaration" of the key class simd_t<T,S>,
// see the extended comment in simd_helpers/base.hpp


template<> struct simd_t<int64_t,2>
{
    __m128i x;

    simd_t() { }
    simd_t(__m128i y) { x = y; }
    simd_t(int64_t y) { x = _mm_set1_epi64x(y); }

    static inline simd_t<int64_t,2> zero()  { return _mm_setzero_si128(); }
    static inline simd_t<int64_t,2> range() { return _mm_set_epi64x(1, 0); }

    static inline simd_t<int64_t,2> load(const int64_t *p)  { return _mm_load_si128((const __m128i *) p); }
    static inline simd_t<int64_t,2> loadu(const int64_t *p) { return _mm_loadu_si128((const __m128i *) p); }

    inline void store(int64_t *p) const  { _mm_store_si128((__m128i *)p, x); }
    inline void storeu(int64_t *p) const { _mm_storeu_si128((__m128i *)p, x); }

    inline simd_t<int64_t,2> &operator+=(simd_t<int64_t,2> t) { x = _mm_add_epi64(x,t.x); return *this; }
    inline simd_t<int64_t,2> &operator-=(simd_t<int64_t,2> t) { x = _mm_sub_epi64(x,t.x); return *this; }

    inline simd_t<int64_t,2> operator+(simd_t<int64_t,2> t) const { return _mm_add_epi64(x,t.x); }
    inline simd_t<int64_t,2> operator-(simd_t<int64_t,2> t) const { return _mm_sub_epi64(x,t.x); }

    // Note: you might need to call this with the weird-looking syntax
    //    x.template extract<M> ();
    template<unsigned int M> inline int64_t extract() const  { return _mm_extract_epi64(x, M); }
};


// -------------------------------------------------------------------------------------------------
//
// simd_t<int64_t,4>


template<> struct simd_t<int64_t,4>
{
    __m256i x;

    simd_t() { }
    simd_t(__m256i y)                                 { x = y; }
    simd_t(int64_t y)                                 { x = _mm256_set1_epi64x(y); }
    simd_t(simd_t<int64_t,2> y, simd_t<int64_t,2> z)  { x = _mm256_insertf128_si256(_mm256_castsi128_si256(y.x), (z.x), 1); }

    static inline simd_t<int64_t,4> zero()  { return _mm256_setzero_si256(); }
    static inline simd_t<int64_t,4> range() { return _mm256_set_epi64x(3, 2, 1, 0); }

    static inline simd_t<int64_t,4> load(const int64_t *p)  { return _mm256_load_si256((const __m256i *) p); }
    static inline simd_t<int64_t,4> loadu(const int64_t *p) { return _mm256_loadu_si256((const __m256i *) p); }

    inline void store(int64_t *p) const  { _mm256_store_si256((__m256i *)p, x); }
    inline void storeu(int64_t *p) const { _mm256_storeu_si256((__m256i *)p, x); }

    inline simd_t<int64_t,4> &operator+=(simd_t<int64_t,4> t) { x = _mm256_add_epi64(x,t.x); return *this; }
    inline simd_t<int64_t,4> &operator-=(simd_t<int64_t,4> t) { x = _mm256_sub_epi64(x,t.x); return *this; }

    inline simd_t<int64_t,4> operator+(simd_t<int64_t,4> t) const { return _mm256_add_epi64(x,t.x); }
    inline simd_t<int64_t,4> operator-(simd_t<int64_t,4> t) const { return _mm256_sub_epi64(x,t.x); }

    template<unsigned int M> inline int extract() const                    { return _mm256_extract_epi64(x,M); }
    template<unsigned int M> inline simd_t<int64_t,2> extract128() const   { return _mm256_extractf128_si256(x,M); }
};


}  // namespace simd_helpers


#endif // _SIMD_HELPERS_INT64_HPP
