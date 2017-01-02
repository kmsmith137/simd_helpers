#ifndef _SIMD_HELPERS_FLOAT64_HPP
#define _SIMD_HELPERS_FLOAT64_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

#include "base.hpp"
#include "simd_int64.hpp"

namespace simd_helpers {
#if 0
}  // pacify emacs c-mode
#endif


// -------------------------------------------------------------------------------------------------
//
// simd_t<double,2>
//
// Note: for a "declaration" of the key class simd_t<T,S>,
// see the extended comment in simd_helpers/base.hpp


template<> struct simd_t<double,2>
{
    __m128d x;

    simd_t() { }
    simd_t(__m128d y)  { x = y; }
    simd_t(double y)   { x = _mm_set1_pd(y); }

    static inline simd_t<double,2> zero()  { return _mm_setzero_pd(); }
    static inline simd_t<double,2> range() { return _mm_set_pd(1.0, 0.0); }

    static inline simd_t<double,2> load(const double *p)  { return _mm_load_pd(p); }
    static inline simd_t<double,2> loadu(const double *p) { return _mm_loadu_pd(p); }

    inline void store(double *p) const  { _mm_store_pd(p,x); }
    inline void storeu(double *p) const { _mm_storeu_pd(p,x); }

    inline simd_t<double,2> &operator+=(simd_t<double,2> t) { x = _mm_add_pd(x,t.x); return *this; }
    inline simd_t<double,2> &operator-=(simd_t<double,2> t) { x = _mm_sub_pd(x,t.x); return *this; }
    inline simd_t<double,2> &operator*=(simd_t<double,2> t) { x = _mm_mul_pd(x,t.x); return *this; }
    inline simd_t<double,2> &operator/=(simd_t<double,2> t) { x = _mm_div_pd(x,t.x); return *this; }

    inline simd_t<double,2> operator+(simd_t<double,2> t) const { return _mm_add_pd(x,t.x); }
    inline simd_t<double,2> operator-(simd_t<double,2> t) const { return _mm_sub_pd(x,t.x); }
    inline simd_t<double,2> operator*(simd_t<double,2> t) const { return _mm_mul_pd(x,t.x); }
    inline simd_t<double,2> operator/(simd_t<double,2> t) const { return _mm_div_pd(x,t.x); }

    inline simd_t<double,2> min(simd_t<double,2> t) const { return _mm_min_pd(x, t.x); }
    inline simd_t<double,2> max(simd_t<double,2> t) const { return _mm_max_pd(x, t.x); }

    inline simd_t<double,2> sqrt() const  { return _mm_sqrt_pd(x); }
    inline simd_t<double,2> rsqrt() const { return _mm_set1_pd(1.0) / _mm_sqrt_pd(x); }
    
    // Fastest abs()?  (A little bit of a hack, clearing the sign bit with a bitwise operator.)
    inline simd_t<double,2> abs() const { return _mm_andnot_pd(_mm_set1_pd(-0.0), x); }

    template<unsigned int M> inline double extract() const;

    inline simd_t<double,2> horizontal_sum() const  { return x + _mm_shuffle_pd(x,x,0x01); }
    inline double sum() const                       { return _mm_cvtsd_f64(x + _mm_shuffle_pd(x,x,0x01)); }
	
};

template<> inline double simd_t<double,2>::extract<0>() const { return _mm_cvtsd_f64(x); }
template<> inline double simd_t<double,2>::extract<1>() const { return _mm_cvtsd_f64(_mm_shuffle_pd(x,x,0x01)); }


// -------------------------------------------------------------------------------------------------
//
// simd_t<double,4>


template<> struct simd_t<double,4>
{
    __m256d x;

    simd_t() { }
    simd_t(__m256d y)                               { x = y; }
    simd_t(double y)                                { x = _mm256_set1_pd(y); }
    simd_t(simd_t<double,2> y, simd_t<double,2> z)  { x = _mm256_insertf128_pd(_mm256_castpd128_pd256(y.x), (z.x), 1); }

    static simd_t<double,4> zero()  { return _mm256_setzero_pd(); }
    static simd_t<double,4> range() { return _mm256_set_pd(3.0, 2.0, 1.0, 0.0); }

    static inline simd_t<double,4> load(const double *p)  { return _mm256_load_pd(p); }
    static inline simd_t<double,4> loadu(const double *p) { return _mm256_loadu_pd(p); }

    inline void store(double *p) const  { _mm256_store_pd(p,x); }
    inline void storeu(double *p) const { _mm256_storeu_pd(p,x); }

    inline simd_t<double,4> &operator+=(simd_t<double,4> t) { x = _mm256_add_pd(x,t.x); return *this; }
    inline simd_t<double,4> &operator-=(simd_t<double,4> t) { x = _mm256_sub_pd(x,t.x); return *this; }
    inline simd_t<double,4> &operator*=(simd_t<double,4> t) { x = _mm256_mul_pd(x,t.x); return *this; }
    inline simd_t<double,4> &operator/=(simd_t<double,4> t) { x = _mm256_div_pd(x,t.x); return *this; }

    inline simd_t<double,4> operator+(simd_t<double,4> t) const { return _mm256_add_pd(x,t.x); }
    inline simd_t<double,4> operator-(simd_t<double,4> t) const { return _mm256_sub_pd(x,t.x); }
    inline simd_t<double,4> operator*(simd_t<double,4> t) const { return _mm256_mul_pd(x,t.x); }
    inline simd_t<double,4> operator/(simd_t<double,4> t) const { return _mm256_div_pd(x,t.x); }

    inline simd_t<double,4> min(simd_t<double,4> t) const { return _mm256_min_pd(x, t.x); }
    inline simd_t<double,4> max(simd_t<double,4> t) const { return _mm256_max_pd(x, t.x); }

    inline simd_t<double,4> sqrt() const  { return _mm256_sqrt_pd(x); }
    inline simd_t<double,4> rsqrt() const { return _mm256_set1_pd(1.0) / _mm256_sqrt_pd(x); }

    // Fastest abs()?  (A little bit of a hack, clearing the sign bit with a bitwise operator.)
    inline simd_t<double,4> abs() const { return _mm256_andnot_pd(_mm256_set1_pd(-0.0), x); }

    template<unsigned int M> 
    inline double extract() const
    {
	simd_t<double,2> x2 = _mm256_extractf128_pd(x, M/2);
	return x2.extract<M%2> ();
    }

    inline simd_t<double,4> horizontal_sum() const
    { 
	__m256d y = x + _mm256_shuffle_pd(x, x, 0x05);
        return y + _mm256_permute2f128_pd(y, y, 0x01);
    }

    inline double sum() const
    { 
	simd_t<double,2> y = _mm256_extractf128_pd(x,0) + _mm256_extractf128_pd(x,1);
	return y.sum();
    }
};


// -------------------------------------------------------------------------------------------------


// blendv(mask,a,b) is morally equivalent to (mask ? a : b)
inline simd_t<double,2> blendv(simd_t<int64_t,2> mask, simd_t<double,2> a, simd_t<double,2> b)  { return _mm_blendv_pd(b.x, a.x, _mm_castsi128_pd(mask.x)); }
inline simd_t<double,4> blendv(simd_t<int64_t,4> mask, simd_t<double,4> a, simd_t<double,4> b)  { return _mm256_blendv_pd(b.x, a.x, _mm256_castsi256_pd(mask.x)); }


}  // namespace simd_helpers


#endif // _SIMD_HELPERS_FLOAT64_HPP
