#ifndef _SIMD_HELPERS_FLOAT32_HPP
#define _SIMD_HELPERS_FLOAT32_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

#include "core.hpp"
#include "simd_int32.hpp"

namespace simd_helpers {
#if 0
}  // pacify emacs c-mode
#endif


// -------------------------------------------------------------------------------------------------
//
// simd_t<float,4>
//
// The member functions of simd_t<T,S> are mostly pretty intuitive, but for a few comments
// see the extended comment in simd_helpers/core.hpp


template<> struct simd_t<float,4>
{
    __m128 x;

    simd_t() { }
    simd_t(__m128 y) { x = y; }
    simd_t(float y)  { x = _mm_set1_ps(y); }

    static inline simd_t<float,4> zero()  { return _mm_setzero_ps(); }
    static inline simd_t<float,4> range() { return _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f); }

    inline void load(const float *p)  { x = _mm_load_ps(p); }
    inline void loadu(const float *p) { x = _mm_loadu_ps(p); }

    inline void store(float *p) const  { _mm_store_ps(p,x); }
    inline void storeu(float *p) const { _mm_storeu_ps(p,x); }

    template<unsigned int M> 
    inline float extract() const
    {
	union { int i; float x; } u;
	u.i = _mm_extract_ps(x, M);
	return u.x;
    }

    inline simd_t<float,4> operator+(simd_t<float,4> t) const { return x + t.x; }
    inline simd_t<float,4> operator-(simd_t<float,4> t) const { return x - t.x; }
    inline simd_t<float,4> operator*(simd_t<float,4> t) const { return x * t.x; }
    inline simd_t<float,4> operator/(simd_t<float,4> t) const { return x / t.x; }
    
    // Unary minus is implemented by flipping the sign bit.
    inline simd_t<float,4> operator-() const  { return _mm_xor_ps(_mm_set1_ps(-0.0), x); }    

    inline simd_t<float,4> &operator+=(simd_t<float,4> t) { x += t.x; return *this; }
    inline simd_t<float,4> &operator-=(simd_t<float,4> t) { x -= t.x; return *this; }
    inline simd_t<float,4> &operator*=(simd_t<float,4> t) { x *= t.x; return *this; }
    inline simd_t<float,4> &operator/=(simd_t<float,4> t) { x /= t.x; return *this; }
    
    inline simd_t<float,4> horizontal_sum() const
    {
	__m128 y = x + _mm_permute_ps(x, 0xb1);   // (2301)_4 = 0xb1
	return y + _mm_permute_ps(y, 0x4e);       // (1032)_4 = 0x4e
    }

    inline simd_t<float,4> horizontal_max() const
    {
	__m128 y = _mm_max_ps(x, _mm_permute_ps(x, 0xb1));   // (2301)_4 = 0xb1
	return _mm_max_ps(y, _mm_permute_ps(y, 0x4e));       // (1032)_4 = 0x4e
    }

    inline float sum() const { return horizontal_sum().extract<0>(); }
    inline float max() const { return horizontal_max().extract<0>(); }

    // abs() is implemented by clearing the sign bit
    inline simd_t<float,4> abs() const   { return _mm_andnot_ps(_mm_set1_ps(-0.0), x); }
    inline simd_t<float,4> sqrt() const  { return _mm_sqrt_ps(x); }

    inline simd_t<float,4> min(simd_t<float,4> t) const { return _mm_min_ps(x, t.x); }
    inline simd_t<float,4> max(simd_t<float,4> t) const { return _mm_max_ps(x, t.x); }

    inline simd_t<int,4> compare_eq(simd_t<float,4> t) const  { return _mm_castps_si128(_mm_cmpeq_ps(x, t.x)); }
    inline simd_t<int,4> compare_ne(simd_t<float,4> t) const  { return _mm_castps_si128(_mm_cmpneq_ps(x, t.x)); }
    inline simd_t<int,4> compare_ge(simd_t<float,4> t) const  { return _mm_castps_si128(_mm_cmpge_ps(x, t.x)); }
    inline simd_t<int,4> compare_gt(simd_t<float,4> t) const  { return _mm_castps_si128(_mm_cmpgt_ps(x, t.x)); }
    inline simd_t<int,4> compare_le(simd_t<float,4> t) const  { return _mm_castps_si128(_mm_cmple_ps(x, t.x)); }
    inline simd_t<int,4> compare_lt(simd_t<float,4> t) const  { return _mm_castps_si128(_mm_cmplt_ps(x, t.x)); }

    inline simd_t<float,4> apply_mask(simd_t<int,4> t) const          { return _mm_and_ps(_mm_castsi128_ps(t.x), x); }
    inline simd_t<float,4> apply_inverse_mask(simd_t<int,4> t) const  { return _mm_andnot_ps(_mm_castsi128_ps(t.x), x); }
};


// blendv(mask,a,b) is morally equivalent to (mask ? a : b)
inline simd_t<float,4> blendv(simd_t<int,4> mask, simd_t<float,4> a, simd_t<float,4> b)  { return _mm_blendv_ps(b.x, a.x, _mm_castsi128_ps(mask.x)); }


// -------------------------------------------------------------------------------------------------
//
// simd_t<float,8>


#ifdef __AVX__

template<> struct simd_t<float,8>
{
    __m256 x;

    simd_t() { }
    simd_t(__m256 y)                              { x = y; }
    simd_t(float y)                               { x = _mm256_set1_ps(y); }
    simd_t(simd_t<float,4> y, simd_t<float,4> z)  { x = _mm256_insertf128_ps(_mm256_castps128_ps256(y.x), (z.x), 1); }

    static simd_t<float,8> zero()  { return _mm256_setzero_ps(); }
    static simd_t<float,8> range() { return _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f); }

    inline void load(const float *p)  { x = _mm256_load_ps(p); }
    inline void loadu(const float *p) { x = _mm256_loadu_ps(p); }

    inline void store(float *p) const  { _mm256_store_ps(p,x); }
    inline void storeu(float *p) const { _mm256_storeu_ps(p,x); }

    template<unsigned int M> 
    inline float extract() const
    {
	simd_t<float,4> x2 = _mm256_extractf128_ps(x, M/4);
	return x2.extract<M%4> ();
    }

    template<unsigned int M> inline simd_t<float,4> extract_half() const 
    { 
	return _mm256_extractf128_ps(x,M); 
    }

    inline simd_t<float,8> operator+(simd_t<float,8> t) const { return x + t.x; }
    inline simd_t<float,8> operator-(simd_t<float,8> t) const { return x - t.x; }
    inline simd_t<float,8> operator*(simd_t<float,8> t) const { return x * t.x; }
    inline simd_t<float,8> operator/(simd_t<float,8> t) const { return x / t.x; }

    // Unary minus is implemented by flipping the sign bit
    inline simd_t<float,8> operator-() const  { return _mm256_xor_ps(_mm256_set1_ps(-0.0), x); }

    inline simd_t<float,8> &operator+=(simd_t<float,8> t) { x += t.x; return *this; }
    inline simd_t<float,8> &operator-=(simd_t<float,8> t) { x -= t.x; return *this; }
    inline simd_t<float,8> &operator*=(simd_t<float,8> t) { x *= t.x; return *this; }
    inline simd_t<float,8> &operator/=(simd_t<float,8> t) { x /= t.x; return *this; }

    // abs() is implemented by clearing the sign bit
    inline simd_t<float,8> abs() const   { return _mm256_andnot_ps(_mm256_set1_ps(-0.0), x); }
    inline simd_t<float,8> sqrt() const  { return _mm256_sqrt_ps(x); }
    inline simd_t<float,8> min(simd_t<float,8> t) const  { return _mm256_min_ps(x, t.x); }
    inline simd_t<float,8> max(simd_t<float,8> t) const  { return _mm256_max_ps(x, t.x); }

    inline simd_t<float,8> horizontal_sum() const
    {
	__m256 y = x + _mm256_permute_ps(x, 0xb1);   // (2301)_4 = 0xb1
	y += _mm256_permute_ps(y, 0x4e);             // (1032)_4 = 0x4e
	return y + _mm256_permute2f128_ps(y, y, 0x01);
    }

    inline simd_t<float,8> horizontal_max() const
    {
	__m256 y = _mm256_max_ps(x, _mm256_permute_ps(x, 0xb1));   // (2301)_4 = 0xb1
	y = _mm256_max_ps(y, _mm256_permute_ps(y, 0x4e));          // (1032)_4 = 0x4e
	return _mm256_max_ps(y, _mm256_permute2f128_ps(y, y, 0x01));
    }

    inline float sum() const { return horizontal_sum().extract<0> (); }
    inline float max() const { return horizontal_max().extract<0> (); }

    inline simd_t<int,8> compare_eq(simd_t<float,8> t) const  { return _mm256_castps_si256(_mm256_cmp_ps(x, t.x, _CMP_EQ_OQ)); }
    inline simd_t<int,8> compare_ne(simd_t<float,8> t) const  { return _mm256_castps_si256(_mm256_cmp_ps(x, t.x, _CMP_NEQ_OQ)); }
    inline simd_t<int,8> compare_ge(simd_t<float,8> t) const  { return _mm256_castps_si256(_mm256_cmp_ps(x, t.x, _CMP_GE_OQ)); }
    inline simd_t<int,8> compare_gt(simd_t<float,8> t) const  { return _mm256_castps_si256(_mm256_cmp_ps(x, t.x, _CMP_GT_OQ)); }
    inline simd_t<int,8> compare_le(simd_t<float,8> t) const  { return _mm256_castps_si256(_mm256_cmp_ps(x, t.x, _CMP_LE_OQ)); }
    inline simd_t<int,8> compare_lt(simd_t<float,8> t) const  { return _mm256_castps_si256(_mm256_cmp_ps(x, t.x, _CMP_LT_OQ)); }

    inline simd_t<float,8> apply_mask(simd_t<int,8> t) const          { return _mm256_and_ps(_mm256_castsi256_ps(t.x), x); }
    inline simd_t<float,8> apply_inverse_mask(simd_t<int,8> t) const  { return _mm256_andnot_ps(_mm256_castsi256_ps(t.x), x); }
};


// blendv(mask,a,b) is morally equivalent to (mask ? a : b)
inline simd_t<float,8> blendv(simd_t<int,8> mask, simd_t<float,8> a, simd_t<float,8> b)  { return _mm256_blendv_ps(b.x, a.x, _mm256_castsi256_ps(mask.x)); }


#endif  // __AVX__


}  // namespace simd_helpers


#endif // _SIMD_HELPERS_FLOAT32_HPP
