#ifndef _SIMD_HELPERS_SIMD_NTUPLE_HPP
#define _SIMD_HELPERS_SIMD_NTUPLE_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

#include "core.hpp"
#include <type_traits>

namespace simd_helpers {
#if 0
}  // pacify emacs c-mode
#endif


// N-tuple of simd-vectors
template<typename T, int S, int N> struct simd_ntuple;

// N-tuple of mask vectors
template<typename T, int S, int N> 
using smask_ntuple = simd_ntuple<smask_t<T>,S,N>;


template<typename T, int S, int N>
struct simd_ntuple
{
    using scalar_type = T;
    using iscalar_type = typename simd_t<T,S>::iscalar_type;

    static constexpr int simd_size = S;
    static constexpr int total_size = S*N;

    simd_ntuple<T,S,N-1> v;
    simd_t<T,S> x;

    simd_ntuple() { }
    simd_ntuple(const simd_ntuple<T,S,N-1> &v_, simd_t<T,S> x_) : v(v_), x(x_) { }
    
    explicit simd_ntuple(const simd_t<T,S> &x_) : v(x_), x(x_) { }

    inline void setzero()
    {
	v.setzero();
	x = simd_t<T,S>::zero();
    }

    inline void loadu(const T *p)
    {
	v.loadu(p);
	x.loadu(p+(N-1)*S);
    }

    inline void storeu(T *p) const
    {
	v.storeu(p);
	x.storeu(p+(N-1)*S);
    }

    template<int M>
    inline void vextract(T *p) const
    {
        v.template vextract<M>(p);
        p[N-1] = x.template extract<M>();
    }

    // internal, used by vextract_all
    template<int M,
             typename std::enable_if<(M == 0),int>::type = 0>
    inline void _vextract_n(T *p) const {
        vextract<0>(p);
    }
    template<int M,
             typename std::enable_if<(M > 0) && (M < S),int>::type = 0>
    inline void _vextract_n(T *p) const {
        _vextract_n<M-1>(p);
        vextract<M>(p + N*M);
    }

    // Equivalent to a storeu followed by a transpose
    inline void vextract_all(T *p) const
    {
        _vextract_n<S-1>(p);
    }

    template<int M, typename std::enable_if<(M == N-1),int>::type = 0>
    inline simd_t<T,S> extract() const { return x; }

    template<int M, typename std::enable_if<(M < N-1),int>::type = 0>
    inline simd_t<T,S> extract() const { return v.template extract<M>(); }

    // Non-const version of extract<>()
    template<int M, typename std::enable_if<(M == N-1),int>::type = 0>
    inline simd_t<T,S> &extract() { return x; }

    // Non-const version of extract<>()
    template<int M, typename std::enable_if<(M < N-1),int>::type = 0>
    inline simd_t<T,S> &extract() { return v.template extract<M>(); }

    inline simd_ntuple<T,S,N> &operator+=(const simd_ntuple<T,S,N> &t)  { v += t.v; x += t.x; return *this; }
    inline simd_ntuple<T,S,N> &operator-=(const simd_ntuple<T,S,N> &t)  { v -= t.v; x -= t.x; return *this; }
    inline simd_ntuple<T,S,N> &operator*=(const simd_ntuple<T,S,N> &t)  { v *= t.v; x *= t.x; return *this; }
    inline simd_ntuple<T,S,N> &operator/=(const simd_ntuple<T,S,N> &t)  { v /= t.v; x /= t.x; return *this; }
    inline simd_ntuple<T,S,N> &operator&=(const simd_ntuple<T,S,N> &t)  { v &= t.v; x &= t.x; return *this; }
    inline simd_ntuple<T,S,N> &operator|=(const simd_ntuple<T,S,N> &t)  { v |= t.v; x |= t.x; return *this; }
    inline simd_ntuple<T,S,N> &operator^=(const simd_ntuple<T,S,N> &t)  { v ^= t.v; x ^= t.x; return *this; }

    inline simd_ntuple<T,S,N> operator+(const simd_ntuple<T,S,N> &t) const  { simd_ntuple<T,S,N> ret; ret.v = v+t.v; ret.x = x+t.x; return ret; }
    inline simd_ntuple<T,S,N> operator-(const simd_ntuple<T,S,N> &t) const  { simd_ntuple<T,S,N> ret; ret.v = v-t.v; ret.x = x-t.x; return ret; }
    inline simd_ntuple<T,S,N> operator*(const simd_ntuple<T,S,N> &t) const  { simd_ntuple<T,S,N> ret; ret.v = v*t.v; ret.x = x*t.x; return ret; }
    inline simd_ntuple<T,S,N> operator/(const simd_ntuple<T,S,N> &t) const  { simd_ntuple<T,S,N> ret; ret.v = v/t.v; ret.x = x/t.x; return ret; }
    inline simd_ntuple<T,S,N> operator&(const simd_ntuple<T,S,N> &t) const  { simd_ntuple<T,S,N> ret; ret.v = v&t.v; ret.x = x&t.x; return ret; }
    inline simd_ntuple<T,S,N> operator|(const simd_ntuple<T,S,N> &t) const  { simd_ntuple<T,S,N> ret; ret.v = v|t.v; ret.x = x|t.x; return ret; }
    inline simd_ntuple<T,S,N> operator^(const simd_ntuple<T,S,N> &t) const  { simd_ntuple<T,S,N> ret; ret.v = v^t.v; ret.x = x^t.x; return ret; }

    inline simd_ntuple<T,S,N> &operator+=(const simd_t<T,S> &t)  { v += t; x += t; return *this; }
    inline simd_ntuple<T,S,N> &operator-=(const simd_t<T,S> &t)  { v -= t; x -= t; return *this; }
    inline simd_ntuple<T,S,N> &operator*=(const simd_t<T,S> &t)  { v *= t; x *= t; return *this; }
    inline simd_ntuple<T,S,N> &operator/=(const simd_t<T,S> &t)  { v /= t; x /= t; return *this; }
    inline simd_ntuple<T,S,N> &operator&=(const simd_t<T,S> &t)  { v &= t; x &= t; return *this; }
    inline simd_ntuple<T,S,N> &operator|=(const simd_t<T,S> &t)  { v |= t; x |= t; return *this; }
    inline simd_ntuple<T,S,N> &operator^=(const simd_t<T,S> &t)  { v ^= t; x ^= t; return *this; }

    inline simd_ntuple<T,S,N> operator+(const simd_t<T,S> &t) const { simd_ntuple<T,S,N> ret; ret.v = v+t; ret.x = x+t; return ret; }
    inline simd_ntuple<T,S,N> operator-(const simd_t<T,S> &t) const { simd_ntuple<T,S,N> ret; ret.v = v-t; ret.x = x-t; return ret; }
    inline simd_ntuple<T,S,N> operator*(const simd_t<T,S> &t) const { simd_ntuple<T,S,N> ret; ret.v = v*t; ret.x = x*t; return ret; }
    inline simd_ntuple<T,S,N> operator/(const simd_t<T,S> &t) const { simd_ntuple<T,S,N> ret; ret.v = v/t; ret.x = x/t; return ret; }
    inline simd_ntuple<T,S,N> operator&(const simd_t<T,S> &t) const { simd_ntuple<T,S,N> ret; ret.v = v&t; ret.x = x&t; return ret; }
    inline simd_ntuple<T,S,N> operator|(const simd_t<T,S> &t) const { simd_ntuple<T,S,N> ret; ret.v = v|t; ret.x = x|t; return ret; }
    inline simd_ntuple<T,S,N> operator^(const simd_t<T,S> &t) const { simd_ntuple<T,S,N> ret; ret.v = v^t; ret.x = x^t; return ret; }

    inline simd_ntuple<T,S,N> apply_mask(const smask_t<T,S> &t) const          { simd_ntuple<T,S,N> ret; ret.v = v.apply_mask(t); ret.x = x.apply_mask(t); return ret; }
    inline simd_ntuple<T,S,N> apply_inverse_mask(const smask_t<T,S> &t) const  { simd_ntuple<T,S,N> ret; ret.v = v.apply_inverse_mask(t); ret.x = x.apply_inverse_mask(t); return ret; }
    
    inline simd_ntuple<T,S,N> _rsub(const simd_t<T,S> &t)  { simd_ntuple<T,S,N> ret; ret.v = t-v; ret.x = t-x; return ret; }
    inline simd_ntuple<T,S,N> _rdiv(const simd_t<T,S> &t)  { simd_ntuple<T,S,N> ret; ret.v = t/v; ret.x = t/x; return ret; }

    // vertical_sum(): returns elementwise sum of all N simd_t's
    inline simd_t<T,S> _vertical_sum(simd_t<T,S> u) const { return v._vertical_sum(u+x); }
    inline simd_t<T,S> vertical_sum() const { return v._vertical_sum(x); }

    // ...and analogously for vertical_max(), etc.
    inline simd_t<T,S> _vertical_max(simd_t<T,S> u) const { return v._vertical_max(u.max(x)); }
    inline simd_t<T,S> vertical_max() const { return v._vertical_max(x); }

    inline simd_t<T,S> _vertical_xor(simd_t<T,S> u) const { return v._vertical_xor(u ^ x); }
    inline simd_t<T,S> vertical_xor() const { return v._vertical_xor(x); }

    // vertical_dot(): returns elementwise length-N dot product of simd_t's
    inline simd_t<T,S> _vertical_dot(const simd_ntuple<T,S,N> &t, simd_t<T,S> u) const  { return v._vertical_dot(t.v, u + x*t.x); }
    inline simd_t<T,S> _vertical_dotn(const simd_ntuple<T,S,N> &t, simd_t<T,S> u) const { return v._vertical_dotn(t.v, u - x*t.x); }
    inline simd_t<T,S> vertical_dot(const simd_ntuple<T,S,N> &t) const  { return v._vertical_dot(t.v, x*t.x); }

    inline void horizontal_sum_in_place()
    {
	v.horizontal_sum_in_place();
	x = x.horizontal_sum();
    }

    inline void max_in_place(const simd_ntuple<T,S,N> &t)
    {
	v.max_in_place(t.v);
	x = x.max(t.x);
    }

    // sum(): returns sum of all scalars in the simd_ntuple
    inline T sum() const { return this->vertical_sum().sum(); }
};


template<typename T, int S> 
struct simd_ntuple<T,S,0> 
{ 
    using scalar_type = T;
    using iscalar_type = typename simd_t<T,S>::iscalar_type;

    static constexpr int simd_size = S;
    static constexpr int total_size = 0;

    inline void setzero() { }
    inline void loadu(const T *p) { }
    inline void storeu(T *p) const { }

    template<int M>
    inline void vextract(T *p) const { }

    inline void vextract_all(T *p) const { }

    inline simd_ntuple<T,S,0> &operator+=(const simd_ntuple<T,S,0> &t) { return *this; }
    inline simd_ntuple<T,S,0> &operator-=(const simd_ntuple<T,S,0> &t) { return *this; }
    inline simd_ntuple<T,S,0> &operator*=(const simd_ntuple<T,S,0> &t) { return *this; }
    inline simd_ntuple<T,S,0> &operator/=(const simd_ntuple<T,S,0> &t) { return *this; }
    inline simd_ntuple<T,S,0> &operator&=(const simd_ntuple<T,S,0> &t) { return *this; }
    inline simd_ntuple<T,S,0> &operator|=(const simd_ntuple<T,S,0> &t) { return *this; }
    inline simd_ntuple<T,S,0> &operator^=(const simd_ntuple<T,S,0> &t) { return *this; }

    inline simd_ntuple<T,S,0> operator+(const simd_ntuple<T,S,0> &t) const { return simd_ntuple<T,S,0>(); }
    inline simd_ntuple<T,S,0> operator-(const simd_ntuple<T,S,0> &t) const { return simd_ntuple<T,S,0>(); }
    inline simd_ntuple<T,S,0> operator*(const simd_ntuple<T,S,0> &t) const { return simd_ntuple<T,S,0>(); }
    inline simd_ntuple<T,S,0> operator/(const simd_ntuple<T,S,0> &t) const { return simd_ntuple<T,S,0>(); }
    inline simd_ntuple<T,S,0> operator&(const simd_ntuple<T,S,0> &t) const { return simd_ntuple<T,S,0>(); }
    inline simd_ntuple<T,S,0> operator|(const simd_ntuple<T,S,0> &t) const { return simd_ntuple<T,S,0>(); }
    inline simd_ntuple<T,S,0> operator^(const simd_ntuple<T,S,0> &t) const { return simd_ntuple<T,S,0>(); }

    inline simd_ntuple<T,S,0> &operator+=(const simd_t<T,S> &t) { return *this; }
    inline simd_ntuple<T,S,0> &operator-=(const simd_t<T,S> &t) { return *this; }
    inline simd_ntuple<T,S,0> &operator*=(const simd_t<T,S> &t) { return *this; }
    inline simd_ntuple<T,S,0> &operator/=(const simd_t<T,S> &t) { return *this; }
    inline simd_ntuple<T,S,0> &operator&=(const simd_t<T,S> &t) { return *this; }
    inline simd_ntuple<T,S,0> &operator|=(const simd_t<T,S> &t) { return *this; }
    inline simd_ntuple<T,S,0> &operator^=(const simd_t<T,S> &t) { return *this; }

    inline simd_ntuple<T,S,0> operator+(const simd_t<T,S> &t) const { return simd_ntuple<T,S,0>(); }
    inline simd_ntuple<T,S,0> operator-(const simd_t<T,S> &t) const { return simd_ntuple<T,S,0>(); }
    inline simd_ntuple<T,S,0> operator*(const simd_t<T,S> &t) const { return simd_ntuple<T,S,0>(); }
    inline simd_ntuple<T,S,0> operator/(const simd_t<T,S> &t) const { return simd_ntuple<T,S,0>(); }
    inline simd_ntuple<T,S,0> operator&(const simd_t<T,S> &t) const { return simd_ntuple<T,S,0>(); }
    inline simd_ntuple<T,S,0> operator|(const simd_t<T,S> &t) const { return simd_ntuple<T,S,0>(); }
    inline simd_ntuple<T,S,0> operator^(const simd_t<T,S> &t) const { return simd_ntuple<T,S,0>(); }
    
    inline simd_ntuple<T,S,0> apply_mask(const smask_t<T,S> &t) const          { return simd_ntuple<T,S,0>(); }
    inline simd_ntuple<T,S,0> apply_inverse_mask(const smask_t<T,S> &t) const  { return simd_ntuple<T,S,0>(); }

    inline simd_ntuple<T,S,0> _rsub(const simd_t<T,S> &t) const { return simd_ntuple<T,S,0>(); }
    inline simd_ntuple<T,S,0> _rdiv(const simd_t<T,S> &t) const { return simd_ntuple<T,S,0>(); }

    inline simd_t<T,S> _vertical_sum(simd_t<T,S> u) const { return u; }
    inline simd_t<T,S> _vertical_max(simd_t<T,S> u) const { return u; }
    inline simd_t<T,S> _vertical_xor(simd_t<T,S> u) const { return u; }
    inline simd_t<T,S> _vertical_dot(const simd_ntuple<T,S,0> &t, simd_t<T,S> u) const { return u; }
    inline simd_t<T,S> _vertical_dotn(const simd_ntuple<T,S,0> &t, simd_t<T,S> u) const { return u; }

    inline void horizontal_sum_in_place() { }
    inline void max_in_place(const simd_ntuple<T,S,0> &t) { }
};


// -------------------------------------------------------------------------------------------------
//
// Arithmetic operators


template<typename T, int S, int N> 
inline simd_ntuple<T,S,N> operator+(const simd_t<T,S> &x, const simd_ntuple<T,S,N> &y) { return y+x; }

template<typename T, int S, int N> 
inline simd_ntuple<T,S,N> operator-(const simd_t<T,S> &x, const simd_ntuple<T,S,N> &y) { return y._rsub(x); }

template<typename T, int S, int N> 
inline simd_ntuple<T,S,N> operator*(const simd_t<T,S> &x, const simd_ntuple<T,S,N> &y) { return y*x; }

template<typename T, int S, int N> 
inline simd_ntuple<T,S,N> operator/(const simd_t<T,S> &x, const simd_ntuple<T,S,N> &y) { return y._rdiv(x); }


}  // namespace simd_helpers

#endif // _SIMD_HELPERS_SIMD_NTUPLE_HPP

