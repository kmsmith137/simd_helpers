#ifndef _SIMD_HELPERS_SIMD_NTUPLE_HPP
#define _SIMD_HELPERS_SIMD_NTUPLE_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

#include "simd_t.hpp"

namespace simd_helpers {
#if 0
}  // pacify emacs c-mode
#endif


template<typename T, unsigned int S, unsigned int N>
struct simd_ntuple
{
    simd_ntuple<T,S,N-1> v;
    simd_t<T,S> x;

    simd_ntuple() { }

    inline void setzero()
    {
	v.setzero();
	x = simd_t<T,S>::zero();
    }

    inline void loadu(const T *p)
    {
	v.loadu(p);
	x = simd_t<T,S>::loadu(p+(N-1)*S);
    }

    inline void storeu(T *p) const
    {
	v.storeu(p);
	x.storeu(p+(N-1)*S);
    }

    inline void set1_slow(const T *p)
    {
	v.set1_slow(p);
	x = p[N-1];
    }

    template<unsigned int M, typename std::enable_if<(M == N-1),int>::type = 0>
    inline simd_t<T,S> extract() const { return x; }

    template<unsigned int M, typename std::enable_if<(M < N-1),int>::type = 0>
    inline simd_t<T,S> extract() const { return v.template extract<M>(); }

    // Non-const version of extract<>()
    template<unsigned int M, typename std::enable_if<(M == N-1),int>::type = 0>
    inline simd_t<T,S> &extract() { return x; }

    // Non-const version of extract<>()
    template<unsigned int M, typename std::enable_if<(M < N-1),int>::type = 0>
    inline simd_t<T,S> &extract() { return v.template extract<M>(); }

    inline simd_ntuple<T,S,N> &operator+=(const simd_ntuple<T,S,N> &t)  { v += t.v; x += t.x; return *this; }
    inline simd_ntuple<T,S,N> &operator-=(const simd_ntuple<T,S,N> &t)  { v -= t.v; x -= t.x; return *this; }
    inline simd_ntuple<T,S,N> &operator*=(const simd_ntuple<T,S,N> &t)  { v *= t.v; x *= t.x; return *this; }
    inline simd_ntuple<T,S,N> &operator/=(const simd_ntuple<T,S,N> &t)  { v /= t.v; x /= t.x; return *this; }

    inline simd_ntuple<T,S,N> operator+(const simd_ntuple<T,S,N> &t) const  { simd_ntuple<T,S,N> ret; ret.v = v+t.v; ret.x = x+t.x; return ret; }
    inline simd_ntuple<T,S,N> operator-(const simd_ntuple<T,S,N> &t) const  { simd_ntuple<T,S,N> ret; ret.v = v-t.v; ret.x = x-t.x; return ret; }
    inline simd_ntuple<T,S,N> operator*(const simd_ntuple<T,S,N> &t) const  { simd_ntuple<T,S,N> ret; ret.v = v*t.v; ret.x = x*t.x; return ret; }
    inline simd_ntuple<T,S,N> operator/(const simd_ntuple<T,S,N> &t) const  { simd_ntuple<T,S,N> ret; ret.v = v/t.v; ret.x = x/t.x; return ret; }

    inline simd_ntuple<T,S,N> &operator+=(const simd_t<T,S> &t)  { v += t; x += t; return *this; }
    inline simd_ntuple<T,S,N> &operator-=(const simd_t<T,S> &t)  { v -= t; x -= t; return *this; }
    inline simd_ntuple<T,S,N> &operator*=(const simd_t<T,S> &t)  { v *= t; x *= t; return *this; }
    inline simd_ntuple<T,S,N> &operator/=(const simd_t<T,S> &t)  { v /= t; x /= t; return *this; }

    inline simd_ntuple<T,S,N> operator+(const simd_t<T,S> &t) const { simd_ntuple<T,S,N> ret; ret.v = v+t; ret.x = x+t; return ret; }
    inline simd_ntuple<T,S,N> operator-(const simd_t<T,S> &t) const { simd_ntuple<T,S,N> ret; ret.v = v-t; ret.x = x-t; return ret; }
    inline simd_ntuple<T,S,N> operator*(const simd_t<T,S> &t) const { simd_ntuple<T,S,N> ret; ret.v = v*t; ret.x = x*t; return ret; }
    inline simd_ntuple<T,S,N> operator/(const simd_t<T,S> &t) const { simd_ntuple<T,S,N> ret; ret.v = v/t; ret.x = x/t; return ret; }

    inline simd_ntuple<T,S,N> _rsub(const simd_t<T,S> &t)  { simd_ntuple<T,S,N> ret; ret.v = t-v; ret.x = t-x; return ret; }
    inline simd_ntuple<T,S,N> _rdiv(const simd_t<T,S> &t)  { simd_ntuple<T,S,N> ret; ret.v = t/v; ret.x = t/x; return ret; }

    // vertical_sum(): returns elementwise sum of all N simd_t's
    inline simd_t<T,S> _vertical_sum(simd_t<T,S> u) const  { return v._vertical_sum(u+x); }
    inline simd_t<T,S> vertical_sum() const  { return v._vertical_sum(x); }

    // vertical_dot(): returns elementwise length-N dot product of simd_t's
    inline simd_t<T,S> _vertical_dot(const simd_ntuple<T,S,N> &t, simd_t<T,S> u) const  { return v._vertical_dot(t.v, u + x*t.x); }
    inline simd_t<T,S> _vertical_dotn(const simd_ntuple<T,S,N> &t, simd_t<T,S> u) const { return v._vertical_dotn(t.v, u - x*t.x); }
    inline simd_t<T,S> vertical_dot(const simd_ntuple<T,S,N> &t) const  { return v._vertical_dot(t.v, x*t.x); }

    inline void horizontal_sum_in_place()
    {
	v.horizontal_sum_in_place();
	x = x.horizontal_sum();
    }

    // sum(): returns sum of all scalars in the simd_ntuple
    inline T sum() const { return this->vertical_sum().sum(); }
};


template<typename T, unsigned int S> 
struct simd_ntuple<T,S,0> 
{ 
    inline void setzero() { }
    inline void loadu(const T *p) { }
    inline void storeu(T *p) const { }
    inline void set1_slow(const T *p) { }

    inline simd_ntuple<T,S,0> &operator+=(const simd_ntuple<T,S,0> &t) { return *this; }
    inline simd_ntuple<T,S,0> &operator-=(const simd_ntuple<T,S,0> &t) { return *this; }
    inline simd_ntuple<T,S,0> &operator*=(const simd_ntuple<T,S,0> &t) { return *this; }
    inline simd_ntuple<T,S,0> &operator/=(const simd_ntuple<T,S,0> &t) { return *this; }

    inline simd_ntuple<T,S,0> operator+(const simd_ntuple<T,S,0> &t) const { return simd_ntuple<T,S,0>(); }
    inline simd_ntuple<T,S,0> operator-(const simd_ntuple<T,S,0> &t) const { return simd_ntuple<T,S,0>(); }
    inline simd_ntuple<T,S,0> operator*(const simd_ntuple<T,S,0> &t) const { return simd_ntuple<T,S,0>(); }
    inline simd_ntuple<T,S,0> operator/(const simd_ntuple<T,S,0> &t) const { return simd_ntuple<T,S,0>(); }

    inline simd_ntuple<T,S,0> &operator+=(const simd_t<T,S> &t) { return *this; }
    inline simd_ntuple<T,S,0> &operator-=(const simd_t<T,S> &t) { return *this; }
    inline simd_ntuple<T,S,0> &operator*=(const simd_t<T,S> &t) { return *this; }
    inline simd_ntuple<T,S,0> &operator/=(const simd_t<T,S> &t) { return *this; }

    inline simd_ntuple<T,S,0> operator+(const simd_t<T,S> &t) const { return simd_ntuple<T,S,0>(); }
    inline simd_ntuple<T,S,0> operator-(const simd_t<T,S> &t) const { return simd_ntuple<T,S,0>(); }
    inline simd_ntuple<T,S,0> operator*(const simd_t<T,S> &t) const { return simd_ntuple<T,S,0>(); }
    inline simd_ntuple<T,S,0> operator/(const simd_t<T,S> &t) const { return simd_ntuple<T,S,0>(); }

    inline simd_ntuple<T,S,0> _rsub(const simd_t<T,S> &t) const { return simd_ntuple<T,S,0>(); }
    inline simd_ntuple<T,S,0> _rdiv(const simd_t<T,S> &t) const { return simd_ntuple<T,S,0>(); }

    inline simd_t<T,S> _vertical_sum(simd_t<T,S> u) const { return u; }
    inline simd_t<T,S> _vertical_dot(const simd_ntuple<T,S,0> &t, simd_t<T,S> u) const { return u; }
    inline simd_t<T,S> _vertical_dotn(const simd_ntuple<T,S,0> &t, simd_t<T,S> u) const { return u; }

    inline void horizontal_sum_in_place() { }
};


// -------------------------------------------------------------------------------------------------
//
// Arithmetic operators


template<typename T, unsigned int S, unsigned int N> 
inline simd_ntuple<T,S,N> operator+(const simd_t<T,S> &x, const simd_ntuple<T,S,N> &y) { return y+x; }

template<typename T, unsigned int S, unsigned int N> 
inline simd_ntuple<T,S,N> operator-(const simd_t<T,S> &x, const simd_ntuple<T,S,N> &y) { return y._rsub(x); }

template<typename T, unsigned int S, unsigned int N> 
inline simd_ntuple<T,S,N> operator*(const simd_t<T,S> &x, const simd_ntuple<T,S,N> &y) { return y*x; }

template<typename T, unsigned int S, unsigned int N> 
inline simd_ntuple<T,S,N> operator/(const simd_t<T,S> &x, const simd_ntuple<T,S,N> &y) { return y._rdiv(x); }


}  // namespace simd_helpers

#endif // _SIMD_HELPERS_SIMD_NTUPLE_HPP

