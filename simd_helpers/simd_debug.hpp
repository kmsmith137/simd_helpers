#ifndef _SIMD_HELPERS_SIMD_DEBUG_HPP
#define _SIMD_HELPERS_SIMD_DEBUG_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

#include <vector>
#include <random>
#include <cassert>
#include <iostream>
#include <type_traits>

#include "simd_t.hpp"
#include "simd_ntuple.hpp"
#include "simd_trimatrix.hpp"
#include "udsample.hpp"

namespace simd_helpers {
#if 0
}  // pacify emacs c-mode
#endif


// -------------------------------------------------------------------------------------------------
//
// vectorize(): unpacks a simd type to a std::vector
// pack_simd_xx(): converts a std::vector back to a simd type


template<typename T, unsigned int S>
inline std::vector<T> vectorize(simd_t<T,S> x)
{
    std::vector<T> ret(S);
    x.storeu(&ret[0]);
    return ret;
}


template<typename T, unsigned int S, unsigned int N>
inline std::vector<T> vectorize(const simd_ntuple<T,S,N> &v)
{
    std::vector<T> ret(N*S);
    v.storeu(&ret[0]);
    return ret;
}


template<typename T, unsigned int S, unsigned int N>
inline std::vector<T> vectorize(const simd_trimatrix<T,S,N> &m)
{
    std::vector<T> ret((N*(N+1)*S)/2);
    m.storeu(&ret[0]);
    return ret;
}


template<typename T, unsigned int S>
inline simd_t<T,S> pack_simd_t(const std::vector<T> &v)
{
    assert(v.size() == S);
    return simd_t<T,S>::loadu(&v[0]);
}


template<typename T, unsigned int S, unsigned int N>
inline simd_ntuple<T,S,N> pack_simd_ntuple(const std::vector<T> &v)
{
    assert(v.size() == N*S);
    simd_ntuple<T,S,N> ret;
    ret.loadu(&v[0]);
    return ret;
}


template<typename T, unsigned int S, unsigned int N>
inline simd_trimatrix<T,S,N> pack_simd_trimatrix(const std::vector<T> &v)
{
    assert(v.size() == (N*(N+1)*S)/2);
    simd_trimatrix<T,S,N> ret;
    ret.loadu(&v[0]);
    return ret;
}


// this is often useful in conjunction with vectorize()
template<typename T>
inline T compare(const std::vector<T> &v, const std::vector<T> &w)
{
    assert(v.size() == w.size());

    T num = 0;
    T den = 0;
    
    for (unsigned int i = 0; i < v.size(); i++) {
	T x = v[i];
	T y = w[i];
	num += (x-y)*(x-y);
	den += x*x + y*y;
    }

    return (den > 0) ? sqrt(num/den) : 0;
}


template<typename T>
inline bool is_equal(const std::vector<T> &v, const std::vector<T> &w)
{
    assert(v.size() == w.size());
    
    for (unsigned int i = 0; i < v.size(); i++) {
	if (v[i] != w[i])
	    return false;
    }

    return true;
}


// -------------------------------------------------------------------------------------------------
//
// Randomizers.
//
// Reminder: an RNG is initialized with
//
//   std::random_device rd;
//   std::mt19937 rng(rd());



// The helper function _uniform_randvec() has syntax
//    std::vector<T> _uniform_randvec(std::mt19937 &rng, unsigned int n, T lo, T hi);
// but is implemented differently for integral and floating-point types.

// Integral case: generate random number in range [lo,hi].  Note that endpoints are included in range.
template<typename T, typename std::enable_if<std::is_integral<T>::value,int>::type = 0>
inline std::vector<T> _uniform_randvec(std::mt19937 &rng, unsigned int n, T lo, T hi)
{
    std::vector<T> ret(n);
    for (unsigned int i = 0; i < n; i++)
	ret[i] = std::uniform_int_distribution<>(lo,hi)(rng);
    return ret;
}

// Floating-point case
template<typename T, typename std::enable_if<std::is_floating_point<T>::value,int>::type = 0>
inline std::vector<T> _uniform_randvec(std::mt19937 &rng, unsigned int n, T lo, T hi)
{
    std::vector<T> ret(n);
    for (unsigned int i = 0; i < n; i++)
	ret[i] = lo + (hi-lo) * std::uniform_real_distribution<>()(rng);
    return ret;
}

template<typename T, unsigned int S>
inline simd_t<T,S> uniform_random_simd_t(std::mt19937 &rng, T lo, T hi)
{
    return pack_simd_t<T,S> (_uniform_randvec<T> (rng, S, lo, hi));
}

template<typename T, unsigned int S, unsigned int N>
inline simd_ntuple<T,S,N> uniform_random_simd_ntuple(std::mt19937 &rng, T lo, T hi)
{
    return pack_simd_ntuple<T,S,N> (_uniform_randvec<T> (rng, N*S, lo, hi));
}


// _gaussian_randvec(): defined only in floating-point case
template<typename T, typename std::enable_if<std::is_floating_point<T>::value,int>::type = 0>
inline std::vector<T> _gaussian_randvec(std::mt19937 &rng, unsigned int n)
{
    std::normal_distribution<> dist;

    std::vector<T> ret(n);
    for (unsigned int i = 0; i < n; i++)
	ret[i] = dist(rng);

    return ret;
}

template<typename T, unsigned int S>
inline simd_t<T,S> gaussian_random_simd_t(std::mt19937 &rng)
{
    return pack_simd_t<T,S> (_gaussian_randvec<T> (rng, S));
}

template<typename T, unsigned int S, unsigned int N>
inline simd_ntuple<T,S,N> gaussian_random_simd_ntuple(std::mt19937 &rng)
{
    return pack_simd_ntuple<T,S,N> (_gaussian_randvec<T> (rng, N*S));
}


template<typename T, unsigned int S, unsigned int N>
inline simd_trimatrix<T,S,N> random_simd_trimatrix(std::mt19937 &rng)
{
    std::vector<T> buf((N*(N+1)*S)/2, 0);
    std::normal_distribution<> dist;

    for (int i = 0; i < N; i++) {
	int r = (i*(i+1)*S) / 2;
	for (int j = r; j < r+i*S; j++)
	    buf[j] = dist(rng);
	for (int j = r+i*S; j < r+(i+1)*S; j++)
	    buf[j] = 5.0 + 5.0 * std::uniform_real_distribution<>()(rng);
    }

    return pack_simd_trimatrix<T,S,N> (buf);
}


// -------------------------------------------------------------------------------------------------
//
// Print routines


template<class T, unsigned int S, unsigned int M>
struct _simd_writer 
{
    static inline void write(std::ostream &os, simd_t<T,S> x)
    {
	_simd_writer<T,S,M-1>::write(os, x);
	os << ", " << x.template extract<M-1>();
    }
};


template<class T, unsigned int S>
struct _simd_writer<T,S,1>
{
    static inline void write(std::ostream &os, simd_t<T,S> x) 
    { 
	os << x.template extract<0> ();
    }
};


template<typename T, unsigned int S>
std::ostream &operator<<(std::ostream &os, simd_t<T,S> x)
{
    os << "[";
    _simd_writer<T,S,S>::write(os, x);
    os << "]";
    return os;
}


// -------------------------------------------------------------------------------------------------
//
// Print routines, part 2


template<typename T, unsigned int S, unsigned int N>
struct _simd_nwriter
{
    static inline void write(std::ostream &os, const simd_ntuple<T,S,N> &v)
    {
	_simd_nwriter<T,S,N-1>::write(os, v.v);
	os << ", " << v.x;
    }

    static inline void write(std::ostream &os, const simd_trimatrix<T,S,N> &m)
    {
	_simd_nwriter<T,S,N-1>::write(os, m.m);
	os << ",\n " << m.v;
    }
};


template<typename T, unsigned int S>
struct _simd_nwriter<T,S,1>
{
    static inline void write(std::ostream &os, const simd_ntuple<T,S,1> &v)
    {
	os << v.x;
    }

    static inline void write(std::ostream &os, const simd_trimatrix<T,S,1> &m)
    {
	os << m.v;
    }
};


template<typename T, unsigned int S, unsigned int N>
std::ostream &operator<<(std::ostream &os, const simd_ntuple<T,S,N> &v)
{
    os << "{";
    _simd_nwriter<T,S,N>::write(os, v);
    os << "}";
    return os;
}


template<typename T, unsigned int S, unsigned int N>
std::ostream &operator<<(std::ostream &os, const simd_trimatrix<T,S,N> &m)
{
    os << "{";
    _simd_nwriter<T,S,N>::write(os, m);
    os << "}";
    return os;
}


}  // namespace simd_helpers

#endif // _SIMD_HELPERS_SIMD_DEBUG_HPP
