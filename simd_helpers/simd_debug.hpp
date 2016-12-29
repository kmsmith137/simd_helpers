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


// type_name(): useful for debugging.  (Note that typename is a reserved word!)
template<typename T> inline std::string type_name();
template<> inline std::string type_name<int> () { return "int"; }
template<> inline std::string type_name<float> () { return "float"; }
template<> inline std::string type_name<double> () { return "double"; }

template<typename T, unsigned int S>
inline std::string type_name(const simd_t<T,S> &x) { std::stringstream ss; ss << "simd_t<" << type_name<T>() << "," << S << ">"; return ss.str(); }

template<typename T, unsigned int S, unsigned int N>
inline std::string type_name(const simd_ntuple<T,S,N> &x) { std::stringstream ss; ss << "simd_ntuple<" << type_name<T>() << "," << S << "," << N << ">"; return ss.str(); }


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


// -------------------------------------------------------------------------------------------------
//
// Helper routines for inspecting vectors (often used in unit tests)
//
//   compare(v,w)          returns comparison statistic |v-w| / (|v|^2 + |w|^2)^(1/2)
//   maxdiff(v,w)          returns max_i |v_i-w_i|
//   maxabs(v)             returns max_i |v_i|
//   strictly_equal(v,w)   returns all(v_i==w-i)


template<typename T, typename S>
inline T compare(S n, const T *v, const T *w)
{
    T num = 0;
    T den = 0;
    
    for (S i = 0; i < n; i++) {
	T x = v[i];
	T y = w[i];
	num += (x-y)*(x-y);
	den += x*x + y*y;
    }

    return (den > 0) ? sqrt(num/den) : 0;
}

template<typename T>
inline T compare(const std::vector<T> &v, const std::vector<T> &w)
{
    assert(v.size() == w.size());
    return compare(v.size(), &v[0], &w[0]);
}


template<typename T, typename S>
inline T maxdiff(S n, const T *v1, const T *v2)
{
    assert(n > 0);

    T ret = std::fabs(v1[0]-v2[0]);
    for (S i = 1; i < n; i++)
	ret = std::max(ret, std::fabs(v1[i]-v2[i]));

    return ret;
}

template<typename T>
inline T maxdiff(const std::vector<T> &v1, const std::vector<T> &v2)
{
    assert(v1.size() == v2.size());
    assert(v1.size() > 0);

    return maxdiff(v1.size(), &v1[0], &v2[0]);
}


template<typename T, typename S>
inline T maxabs(S n, const T *v)
{
    assert(n > 0);

    T ret = std::fabs(v[0]);
    for (unsigned int i = 1; i < n; i++)
	ret = std::max(ret, std::fabs(v[i]));

    return ret;
}

template<typename T>
inline T maxabs(const std::vector<T> &v)
{
    assert(v.size() > 0);
    return maxabs(v.size(), &v[0]);
}


template<typename T, typename S>
inline bool strictly_equal(S n, const T *v, const T *w)
{
    for (S i = 0; i < n; i++) {
	if (v[i] != w[i])
	    return false;
    }

    return true;
}

template<typename T>
inline bool strictly_equal(const std::vector<T> &v, const std::vector<T> &w)
{
    assert(v.size() == w.size());
    return strictly_equal(v.size(), &v[0], &w[0]);
}


// -------------------------------------------------------------------------------------------------
//
// Randomizers.
//
// Reminder: an RNG is initialized with
//
//   std::random_device rd;
//   std::mt19937 rng(rd());



// The helper function uniform_randvec() has syntax
//    std::vector<T> uniform_randvec(std::mt19937 &rng, unsigned int n, T lo, T hi);
// but is implemented differently for integral and floating-point types.

// Integral case: generate random number in range [lo,hi].  Note that endpoints are included in range.
template<typename T, typename std::enable_if<std::is_integral<T>::value,int>::type = 0>
inline std::vector<T> uniform_randvec(std::mt19937 &rng, unsigned int n, T lo, T hi)
{
    std::vector<T> ret(n);
    for (unsigned int i = 0; i < n; i++)
	ret[i] = std::uniform_int_distribution<>(lo,hi)(rng);
    return ret;
}

// Floating-point case
template<typename T, typename std::enable_if<std::is_floating_point<T>::value,int>::type = 0>
inline std::vector<T> uniform_randvec(std::mt19937 &rng, unsigned int n, T lo, T hi)
{
    std::vector<T> ret(n);
    for (unsigned int i = 0; i < n; i++)
	ret[i] = lo + (hi-lo) * std::uniform_real_distribution<>()(rng);
    return ret;
}

template<typename T, unsigned int S>
inline simd_t<T,S> uniform_random_simd_t(std::mt19937 &rng, T lo, T hi)
{
    return pack_simd_t<T,S> (uniform_randvec<T> (rng, S, lo, hi));
}

template<typename T, unsigned int S, unsigned int N>
inline simd_ntuple<T,S,N> uniform_random_simd_ntuple(std::mt19937 &rng, T lo, T hi)
{
    return pack_simd_ntuple<T,S,N> (uniform_randvec<T> (rng, N*S, lo, hi));
}


// gaussian_randvec(): defined only in floating-point case
template<typename T, typename std::enable_if<std::is_floating_point<T>::value,int>::type = 0>
inline std::vector<T> gaussian_randvec(std::mt19937 &rng, unsigned int n)
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
    return pack_simd_t<T,S> (gaussian_randvec<T> (rng, S));
}

template<typename T, unsigned int S, unsigned int N>
inline simd_ntuple<T,S,N> gaussian_random_simd_ntuple(std::mt19937 &rng)
{
    return pack_simd_ntuple<T,S,N> (gaussian_randvec<T> (rng, N*S));
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
