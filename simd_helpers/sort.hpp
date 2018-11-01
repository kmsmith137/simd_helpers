#ifndef _SIMD_HELPERS_SORT_HPP
#define _SIMD_HELPERS_SORT_HPP

#if (__cplusplus < 201103) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
#error "This source file needs to be compiled with C++11 support (g++ -std=c++11)"
#endif

#include "simd_ntuple.hpp"

namespace simd_helpers {
#if 0
}  // pacify emacs c-mode
#endif

// The sorting API is really simple: we define an inline function
//
//  void simd_sort(simd_ntuple<T,S,N> &x);
//
// which performs an N-element sort, independently in each of the S simd "slots".
//
// Implemented using sorting reference.  Here is a very useful reference!
//   http://pages.ripco.net/~jgamble/nw.html


// -------------------------------------------------------------------------------------------------
//
// _simd_sort_pair<I,J> (simd_ntuple<T,S,N> &x)
//
// This inline helper function ensures that x[I] and x[J] are in sorted order, swapping them if necessary.
//
// I experimented with two implementations: _simd_sort_pair_minmax() which uses min/max intrinsics, and
// _simd_sort_pair_blend() which uses comparison/blend.
//
// Empirically, _simd_sort_pair_minmax() is always faster, except for the case T=int64_t when there is
// a tie.  (In the int64_t case, the "blend" version makes more sense, since the min/max operations are
// being emulated using comparison/blend, but the compiler is able to make the two implementations
// equivalent through optimization.)


template<int I, int J, typename T, int S, int N>
inline void _simd_sort_pair_minmax(simd_ntuple<T,S,N> &x)
{
    simd_t<T,S> a = x.template extract<I> ();
    simd_t<T,S> b = x.template extract<J> ();

    x.template extract<I> () = a.min(b);
    x.template extract<J> () = a.max(b);
}

template<int I, int J, typename T, int S, int N>
inline void _simd_sort_pair_blend(simd_ntuple<T,S,N> &x)
{
    simd_t<T,S> a = x.template extract<I> ();
    simd_t<T,S> b = x.template extract<J> ();
    
    // Note: slightly better to use (>) here, instead of a different comparison operator (>=, <, <=)
    // since this is a little more efficient in the cases T=int64_t, and (T,S)=(int,8).
    simd_t<T,S> c = (a > b);

    x.template extract<I> () = simd_if(c, b, a);
    x.template extract<J> () = simd_if(c, a, b);
}

// As mentioned above, _simd_sort_pair_minmax() always seems to be faster.
// (Therefore, _simd_sort_pair_blend() is "dead code", but I kept it anyway.)
template<int I, int J, typename T, int S, int N>
inline void _simd_sort_pair(simd_ntuple<T,S,N> &x)
{
    _simd_sort_pair_minmax<I,J> (x);
}


// -------------------------------------------------------------------------------------------------
//
// Sorting networks follow.


template<typename T, int S>
inline void simd_sort(simd_ntuple<T,S,2> &x)
{
    _simd_sort_pair<0,1> (x);
}


template<typename T, int S>
inline void simd_sort(simd_ntuple<T,S,3> &x)
{
    _simd_sort_pair<1,2> (x);
    _simd_sort_pair<0,2> (x);
    _simd_sort_pair<0,1> (x);
}


template<typename T, int S>
inline void simd_sort(simd_ntuple<T,S,4> &x)
{
    _simd_sort_pair<0,1> (x);
    _simd_sort_pair<2,3> (x);
    _simd_sort_pair<0,2> (x);
    _simd_sort_pair<1,3> (x);
    _simd_sort_pair<1,2> (x);
}


template<typename T, int S>
inline void simd_sort(simd_ntuple<T,S,5> &x)
{
    _simd_sort_pair<0,1> (x);
    _simd_sort_pair<3,4> (x);
    _simd_sort_pair<2,4> (x);
    _simd_sort_pair<2,3> (x);
    _simd_sort_pair<0,3> (x);
    _simd_sort_pair<0,2> (x);
    _simd_sort_pair<1,4> (x);
    _simd_sort_pair<1,3> (x);
    _simd_sort_pair<1,2> (x);
}


template<typename T, int S>
inline void simd_sort(simd_ntuple<T,S,6> &x)
{
    _simd_sort_pair<1,2> (x);
    _simd_sort_pair<0,2> (x);
    _simd_sort_pair<0,1> (x);
    _simd_sort_pair<4,5> (x);
    _simd_sort_pair<3,5> (x);
    _simd_sort_pair<3,4> (x);
    _simd_sort_pair<0,3> (x);
    _simd_sort_pair<1,4> (x);
    _simd_sort_pair<2,5> (x);
    _simd_sort_pair<2,4> (x);
    _simd_sort_pair<1,3> (x);
    _simd_sort_pair<2,3> (x);
}


template<typename T, int S>
inline void simd_sort(simd_ntuple<T,S,7> &x)
{
    _simd_sort_pair<1,2> (x);
    _simd_sort_pair<0,2> (x);
    _simd_sort_pair<0,1> (x);
    _simd_sort_pair<3,4> (x);
    _simd_sort_pair<5,6> (x);
    _simd_sort_pair<3,5> (x);
    _simd_sort_pair<4,6> (x);
    _simd_sort_pair<4,5> (x);
    _simd_sort_pair<0,4> (x);
    _simd_sort_pair<0,3> (x);
    _simd_sort_pair<1,5> (x);
    _simd_sort_pair<2,6> (x);
    _simd_sort_pair<2,5> (x);
    _simd_sort_pair<1,3> (x);
    _simd_sort_pair<2,4> (x);
    _simd_sort_pair<2,3> (x);
}


template<typename T, int S>
inline void simd_sort(simd_ntuple<T,S,8> &x)
{
    _simd_sort_pair<0,1> (x);
    _simd_sort_pair<2,3> (x);
    _simd_sort_pair<0,2> (x);
    _simd_sort_pair<1,3> (x);
    _simd_sort_pair<1,2> (x);
    _simd_sort_pair<4,5> (x);
    _simd_sort_pair<6,7> (x);
    _simd_sort_pair<4,6> (x);
    _simd_sort_pair<5,7> (x);
    _simd_sort_pair<5,6> (x);
    _simd_sort_pair<0,4> (x);
    _simd_sort_pair<1,5> (x);
    _simd_sort_pair<1,4> (x);
    _simd_sort_pair<2,6> (x);
    _simd_sort_pair<3,7> (x);
    _simd_sort_pair<3,6> (x);
    _simd_sort_pair<2,4> (x);
    _simd_sort_pair<3,5> (x);
    _simd_sort_pair<3,4> (x);
}


template<typename T, int S>
inline void simd_sort(simd_ntuple<T,S,9> &x)
{
    _simd_sort_pair<0,1> (x);
    _simd_sort_pair<3,4> (x);
    _simd_sort_pair<6,7> (x);
    _simd_sort_pair<1,2> (x);
    _simd_sort_pair<4,5> (x);
    _simd_sort_pair<7,8> (x);
    _simd_sort_pair<0,1> (x);
    _simd_sort_pair<3,4> (x);
    _simd_sort_pair<6,7> (x);
    _simd_sort_pair<0,3> (x);
    _simd_sort_pair<3,6> (x);
    _simd_sort_pair<0,3> (x);
    _simd_sort_pair<1,4> (x);
    _simd_sort_pair<4,7> (x);
    _simd_sort_pair<1,4> (x);
    _simd_sort_pair<2,5> (x);
    _simd_sort_pair<5,8> (x);
    _simd_sort_pair<2,5> (x);
    _simd_sort_pair<1,3> (x);
    _simd_sort_pair<5,7> (x);
    _simd_sort_pair<2,6> (x);
    _simd_sort_pair<4,6> (x);
    _simd_sort_pair<2,4> (x);
    _simd_sort_pair<2,3> (x);
    _simd_sort_pair<5,6> (x);
}


template<typename T, int S>
inline void simd_sort(simd_ntuple<T,S,10> &x)
{
    _simd_sort_pair<4,9> (x);
    _simd_sort_pair<3,8> (x);
    _simd_sort_pair<2,7> (x);
    _simd_sort_pair<1,6> (x);
    _simd_sort_pair<0,5> (x);
    _simd_sort_pair<1,4> (x);
    _simd_sort_pair<6,9> (x);
    _simd_sort_pair<0,3> (x);
    _simd_sort_pair<5,8> (x);
    _simd_sort_pair<0,2> (x);
    _simd_sort_pair<3,6> (x);
    _simd_sort_pair<7,9> (x);
    _simd_sort_pair<0,1> (x);
    _simd_sort_pair<2,4> (x);
    _simd_sort_pair<5,7> (x);
    _simd_sort_pair<8,9> (x);
    _simd_sort_pair<1,2> (x);
    _simd_sort_pair<4,6> (x);
    _simd_sort_pair<7,8> (x);
    _simd_sort_pair<3,5> (x);
    _simd_sort_pair<2,5> (x);
    _simd_sort_pair<6,8> (x);
    _simd_sort_pair<1,3> (x);
    _simd_sort_pair<4,7> (x);
    _simd_sort_pair<2,3> (x);
    _simd_sort_pair<6,7> (x);
    _simd_sort_pair<3,4> (x);
    _simd_sort_pair<5,6> (x);
    _simd_sort_pair<4,5> (x);
}


template<typename T, int S>
inline void simd_sort(simd_ntuple<T,S,11> &x)
{
    _simd_sort_pair<0,1> (x);
    _simd_sort_pair<2,3> (x);
    _simd_sort_pair<4,5> (x);
    _simd_sort_pair<6,7> (x);
    _simd_sort_pair<8,9> (x);
    _simd_sort_pair<1,3> (x);
    _simd_sort_pair<5,7> (x);
    _simd_sort_pair<0,2> (x);
    _simd_sort_pair<4,6> (x);
    _simd_sort_pair<8,10> (x);
    _simd_sort_pair<1,2> (x);
    _simd_sort_pair<5,6> (x);
    _simd_sort_pair<9,10> (x);
    _simd_sort_pair<1,5> (x);
    _simd_sort_pair<6,10> (x);
    _simd_sort_pair<5,9> (x);
    _simd_sort_pair<2,6> (x);
    _simd_sort_pair<1,5> (x);
    _simd_sort_pair<6,10> (x);
    _simd_sort_pair<0,4> (x);
    _simd_sort_pair<3,7> (x);
    _simd_sort_pair<4,8> (x);
    _simd_sort_pair<0,4> (x);
    _simd_sort_pair<1,4> (x);
    _simd_sort_pair<7,10> (x);
    _simd_sort_pair<3,8> (x);
    _simd_sort_pair<2,3> (x);
    _simd_sort_pair<8,9> (x);
    _simd_sort_pair<2,4> (x);
    _simd_sort_pair<7,9> (x);
    _simd_sort_pair<3,5> (x);
    _simd_sort_pair<6,8> (x);
    _simd_sort_pair<3,4> (x);
    _simd_sort_pair<5,6> (x);
    _simd_sort_pair<7,8> (x);
}


template<typename T, int S>
inline void simd_sort(simd_ntuple<T,S,12> &x)
{
    _simd_sort_pair<0,1> (x);
    _simd_sort_pair<2,3> (x);
    _simd_sort_pair<4,5> (x);
    _simd_sort_pair<6,7> (x);
    _simd_sort_pair<8,9> (x);
    _simd_sort_pair<10,11> (x);
    _simd_sort_pair<1,3> (x);
    _simd_sort_pair<5,7> (x);
    _simd_sort_pair<9,11> (x);
    _simd_sort_pair<0,2> (x);
    _simd_sort_pair<4,6> (x);
    _simd_sort_pair<8,10> (x);
    _simd_sort_pair<1,2> (x);
    _simd_sort_pair<5,6> (x);
    _simd_sort_pair<9,10> (x);
    _simd_sort_pair<1,5> (x);
    _simd_sort_pair<6,10> (x);
    _simd_sort_pair<5,9> (x);
    _simd_sort_pair<2,6> (x);
    _simd_sort_pair<1,5> (x);
    _simd_sort_pair<6,10> (x);
    _simd_sort_pair<0,4> (x);
    _simd_sort_pair<7,11> (x);
    _simd_sort_pair<3,7> (x);
    _simd_sort_pair<4,8> (x);
    _simd_sort_pair<0,4> (x);
    _simd_sort_pair<7,11> (x);
    _simd_sort_pair<1,4> (x);
    _simd_sort_pair<7,10> (x);
    _simd_sort_pair<3,8> (x);
    _simd_sort_pair<2,3> (x);
    _simd_sort_pair<8,9> (x);
    _simd_sort_pair<2,4> (x);
    _simd_sort_pair<7,9> (x);
    _simd_sort_pair<3,5> (x);
    _simd_sort_pair<6,8> (x);
    _simd_sort_pair<3,4> (x);
    _simd_sort_pair<5,6> (x);
    _simd_sort_pair<7,8> (x);
}


template<typename T, int S>
inline void simd_sort(simd_ntuple<T,S,13> &x)
{
    _simd_sort_pair<1,7> (x);
    _simd_sort_pair<9,11> (x);
    _simd_sort_pair<3,4> (x);
    _simd_sort_pair<5,8> (x);
    _simd_sort_pair<0,12> (x);
    _simd_sort_pair<2,6> (x);
    _simd_sort_pair<0,1> (x);
    _simd_sort_pair<2,3> (x);
    _simd_sort_pair<4,6> (x);
    _simd_sort_pair<8,11> (x);
    _simd_sort_pair<7,12> (x);
    _simd_sort_pair<5,9> (x);
    _simd_sort_pair<0,2> (x);
    _simd_sort_pair<3,7> (x);
    _simd_sort_pair<10,11> (x);
    _simd_sort_pair<1,4> (x);
    _simd_sort_pair<6,12> (x);
    _simd_sort_pair<7,8> (x);
    _simd_sort_pair<11,12> (x);
    _simd_sort_pair<4,9> (x);
    _simd_sort_pair<6,10> (x);
    _simd_sort_pair<3,4> (x);
    _simd_sort_pair<5,6> (x);
    _simd_sort_pair<8,9> (x);
    _simd_sort_pair<10,11> (x);
    _simd_sort_pair<1,7> (x);
    _simd_sort_pair<2,6> (x);
    _simd_sort_pair<9,11> (x);
    _simd_sort_pair<1,3> (x);
    _simd_sort_pair<4,7> (x);
    _simd_sort_pair<8,10> (x);
    _simd_sort_pair<0,5> (x);
    _simd_sort_pair<2,5> (x);
    _simd_sort_pair<6,8> (x);
    _simd_sort_pair<9,10> (x);
    _simd_sort_pair<1,2> (x);
    _simd_sort_pair<3,5> (x);
    _simd_sort_pair<7,8> (x);
    _simd_sort_pair<4,6> (x);
    _simd_sort_pair<2,3> (x);
    _simd_sort_pair<4,5> (x);
    _simd_sort_pair<6,7> (x);
    _simd_sort_pair<8,9> (x);
    _simd_sort_pair<3,4> (x);
    _simd_sort_pair<5,6> (x);
}


template<typename T, int S>
inline void simd_sort(simd_ntuple<T,S,14> &x)
{
    _simd_sort_pair<0,1> (x);
    _simd_sort_pair<2,3> (x);
    _simd_sort_pair<4,5> (x);
    _simd_sort_pair<6,7> (x);
    _simd_sort_pair<8,9> (x);
    _simd_sort_pair<10,11> (x);
    _simd_sort_pair<12,13> (x);
    _simd_sort_pair<0,2> (x);
    _simd_sort_pair<4,6> (x);
    _simd_sort_pair<8,10> (x);
    _simd_sort_pair<1,3> (x);
    _simd_sort_pair<5,7> (x);
    _simd_sort_pair<9,11> (x);
    _simd_sort_pair<0,4> (x);
    _simd_sort_pair<8,12> (x);
    _simd_sort_pair<1,5> (x);
    _simd_sort_pair<9,13> (x);
    _simd_sort_pair<2,6> (x);
    _simd_sort_pair<3,7> (x);
    _simd_sort_pair<0,8> (x);
    _simd_sort_pair<1,9> (x);
    _simd_sort_pair<2,10> (x);
    _simd_sort_pair<3,11> (x);
    _simd_sort_pair<4,12> (x);
    _simd_sort_pair<5,13> (x);
    _simd_sort_pair<5,10> (x);
    _simd_sort_pair<6,9> (x);
    _simd_sort_pair<3,12> (x);
    _simd_sort_pair<7,11> (x);
    _simd_sort_pair<1,2> (x);
    _simd_sort_pair<4,8> (x);
    _simd_sort_pair<1,4> (x);
    _simd_sort_pair<7,13> (x);
    _simd_sort_pair<2,8> (x);
    _simd_sort_pair<2,4> (x);
    _simd_sort_pair<5,6> (x);
    _simd_sort_pair<9,10> (x);
    _simd_sort_pair<11,13> (x);
    _simd_sort_pair<3,8> (x);
    _simd_sort_pair<7,12> (x);
    _simd_sort_pair<6,8> (x);
    _simd_sort_pair<10,12> (x);
    _simd_sort_pair<3,5> (x);
    _simd_sort_pair<7,9> (x);
    _simd_sort_pair<3,4> (x);
    _simd_sort_pair<5,6> (x);
    _simd_sort_pair<7,8> (x);
    _simd_sort_pair<9,10> (x);
    _simd_sort_pair<11,12> (x);
    _simd_sort_pair<6,7> (x);
    _simd_sort_pair<8,9> (x);
}


template<typename T, int S>
inline void simd_sort(simd_ntuple<T,S,15> &x)
{
    _simd_sort_pair<0,1> (x);
    _simd_sort_pair<2,3> (x);
    _simd_sort_pair<4,5> (x);
    _simd_sort_pair<6,7> (x);
    _simd_sort_pair<8,9> (x);
    _simd_sort_pair<10,11> (x);
    _simd_sort_pair<12,13> (x);
    _simd_sort_pair<0,2> (x);
    _simd_sort_pair<4,6> (x);
    _simd_sort_pair<8,10> (x);
    _simd_sort_pair<12,14> (x);
    _simd_sort_pair<1,3> (x);
    _simd_sort_pair<5,7> (x);
    _simd_sort_pair<9,11> (x);
    _simd_sort_pair<0,4> (x);
    _simd_sort_pair<8,12> (x);
    _simd_sort_pair<1,5> (x);
    _simd_sort_pair<9,13> (x);
    _simd_sort_pair<2,6> (x);
    _simd_sort_pair<10,14> (x);
    _simd_sort_pair<3,7> (x);
    _simd_sort_pair<0,8> (x);
    _simd_sort_pair<1,9> (x);
    _simd_sort_pair<2,10> (x);
    _simd_sort_pair<3,11> (x);
    _simd_sort_pair<4,12> (x);
    _simd_sort_pair<5,13> (x);
    _simd_sort_pair<6,14> (x);
    _simd_sort_pair<5,10> (x);
    _simd_sort_pair<6,9> (x);
    _simd_sort_pair<3,12> (x);
    _simd_sort_pair<13,14> (x);
    _simd_sort_pair<7,11> (x);
    _simd_sort_pair<1,2> (x);
    _simd_sort_pair<4,8> (x);
    _simd_sort_pair<1,4> (x);
    _simd_sort_pair<7,13> (x);
    _simd_sort_pair<2,8> (x);
    _simd_sort_pair<11,14> (x);
    _simd_sort_pair<2,4> (x);
    _simd_sort_pair<5,6> (x);
    _simd_sort_pair<9,10> (x);
    _simd_sort_pair<11,13> (x);
    _simd_sort_pair<3,8> (x);
    _simd_sort_pair<7,12> (x);
    _simd_sort_pair<6,8> (x);
    _simd_sort_pair<10,12> (x);
    _simd_sort_pair<3,5> (x);
    _simd_sort_pair<7,9> (x);
    _simd_sort_pair<3,4> (x);
    _simd_sort_pair<5,6> (x);
    _simd_sort_pair<7,8> (x);
    _simd_sort_pair<9,10> (x);
    _simd_sort_pair<11,12> (x);
    _simd_sort_pair<6,7> (x);
    _simd_sort_pair<8,9> (x);
}


template<typename T, int S>
inline void simd_sort(simd_ntuple<T,S,16> &x)
{
    _simd_sort_pair<0,1> (x);
    _simd_sort_pair<2,3> (x);
    _simd_sort_pair<4,5> (x);
    _simd_sort_pair<6,7> (x);
    _simd_sort_pair<8,9> (x);
    _simd_sort_pair<10,11> (x);
    _simd_sort_pair<12,13> (x);
    _simd_sort_pair<14,15> (x);
    _simd_sort_pair<0,2> (x);
    _simd_sort_pair<4,6> (x);
    _simd_sort_pair<8,10> (x);
    _simd_sort_pair<12,14> (x);
    _simd_sort_pair<1,3> (x);
    _simd_sort_pair<5,7> (x);
    _simd_sort_pair<9,11> (x);
    _simd_sort_pair<13,15> (x);
    _simd_sort_pair<0,4> (x);
    _simd_sort_pair<8,12> (x);
    _simd_sort_pair<1,5> (x);
    _simd_sort_pair<9,13> (x);
    _simd_sort_pair<2,6> (x);
    _simd_sort_pair<10,14> (x);
    _simd_sort_pair<3,7> (x);
    _simd_sort_pair<11,15> (x);
    _simd_sort_pair<0,8> (x);
    _simd_sort_pair<1,9> (x);
    _simd_sort_pair<2,10> (x);
    _simd_sort_pair<3,11> (x);
    _simd_sort_pair<4,12> (x);
    _simd_sort_pair<5,13> (x);
    _simd_sort_pair<6,14> (x);
    _simd_sort_pair<7,15> (x);
    _simd_sort_pair<5,10> (x);
    _simd_sort_pair<6,9> (x);
    _simd_sort_pair<3,12> (x);
    _simd_sort_pair<13,14> (x);
    _simd_sort_pair<7,11> (x);
    _simd_sort_pair<1,2> (x);
    _simd_sort_pair<4,8> (x);
    _simd_sort_pair<1,4> (x);
    _simd_sort_pair<7,13> (x);
    _simd_sort_pair<2,8> (x);
    _simd_sort_pair<11,14> (x);
    _simd_sort_pair<2,4> (x);
    _simd_sort_pair<5,6> (x);
    _simd_sort_pair<9,10> (x);
    _simd_sort_pair<11,13> (x);
    _simd_sort_pair<3,8> (x);
    _simd_sort_pair<7,12> (x);
    _simd_sort_pair<6,8> (x);
    _simd_sort_pair<10,12> (x);
    _simd_sort_pair<3,5> (x);
    _simd_sort_pair<7,9> (x);
    _simd_sort_pair<3,4> (x);
    _simd_sort_pair<5,6> (x);
    _simd_sort_pair<7,8> (x);
    _simd_sort_pair<9,10> (x);
    _simd_sort_pair<11,12> (x);
    _simd_sort_pair<6,7> (x);
    _simd_sort_pair<8,9> (x);
}


}  // namespace simd_helpers

#endif  // _SIMD_HELPERS_SORT_HPP
