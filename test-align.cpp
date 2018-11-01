#include "simd_helpers/align.hpp"
#include "simd_helpers/simd_debug.hpp"

using namespace std;
using namespace simd_helpers;


// -------------------------------------------------------------------------------------------------
//
// void test_align<T,S>(rng)
//    Tests all align() operations for a given (T,S)
//
// simd_t<T,S> simd_align_slow(int A, simd_t<T,S> x, simd_t<T,S> y);
//    Slow version of align() in which A is a function argument, not a template parameter.
//
// void simd_align_slow(int A, simd_ntuple<T,S,N> &dst, const simd_ntuple<T,S,N> &x, const simd_ntuple<T,S,N> &src)
//    Slow version of align() in which A is a function argument, not a template parameter.


template<typename T, int S, int A1=S+1, typename std::enable_if<(A1==0),int>::type = 0>
static simd_t<T,S> simd_align_slow(int A, simd_t<T,S> x, simd_t<T,S> y)
{
    throw runtime_error("simd_align_slow() internal error");
}

template<typename T, int S, int A1=S+1, typename std::enable_if<(A1>0),int>::type = 0>
static simd_t<T,S> simd_align_slow(int A, simd_t<T,S> x, simd_t<T,S> y)
{
    return (A == A1-1) ? (simd_align<A1-1>(x,y)) : simd_align_slow<T,S,A1-1>(A,x,y);
}


template<typename T, int S, int N, int A1=S+1, typename std::enable_if<(A1==0),int>::type = 0>
static void simd_align_slow(int A, simd_ntuple<T,S,N> &dst, const simd_ntuple<T,S,N> &x, const simd_ntuple<T,S,N> &y)
{
    throw runtime_error("simd_align_slow() internal error");
}

template<typename T, int S, int N, int A1=S+1, typename std::enable_if<(A1>0),int>::type = 0>
static void simd_align_slow(int A, simd_ntuple<T,S,N> &dst, const simd_ntuple<T,S,N> &x, const simd_ntuple<T,S,N> &y)
{
    if (A == A1-1)
	simd_align<A1-1> (dst, x, y);
    else
	simd_align_slow<T,S,N,A1-1> (A, dst, x, y);
}


template<typename T, int S, int N>
static void test_align_ntuple(std::mt19937 &rng)
{
    for (int A = 0; A <= S; A++) {
	simd_ntuple<T,S,N> x = uniform_random_simd_ntuple<T,S,N> (rng, 0, 100);
	simd_ntuple<T,S,N> y = uniform_random_simd_ntuple<T,S,N> (rng, 0, 100);

	simd_ntuple<T,S,N> t;
	simd_align_slow(A, t, x, y);

	vector<T> vx = vectorize(x);
	vector<T> vy = vectorize(y);
	vector<T> vt = vectorize(t);

	for (int i = 0; i < N; i++) {
	    for (int s = 0; s < S; s++) {
		T u = vt[i*S+s];
		T v = (s+A < S) ? vx[i*S+(s+A)] : vy[i*S+(s+A-S)];
		assert(u == v);
	    }
	}
    }
}


template<typename T, int S>
static void test_align(std::mt19937 &rng)
{
    for (int A = 0; A <= S; A++) {
	simd_t<T,S> x = uniform_random_simd_t<T,S> (rng, 0, 100);
	simd_t<T,S> y = uniform_random_simd_t<T,S> (rng, 0, 100);
	simd_t<T,S> t = simd_align_slow(A, x, y);

	vector<T> vx = vectorize(x);
	vector<T> vy = vectorize(y);
	vector<T> vt = vectorize(t);

	for (int s = 0; s < S; s++) {
	    T u = vt[s];
	    T v = (s+A < S) ? vx[s+A] : vy[s+A-S];
	    assert(u == v);
	}
    }

    test_align_ntuple<T,S,1> (rng);
    test_align_ntuple<T,S,2> (rng);
    test_align_ntuple<T,S,3> (rng);
    test_align_ntuple<T,S,4> (rng);
}


// -------------------------------------------------------------------------------------------------



int main(int argc, char **argv)
{
    std::random_device rd;
    std::mt19937 rng(rd());

    for (int iter = 0; iter < 10000; iter++) {
	test_align<int,4> (rng);
	test_align<float,4> (rng);
	test_align<double,2> (rng);
	test_align<int64_t,2> (rng);
#ifdef __AVX__
	test_align<int,8> (rng);
	test_align<float,8> (rng);	
	test_align<double,4> (rng);
	test_align<int64_t,4> (rng);
#endif
    }

    cout << "test-align: pass\n";
    return 0;
}
