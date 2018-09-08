#include "simd_helpers/simd_int32.hpp"
#include "simd_helpers/simd_int64.hpp"
#include "simd_helpers/simd_float32.hpp"
#include "simd_helpers/simd_float64.hpp"
#include "simd_helpers/simd_ntuple.hpp"
#include "simd_helpers/sort.hpp"

#include "simd_helpers/simd_debug.hpp"

using namespace std;
using namespace simd_helpers;


// ------------------------------------------------------------------------------------------------


template<typename T, int S, int N>
static void test_sort1(std::mt19937 &rng)
{
    vector<T> x(S*N, 0);
    vector<T> y(S*N, 0);
    vector<T> z(S*N, 0);
    vector<T> t(N, 0);
    
    for (int iter = 0; iter < 10000; iter++) {
	for (int i = 0; i < S*N; i++)
	    x[i] = uniform_rand<T>(rng, -10, 10);

	simd_ntuple<T,S,N> a;
	a.loadu(&x[0]);
	simd_sort(a);
	a.storeu(&y[0]);

	for (int i = 0; i < S; i++) {
	    for (int j = 0; j < N; j++)
		t[j] = x[j*S+i];

	    std::sort(t.begin(), t.end());

	    for (int j = 0; j < N; j++)
		z[j*S+i] = t[j];
	}

	for (int i = 0; i < S*N; i++) {
	    if (y[i] != z[i])
		throw runtime_error("sort failed");
	}
    }
}


template<typename T, int S>
static void test_sort(std::mt19937 &rng)
{
    test_sort1<T,S,2> (rng);
    test_sort1<T,S,3> (rng);
    test_sort1<T,S,4> (rng);
    test_sort1<T,S,5> (rng);
    test_sort1<T,S,6> (rng);
    test_sort1<T,S,7> (rng);
    test_sort1<T,S,8> (rng);
    test_sort1<T,S,9> (rng);
    test_sort1<T,S,10> (rng);
    test_sort1<T,S,11> (rng);
    test_sort1<T,S,12> (rng);
    test_sort1<T,S,13> (rng);
    test_sort1<T,S,14> (rng);
    test_sort1<T,S,15> (rng);
    test_sort1<T,S,16> (rng);

    cout << "test_sort<" << type_name<T>() << "," << S << ">: pass" << endl;
}


int main(int argc, char **argv)
{
    std::random_device rd;
    std::mt19937 rng(rd());

    test_sort<int,4> (rng);
    test_sort<int64_t,2> (rng);
    test_sort<float,4> (rng);
    test_sort<double,2> (rng);
#ifdef __AVX__
    test_sort<int,8> (rng);
    test_sort<int64_t,4> (rng);
    test_sort<float,8> (rng);
    test_sort<double,4> (rng);
#endif

    return 0;
}
