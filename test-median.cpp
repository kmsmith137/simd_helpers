#include <algorithm>

#include "simd_helpers/simd_int32.hpp"
#include "simd_helpers/simd_int64.hpp"
#include "simd_helpers/simd_float32.hpp"
#include "simd_helpers/simd_float64.hpp"
#include "simd_helpers/simd_ntuple.hpp"
#include "simd_helpers/median.hpp"

#include "simd_helpers/simd_debug.hpp"

using namespace std;
using namespace simd_helpers;


// ------------------------------------------------------------------------------------------------


template<typename T, int S, int N>
static void test_median1(std::mt19937 &rng)
{
    double thresh = (N % 2) ? 0.0 : 1.0e-5;

    vector<T> x(S*N, 0);
    vector<T> y(S, 0);
    vector<T> z(S, 0);
    vector<T> t(N, 0);
    
    for (int iter = 0; iter < 10000; iter++) {
	for (int i = 0; i < S*N; i++)
	    x[i] = uniform_rand<T>(rng, -10, 10);

	simd_ntuple<T,S,N> a;
	a.loadu(&x[0]);

	simd_t<T,S> b = simd_median(a);
	b.storeu(&y[0]);

	for (int i = 0; i < S; i++) {
	    for (int j = 0; j < N; j++)
		t[j] = x[j*S+i];

	    std::sort(t.begin(), t.end());

	    if (N % 2 == 0)
		z[i] = (t[N/2-1] + t[N/2]) / T(2);
	    else
		z[i] = t[(N-1)/2];
	}

	for (int i = 0; i < S; i++) {
	    if (fabs(y[i] - z[i]) > thresh) {
		cout << "simd_median<" << type_name<T>() << "," << S << "," << N << "> failed\n";
		exit(1);
	    }
	}
    }
}


template<typename T, int S>
static void test_median_even(std::mt19937 &rng)
{
    test_median1<T,S,2> (rng);
    test_median1<T,S,4> (rng);
    test_median1<T,S,6> (rng);
    test_median1<T,S,8> (rng);
    test_median1<T,S,10> (rng);
    test_median1<T,S,12> (rng);
    test_median1<T,S,14> (rng);
    test_median1<T,S,16> (rng);
}

template<typename T, int S>
static void test_median_odd(std::mt19937 &rng)
{
    test_median1<T,S,3> (rng);
    test_median1<T,S,5> (rng);
    test_median1<T,S,7> (rng);
    test_median1<T,S,9> (rng);
    test_median1<T,S,11> (rng);
    test_median1<T,S,13> (rng);
    test_median1<T,S,15> (rng);
}


template<typename T, int S, typename std::enable_if<is_floating_point<T>::value,int>::type = 0>
static void test_median(std::mt19937 &rng)
{
    test_median_even<T,S>(rng);
    test_median_odd<T,S>(rng);
    cout << "test_median<" << type_name<T>() << "," << S << ">: pass" << endl;
}

template<typename T, int S, typename std::enable_if<is_integral<T>::value,int>::type = 0>
static void test_median(std::mt19937 &rng)
{
    test_median_odd<T,S>(rng);
    cout << "test_median<" << type_name<T>() << "," << S << ">: pass" << endl;
}


    

int main(int argc, char **argv)
{
    std::random_device rd;
    std::mt19937 rng(rd());

    test_median<int,4> (rng);
    test_median<int64_t,2> (rng);
    test_median<float,4> (rng);
    test_median<double,2> (rng);
#ifdef __AVX__
    test_median<int,8> (rng);
    test_median<int64_t,4> (rng);
    test_median<float,8> (rng);
    test_median<double,4> (rng);
#endif

    return 0;
}
