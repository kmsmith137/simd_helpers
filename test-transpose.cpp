#include <iostream>
#include <iomanip>
#include "simd_helpers/simd_debug.hpp"

using namespace std;
using namespace simd_helpers;


// -------------------------------------------------------------------------------------------------
//
// For debugging.
//
// FIXME: move somewhere more general.


template<typename T>
static void range_2d(int N, int S, T *arr)
{
    for (int i = 0; i < N; i++)
	for (int j = 0; j < S; j++)
	    arr[i*S+j] = 10*i + j;
}


template<typename T>
static void show_2d(const string &label, int N, int S, const T *arr)
{
    cout << label << "\n";

    for (int i = 0; i < N; i++) {
	for (int j = 0; j < S; j++)
	    cout << " " << std::setw(3) << arr[i*S+j];
	cout << "\n";
    }
}


// -------------------------------------------------------------------------------------------------


template<typename T>
static void reference_transpose(T *dst, const T *src, int S, int N, int M)
{
    int P = S / (M*N);
    assert(M*N*P == S);

    // Shape: (N,M,N,P)
    for (int i = 0; i < N; i++)
	for (int j = 0; j < M; j++)
	    for (int k = 0; k < N; k++)
		for (int l = 0; l < P; l++)
		    dst[i*S + j*N*P + k*P + l] = src[k*S + j*N*P + i*P + l];
}


template<typename T, int S, int N, int M>
static void test_transpose(std::mt19937 &rng)
{
    vector<T> src = uniform_randvec<T> (rng, N*S, 0, 1000000);
    vector<T> dst1(N*S);
    vector<T> dst2(N*S);

    simd_ntuple<T,S,N> t;
    t.loadu(&src[0]);
    simd_transpose<T,S,N,M> (t);
    t.storeu(&dst1[0]);

    reference_transpose(&dst2[0], &src[0], S, N, M);

    if (!strictly_equal(S*N, &dst1[0], &dst2[0])) {
	cerr << "test_transpose<" << type_name<T>() << "," << S << "," << N << "," << M << "> failed\n";
	exit(1);
    }
}

int main(int argc, char **argv)
{
#ifdef __AVX2__
    std::random_device rd;
    std::mt19937 rng(rd());

    for (int iter = 0; iter < 1000; iter++) {
	test_transpose<float,8,2,1> (rng);
	test_transpose<float,8,2,2> (rng);
	test_transpose<float,8,2,4> (rng);
	test_transpose<float,8,4,1> (rng);
	test_transpose<float,8,4,2> (rng);
	test_transpose<float,8,8,1> (rng);
    }
    cout << "test-transpose: all tests passed\n";

#else
    cout << "Transpose kernels are only implemented for AVX2, nothing to do!\n";
#endif

    // test vextract_all
    const int n=3;
    const int s=8;
    float arr[n*s];
    float outarr[n*s];
    range_2d(n, s, arr);
    //show_2d("arr", n, s, arr);
    simd_ntuple<float,s,n> t;
    t.loadu(arr);
    for (int i=0; i<n*s; i++)
        outarr[i] = -42.0;
    t.vextract_all(outarr);
    //show_2d("vextract_all", s, n, outarr);
    for (int i=0; i<n; i++)
        for (int j=0; j<s; j++)
            if (outarr[j*n + i] != arr[i*s + j])
                cerr << "test_transpose<" << "," << s << "," << n << "," << "> vextract_all failed\n";
    float vec[n];
    const int j = 2;
    for (int i=0; i<n; i++)
        vec[i] = -42.0;
    t.template vextract<j>(vec);
    for (int i=0; i<n; i++)
        if (vec[i] != arr[i*s + j])
            cerr << "test_transpose<" << "," << s << "," << n << "," << "> vextract failed\n";
    //show_2d("vec", 1, n, vec);

    return 0;
}
