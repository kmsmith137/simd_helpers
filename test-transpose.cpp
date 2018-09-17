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
	    cout << " " << arr[i*S+j];
	cout << "\n";
    }
}


// -------------------------------------------------------------------------------------------------


template<typename T>
static void reference_transpose(T *dst, const T *src, int S, int N)
{
    // Number of spectator indices
    int M = S/N;

    // Shape: (N,M,N)
    for (int i = 0; i < N; i++)
	for (int j = 0; j < M; j++)
	    for (int k = 0; k < N; k++)
		dst[i*M*N+j*N+k] = src[k*M*N+j*N+i];
}


template<typename T>
static void reference_btranspose(T *dst, const T *src, int S, int N)
{
    // Number of spectator indices
    int M = S/N;

    // Shape: (N,N,M)
    for (int i = 0; i < N; i++)
	for (int j = 0; j < N; j++)
	    for (int k = 0; k < M; k++)
		dst[i*M*N+j*M+k] = src[j*M*N+i*M+k];
}


template<typename T, int S, int N>
static void test_transpose(std::mt19937 &rng)
{
    vector<T> src = uniform_randvec<T> (rng, S*N, 0, 1000000);
    vector<T> dst1(S*N);
    vector<T> dst2(S*N);

    simd_ntuple<T,S,N> t;
    t.loadu(&src[0]);
    simd_transpose(t);
    t.storeu(&dst1[0]);

    reference_transpose(&dst2[0], &src[0], S, N);

    if (!strictly_equal(S*N, &dst1[0], &dst2[0])) {
	cerr << "test_transpose(T=" << type_name<T>() << ",S=" << S << ",N=" << N << ") failed\n";
	exit(1);
    }
}


template<typename T, int S, int N>
static void test_btranspose(std::mt19937 &rng)
{
    vector<T> src = uniform_randvec<T> (rng, S*N, 0, 1000000);
    vector<T> dst1(S*N);
    vector<T> dst2(S*N);

    // range_2d(N, S, &src[0]);
    // show_2d("input", N, S, &src[0]);

    simd_ntuple<T,S,N> t;
    t.loadu(&src[0]);
    simd_btranspose(t);
    t.storeu(&dst1[0]);

    reference_btranspose(&dst2[0], &src[0], S, N);
    // show_2d("fast kernel", N, S, &dst1[0]);
    // show_2d("reference kernel", N, S, &dst2[0]);

    if (!strictly_equal(S*N, &dst1[0], &dst2[0])) {
	cerr << "test_btranspose(T=" << type_name<T>() << ",S=" << S << ",N=" << N << ") failed\n";
	exit(1);
    }
}


int main(int argc, char **argv)
{
#ifdef __AVX2__
    std::random_device rd;
    std::mt19937 rng(rd());

    for (int iter = 0; iter < 1000; iter++) {
	test_transpose<float,8,2> (rng);
	test_transpose<float,8,4> (rng);
	test_transpose<float,8,8> (rng);
	test_btranspose<float,8,2> (rng);
	test_btranspose<float,8,4> (rng);
	test_btranspose<float,8,8> (rng);
    }
    cout << "test-transpose: all tests passed\n";
#else
    cout << "Transpose kernels are only implemented for AVX2, nothing to do!\n";
#endif

    return 0;
}
