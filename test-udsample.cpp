#include "simd_helpers/simd_debug.hpp"
#include "simd_helpers/udsample.hpp"

using namespace std;
using namespace simd_helpers;


template<typename T>
static vector<T> reference_downsample(const vector<T> &v, int N)
{
    assert(N > 0);
    assert(v.size() > 0);
    assert(v.size() % N == 0);

    int m = v.size() / N;
    vector<T> ret(m, 0);

    for (int i = 0; i < m; i++)
	for (int j = 0; j < N; j++)
	    ret[i] += v[i*N+j];

    return ret;
}

template<typename T>
static vector<T> reference_upsample(const vector<T> &v, int N)
{
    assert(N > 0);
    assert(v.size() > 0);

    vector<T> ret(v.size()*N, 0);

    for (unsigned int i = 0; i < v.size(); i++)
	for (int j = 0; j < N; j++)
	    ret[i*N+j] = v[i];
    
    return ret;
}


template<typename T, unsigned int S, unsigned int N>
static void test_downsample(std::mt19937 &rng)
{
    simd_ntuple<T,S,N> x = gaussian_random_simd_ntuple<T,S,N> (rng);
    simd_t<T,S> y = downsample(x);

    double epsilon = compare(vectorize(y), reference_downsample(vectorize(x),N));
    assert(epsilon < 1.0e-6);
}


template<typename T, unsigned int S, unsigned int N>
static void test_upsample(std::mt19937 &rng)
{
    simd_t<T,S> x = gaussian_random_simd_t<T,S> (rng);

    simd_ntuple<T,S,N> y;
    upsample(y, x);

    double epsilon = compare(vectorize(y), reference_upsample(vectorize(x),N));
    assert(epsilon == 0);
}


int main(int argc, char **argv)
{
    std::random_device rd;
    std::mt19937 rng(rd());

    for (int iter = 0; iter < 100; iter++) {
	test_downsample<float,4,2> (rng);
	test_downsample<float,4,4> (rng);
	test_downsample<float,8,2> (rng);
	test_downsample<float,8,4> (rng);
	test_downsample<float,8,8> (rng);
	
	test_upsample<float,4,2> (rng);
	test_upsample<float,4,4> (rng);
	test_upsample<float,8,2> (rng);
    }

    cout << "test-udsample: pass\n";
    return 0;
}
