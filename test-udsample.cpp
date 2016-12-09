#include "simd_helpers/simd_debug.hpp"

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


template<typename T, unsigned int S, unsigned int N>
static void test_downsample(std::mt19937 &rng)
{
    simd_ntuple<T,S,N> x = gaussian_random_simd_ntuple<T,S,N> (rng);
    simd_t<T,S> y = downsample(x);

    double epsilon = compare(vectorize(y), reference_downsample(vectorize(x),N));
    cout << epsilon << endl;
}


int main(int argc, char **argv)
{
    std::random_device rd;
    std::mt19937 rng(rd());

    for (int iter = 0; iter < 10; iter++) {
	test_downsample<float,4,2> (rng);
    }

    cout << "pass\n";
    return 0;
}
