#include "simd_helpers/simd_debug.hpp"

using namespace std;
using namespace simd_helpers;


// Tests constructor (simd_t<T,S>, simd_t<T,S>) -> simd_t<T,2*S>
template<typename T, unsigned int S>
inline void test_merging_constructor(std::mt19937 &rng) 
{
    simd_t<T,S> a = uniform_random_simd_t<T,S> (rng, -1000, 1000);
    simd_t<T,S> b = uniform_random_simd_t<T,S> (rng, -1000, 1000);
    simd_t<T,2*S> c(a, b);

    vector<T> va = vectorize(a);
    vector<T> vb = vectorize(b);
    vector<T> vc = vectorize(c);

    for (int i = 0; i < S; i++) {
	assert(va[i] == vc[i]);
	assert(vb[i] == vc[i+4]);
    }
};

template<typename T, unsigned int S>
inline void test_abs(std::mt19937 &rng)
{
    simd_t<T,S> x = uniform_random_simd_t<T,S> (rng, -1000, 1000);
    simd_t<T,S> y = x.abs();

    vector<T> vx = vectorize(x);
    vector<T> vy = vectorize(y);
    
    for (unsigned int i = 0; i < S; i++)
	assert(vy[i] == max(vx[i],-vx[i]));
}


int main(int argc, char **argv)
{
    std::random_device rd;
    std::mt19937 rng(rd());

    for (int iter = 0; iter < 1000; iter++) {
	test_merging_constructor<int,4> (rng);
	test_merging_constructor<float,4> (rng);

	test_abs<int,4> (rng);
	test_abs<int,8> (rng);
	test_abs<float,4> (rng);
	test_abs<float,8> (rng);
    }

    cout << "test-basics: pass\n";
    return 0;
}
