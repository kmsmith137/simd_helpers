#include "simd_helpers/simd_debug.hpp"

using namespace std;
using namespace simd_helpers;


template<typename T, unsigned int S>
static void test_abs(std::mt19937 &rng)
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
	test_abs<int,4> (rng);
	test_abs<int,8> (rng);
	test_abs<float,4> (rng);
	test_abs<float,8> (rng);
    }

    cout << "test-basics: pass\n";
    return 0;
}
