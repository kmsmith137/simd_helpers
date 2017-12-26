#include <memory>
#include "simd_helpers/simd_debug.hpp"

using namespace std;
using namespace simd_helpers;


void slow_quantize(uint8_t *dst, const float *src, int nsrc, int B)
{
    assert(B == 1);   // only 1-bit kernel implemented for now!
    assert(nsrc % 8 == 0);

    for (int m = 0; m < nsrc/8; m++) {
	uint8_t iout = 0;
	for (int n = 0; n < 8; n++)
	    if (src[8*m+n] > 0.0f)
		iout |= (1 << n);
	dst[m] = iout;
    }
}


template<int S, int B>
void fast_quantize(uint8_t *dst, const float *src, int nsrc)
{
    // Length of quantization kernel, in units sizeof(simd_t<T,S>).
    constexpr int K = (sizeof(*src) * 8) / B;
    assert(nsrc % (K*S) == 0);

    simd_quantizer<float,S,B> q;
    int *dst32 = reinterpret_cast<int *> (dst);

    for (int m = 0; m < nsrc/(K*S); m++)
	simd_store(dst32 + m*S, q.quantize(src + m*K*S));
}


template<typename T, int S, int B>
void test_quantize(std::mt19937 &rng)
{
    int n = simd_randint(rng, 1, 10);
    int ndst = n * (sizeof(T) * S);    // number of uint8_t's in output array.
    int nsrc = (ndst * 8) / B;         // number of T's in input array.

    unique_ptr<uint8_t[]> dst1(new uint8_t[ndst]);
    unique_ptr<uint8_t[]> dst2(new uint8_t[ndst]);
    unique_ptr<float[]> src(new float[nsrc]);

    for (int i = 0; i < nsrc; i++)
	src[i] = simd_randint(rng,0,4) ? uniform_rand(rng,-1,1) : 0.0;

    for (int i = 0; i < n; i++) {
	slow_quantize(dst1.get(), src.get(), nsrc, B);
	fast_quantize<S,B> (dst2.get(), src.get(), nsrc);
    }

    for (int i = 0; i < ndst; i++) {
	if (dst1[i] != dst2[i]) {
	    cout << "test_quantize<" << type_name<T>() << "," << S << "," << B << "> failed\n";
	    exit(1);
	}
    }
}


int main(int argc, char **argv)
{
    std::random_device rd;
    std::mt19937 rng(rd());

    for (int iter = 0; iter < 1000; iter++) {
	test_quantize<float,4,1> (rng);
#ifdef __AVX__
	test_quantize<float,8,1> (rng);
#endif
    }

    cout << "test-quantize: pass\n";
    return 0;
}
