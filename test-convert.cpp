#include "simd_helpers/simd_debug.hpp"

namespace simd_helpers {
    // Convenient when defining templated unit tests which work for both integral and floating-point types
    template<> inline constexpr int machine_epsilon()  { return 0; }
    template<> inline constexpr int64_t machine_epsilon()  { return 0; }
}

using namespace std;
using namespace simd_helpers;


// -------------------------------------------------------------------------------------------------


// convert(simd_t<T,S> &dst, simd_t<T2,S> src)
template<typename T, typename T2, int S>
static void test_convert(std::mt19937 &rng)
{
    const double epsilon = 2000. * max<double> (machine_epsilon<T>(), machine_epsilon<T2>());

    simd_t<T,S> dst;
    simd_t<T2,S> src = uniform_random_simd_t<T2,S> (rng, -1000, 1000);

    convert(dst, src);

    for (int s = 0; s < S; s++) {
	if (std::abs(extract_slow(dst,s) - extract_slow(src,s)) <= epsilon)
	    continue;

	cerr << "test_convert() failed: simd_t<" << type_name<T2>() << "," << S << ">"
	     << " -> simd_t<" << type_name<T>() << "," << S << ">\n"
	     << "   input: " << src << "\n"
	     << "   output: " << dst << "\n";

	exit(1);
    }
}


// convert(simd_ntuple<T,S,N> &dst, simd_t<T2,S*N> src)
template<typename T, typename T2, int S, int N>
inline void test_upconvert(std::mt19937 &rng)
{
    const double epsilon = 2000. * max<double> (machine_epsilon<T>(), machine_epsilon<T2>());

    simd_ntuple<T,S,N> dst;
    simd_t<T2,S*N> src = uniform_random_simd_t<T2,S*N> (rng, -1000, 1000);

    convert(dst, src);

    for (int n = 0; n < N; n++) {
	for (int s = 0; s < S; s++) {
	    if (std::abs(extract_slow(dst,n,s) - extract_slow(src,n*S+s)) <= epsilon)
		continue;

	    cerr << "test_upconvert() failed: simd_t<" << type_name<T2>() << "," << S << ">"
		 << " -> simd_ntuple<" << type_name<T>() << "," << S << "," << N << ">\n"
		 << "   input: " << src << "\n"
		 << "   output: " << dst << "\n";
	    
	    exit(1);
	}
    }
}


// convert(simd_t<T,S*N> &dst, simd_ntuple<T2,S,N> src)
template<typename T, typename T2, int S, int N>
inline void test_downconvert(std::mt19937 &rng)
{
    const double epsilon = 2000. * max<double> (machine_epsilon<T>(), machine_epsilon<T2>());

    simd_t<T,S*N> dst;
    simd_ntuple<T2,S,N> src = uniform_random_simd_ntuple<T2,S,N> (rng, -1000, 1000);

    convert(dst, src);

    for (int n = 0; n < N; n++) {
	for (int s = 0; s < S; s++) {
	    if (std::abs(extract_slow(dst,n*S+s) - extract_slow(src,n,s)) <= epsilon)
		continue;

	    cerr << "test_downconvert() failed: simd_ntuple<" << type_name<T2>() << "," << S << "," << N << ">"
		 << " -> simd_t<" << type_name<T>() << "," << (S*N) << ">\n"
		 << "   input: " << src << "\n"
		 << "   output: " << dst << "\n";
	    
	    exit(1);
	}
    }
}


// -------------------------------------------------------------------------------------------------


int main(int argc, char **argv)
{
    std::random_device rd;
    std::mt19937 rng(rd());

    cout << "test-convert: start\n";

#ifdef __AVX__
    for (int iter = 0; iter < 100; iter++) {
	test_convert<float,double,4> (rng);
	test_convert<double,float,4> (rng);
	test_upconvert<double,float,4,2> (rng);
	test_downconvert<float,double,4,2> (rng);
    }
#endif
    
    cout << "test-convert: all tests passed\n";
    return 0;
}
