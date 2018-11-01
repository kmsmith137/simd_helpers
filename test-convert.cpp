#include <type_traits>
#include "simd_helpers/simd_debug.hpp"

using namespace std;
using namespace simd_helpers;


// Convenient when defining templated unit tests which work for both integral and floating-point types
namespace simd_helpers {
    template<> inline constexpr int machine_epsilon()  { return 0; }
    template<> inline constexpr int64_t machine_epsilon()  { return 0; }
};


inline int64_t xrandint(std::mt19937 &rng, int64_t minval, int64_t maxval)
{
    if (uniform_rand(rng, 0.0, 1.0) < 0.1)
	return minval;
    if (uniform_rand(rng, 0.0, 1.0) < 0.1)
	return maxval;
    return std::uniform_int_distribution<int64_t>(minval,maxval)(rng);
}


// -------------------------------------------------------------------------------------------------


template<typename Tdst, typename Tsrc>
struct convert_tester
{
    static_assert(Tdst::total_size == Tsrc::total_size, "total_size mismatch");
    static constexpr int total_size = Tdst::total_size;

    using Sdst = typename Tdst::scalar_type;
    using Ssrc = typename Tsrc::scalar_type;

    Ssrc src_arr[total_size];
    Sdst dst_arr[total_size];


    void _convert()
    {
	Tsrc src;
	src.loadu(src_arr);

	Tdst dst;
	convert(dst, src);
	dst.storeu(dst_arr);
    }


    void _fail()
    {
	cout << "simd_convert() failed: " 
	     << type_name(Tsrc()) << " -> " 
	     << type_name(Tdst()) << "\n"
	     << "  input: [";
	
	for (int j = 0; j < total_size; j++)
	    cout << " " << src_arr[j];
	
	cout << " ]\n"
	     << "  output: [";
	
	for (int j = 0; j < total_size; j++)
	    cout << " " << dst_arr[j];
	
	cout << "]\n";
	exit(1);
    }


    // Used to test float <-> double conversions
    void _test_fp_fp(std::mt19937 &rng)
    {
	double eps1 = machine_epsilon<Ssrc> ();
	double eps2 = machine_epsilon<Sdst> ();
	double eps = max(eps1, eps2);

	for (int i = 0; i < total_size; i++) {
	    // Generates a random number in the range (-1.7e28, 1.7e28) with log-like spacing.
	    src_arr[i] = uniform_rand(rng, -1.0, 1.0) * exp(uniform_rand(rng, -65.0, 65.0));
	}
	    
	_convert();

	for (int i = 0; i < total_size; i++) {
	    double thresh = eps * (1.0 + abs(src_arr[i]));

	    if (std::abs(dst_arr[i] - src_arr[i]) > thresh)
		_fail();
	}
    }


    // Used to test (int <-> float) and (int64 <-> double) conversions
    void _test_ip_fp(std::mt19937 &rng)
    {
	double eps1 = machine_epsilon<Ssrc> ();
	double eps2 = machine_epsilon<Sdst> ();
	double eps = max(eps1, eps2);

	// Could be relaxed, but currently assumed.
	assert(sizeof(Ssrc) == sizeof(Sdst));
	assert(sizeof(Ssrc) == 4 || sizeof(Ssrc) == 8);

	// ----------  First pass  -----------

	// FIXME need to determine exact limits here.
	int64_t max_exact = (sizeof(Ssrc) == 8) ? (1L << 50) : (1L << 21);
	int64_t min_exact = -max_exact;

	for (int i = 0; i < total_size; i++)
	    src_arr[i] = xrandint(rng, min_exact, max_exact);

	_convert();

	for (int i = 0; i < total_size; i++)
	    if (dst_arr[i] != Sdst(src_arr[i]))
		_fail();

	// ----------  Second pass  -----------

	// FIXME need to determine exact limits here.
	int64_t max_valid = (sizeof(Ssrc) == 8) ? (1L << 50) : ((1L << 31) - 1);
	int64_t min_valid = (sizeof(Ssrc) == 8) ? (-(1L << 50)) : (-(1L << 31));

	for (int i = 0; i < total_size; i++)
	    src_arr[i] = xrandint(rng, min_valid, max_valid);

	_convert();

	for (int i = 0; i < total_size; i++) {
	    double thresh = eps * (1.0 + abs(double(src_arr[i])));
	    double diff = double(dst_arr[i]) - double(src_arr[i]);

	    if (diff > thresh)
		_fail();
	}
    }


    void test(std::mt19937 &rng)
    {
	bool Fdst = std::is_floating_point<Sdst>::value;
	bool Fsrc = std::is_floating_point<Ssrc>::value;
	bool Idst = std::is_integral<Sdst>::value;
	bool Isrc = std::is_integral<Ssrc>::value;

	if (Fdst && Fsrc)
	    _test_fp_fp(rng);
	else if ((Fdst && Isrc) || (Idst && Fsrc))
	    _test_ip_fp(rng);
	else
	    throw runtime_error("Bad types in test()?!");
    }
};


template<typename Tdst, typename Tsrc>
void test_convert(std::mt19937 &rng)
{
    convert_tester<Tdst, Tsrc> t;
    t.test(rng);
}


int main(int argc, char **argv)
{
    std::random_device rd;
    std::mt19937 rng(rd());

    cout << "test-convert: start\n";

    for (int iter = 0; iter < 100; iter++) {
	// The ordering below agrees with the declaration ordering in convert.hpp

#ifdef __AVX__
	// Float <-> double conversions
	test_convert<simd_t<double,4>, simd_t<float,4>> (rng);
	test_convert<simd_ntuple<double,4,2>, simd_t<float,8>> (rng);
	test_convert<simd_t<float,4>, simd_t<double,4>> (rng);
	test_convert<simd_t<float,8>, simd_ntuple<double,4,2>> (rng);
#endif

	// Float <-> int32 conversions
	test_convert<simd_t<int,4>, simd_t<float,4>> (rng);
	test_convert<simd_t<float,4>, simd_t<int,4>> (rng);
#ifdef __AVX__
	test_convert<simd_t<int,8>, simd_t<float,8>> (rng);
	test_convert<simd_t<float,8>, simd_t<int,8>> (rng);
#endif

	// Double <-> int64 conversions
	test_convert<simd_t<int64_t,2>, simd_t<double,2>> (rng);
	test_convert<simd_t<double,2>, simd_t<int64_t,2>> (rng);
#ifdef __AVX__
	test_convert<simd_t<int64_t,4>, simd_t<double,4>> (rng);
	test_convert<simd_t<double,4>, simd_t<int64_t,4>> (rng);
#endif
    }
    
    cout << "test-convert: all tests passed\n";
    return 0;
}
