#include <functional>
#include "simd_helpers/simd_debug.hpp"


namespace simd_helpers {
    // Convenient when defining templated unit tests which work for both integral and floating-point types
    template<> inline constexpr int machine_epsilon()  { return 0; }
    template<> inline constexpr int64_t machine_epsilon()  { return 0; }
}


using namespace std;
using namespace simd_helpers;


template<typename T>
static vector<T> reference_downsample(const vector<T> &v, int N, std::function<T(T,T)> op)
{
    assert(N > 0);
    assert(v.size() > 0);
    assert(v.size() % N == 0);

    int m = v.size() / N;
    vector<T> ret(m, 0);

    for (int i = 0; i < m; i++) {
	ret[i] = v[i*N];
	for (int j = 1; j < N; j++)
	    ret[i] = op(ret[i], v[i*N+j]);
    }

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


// Test upsample for a given (T,S,N) triple.
template<typename T, int S, int N>
static void test_upsample1(std::mt19937 &rng)
{
    simd_t<T,S> x = uniform_random_simd_t<T,S> (rng, 0, 100);

    simd_ntuple<T,S,N> y;
    simd_upsample(y, x);

    assert(strictly_equal(vectorize(y), reference_upsample(vectorize(x),N)));
}


// Test downsample for a given (T,S,N,Op) quadruple.
template<typename T, int S, int N, typename Op = simd_add<T,S> >
static void test_downsample1(std::mt19937 &rng, std::function<T(T,T)> op, const char *op_name)
{
    // Factor 300 is because the inputs go up to 100.
    const double epsilon0 = 300. * N * machine_epsilon<T>();

    simd_ntuple<T,S,N> x = uniform_random_simd_ntuple<T,S,N> (rng, 0, 100);
    simd_t<T,S> y = simd_downsample<T,S,Op> (x);
    vector<T> d = vectorize(y);

    vector<T> refd = reference_downsample(vectorize(x), N, op);
    double epsilon = maxdiff(d, refd);
    
    if (epsilon > epsilon0) {
	cerr << "test_downsample(" << type_name<T>() << "," << S << "," << N << "," << op_name << ") failed: "
	     << " epsilon=" << epsilon << ", epsilon0=" << epsilon0 << endl;
	exit(1);
    }
}


// Test up/downsample for a given (S,N) pair.
template<int S, int N>
static void test_udsample(std::mt19937 &rng)
{
    test_upsample1<int,S,N> (rng);
    test_upsample1<float,S,N> (rng);

    test_downsample1<int,S,N> (rng, [](int x, int y) { return x+y; }, "+");
    test_downsample1<float,S,N> (rng, [](float x, float y) { return x+y; }, "+");

    test_downsample1<int,S,N,simd_max<int,S>> (rng, [](int x, int y) { return max(x,y); }, "max");
    test_downsample1<float,S,N,simd_max<float,S>> (rng, [](float x, float y) { return max(x,y); }, "max");
    
    test_downsample1<int,S,N,simd_bitwise_or<int,S>> (rng, [](int x, int y) { return x|y; }, "|");
}


int main(int argc, char **argv)
{
    std::random_device rd;
    std::mt19937 rng(rd());

    for (int iter = 0; iter < 1000; iter++) {
	test_udsample<4,1> (rng);
	test_udsample<4,2> (rng);
	test_udsample<4,4> (rng);
	test_udsample<4,8> (rng);
	test_udsample<4,12> (rng);
	test_udsample<4,16> (rng);
	
#ifdef __AVX__
	// Upsampling is implemented for (D <= S).
	test_udsample<8,1> (rng);
	test_udsample<8,2> (rng);
	test_udsample<8,4> (rng);
	test_udsample<8,8> (rng);
	test_udsample<8,16> (rng);
	test_udsample<8,24> (rng);
	test_udsample<8,32> (rng);
#endif
    }
    
    cout << "test-udsample: pass\n";
    return 0;
}
