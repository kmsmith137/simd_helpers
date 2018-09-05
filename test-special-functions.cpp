#include "simd_helpers/simd_debug.hpp"

using namespace std;
using namespace simd_helpers;


template<typename T, int S>
void test_exp2(std::mt19937 &rng)
{
    const T thresh = (sizeof(T)==8) ? 1.0e-13 : 3.0e-6;
    simd_t<T,S> x, y;

    T x_arr[S];
    T y_arr[S];

    for (int iter = 0; iter < 1000; iter++) {
	// FIXME should test underflow/overflow
	for (int i = 0; i < S; i++)
	    x_arr[i] = uniform_rand(rng, -40.0, 40.0);

	x.loadu(x_arr);
	y = simd_exp2_unsafe(x);
	y.storeu(y_arr);

	for (int i = 0; i < S; i++) {
	    T dlog = log2(y_arr[i]) - x_arr[i];

	    if (abs(dlog) > thresh) {
		cout << "test_exp2<" << type_name<T>() << "," << S << "> failed:"
		     << " input=" << x_arr[i] << ", output=" << y_arr[i] << ", expected=" << exp2(x_arr[i])
		     << ", delta_log=" << dlog << ", thresh=" << thresh << endl;
		exit(1);
	    }
	}
    }
}


template<typename T, int S>
void test_log2p(std::mt19937 &rng)
{
    const T thresh = (sizeof(T)==8) ? 3.0e-13 : 3.0e-6;
    simd_t<T,S> x, y;

    T x_arr[S];
    T y_arr[S];

    for (int iter = 0; iter < 1000; iter++) {
	for (int i = 0; i < S; i++)
	    x_arr[i] = uniform_rand(rng, 0.0, 1.0);

	x.loadu(x_arr);
	y = _simd_log2p_restricted(x);
	y.storeu(y_arr);

	for (int i = 0; i < S; i++) {
	    T z = log1p(x_arr[i]) / log(2.0);
	    T dlog = log(y_arr[i]) - log(z);

	    if (abs(dlog) > thresh) {
		cout << "test_log2p<" << type_name<T>() << "," << S << "> failed:"
		     << " input=" << x_arr[i] << ", output=" << y_arr[i] << ", expected=" << z
		     << ", delta_log=" << dlog << ", thresh=" << thresh << endl;
		exit(1);
	    }
	}
    }
}


template<typename T, int S>
void test_log_add(std::mt19937 &rng)
{
    const T xmax = (sizeof(T)==8) ? 1000.0 : 100.0;
    const T thresh = (sizeof(T)==8) ? 3.0e-13 : 3.0e-5;

    simd_t<T,S> x, y, z;
    T x_arr[S];
    T y_arr[S];
    T z_arr[S];

    // Use huge number of iterations to ensure that overflow/underflow is exhaustively tested.
    for (int iter = 0; iter < 100000; iter++) {
	for (int i = 0; i < S; i++) {
	    x_arr[i] = uniform_rand(rng, -xmax, xmax);
	    y_arr[i] = uniform_rand(rng, -xmax, xmax);
	}

	x.loadu(x_arr);
	y.loadu(y_arr);

	z = simd_ln_add(x, y);
	z.storeu(z_arr);

	for (int i = 0; i < S; i++) {
	    T m = max(x_arr[i], y_arr[i]);
	    T w = m + log(exp(x_arr[i]-m) + exp(y_arr[i]-m));

	    if (abs(z_arr[i]-w) > thresh) {
		cout << "test_log_add<" << type_name<T>() << "," << S << "> failed:"
		     << " input=(" << x_arr[i] << "," << y_arr[i] << "), output=" << z_arr[i] 
		     << ", expected=" << w << ", delta=" << abs(z_arr[i]-w) << ", thresh=" << thresh << endl;
		exit(1);
	    }
	}
    }
}


template<typename T, int S>
void test_special_functions(std::mt19937 &rng)
{
    test_exp2<T,S> (rng);
    test_log2p<T,S> (rng);
    test_log_add<T,S> (rng);

    cout << "test_special_functions<" << type_name<T>() << "," << S << ">: pass" << endl;
}


int main(int argc, char **argv)
{
    std::random_device rd;
    std::mt19937 rng(rd());

    test_special_functions<float,4> (rng);
    test_special_functions<double,2> (rng);
#ifdef __AVX__
    test_special_functions<float,8> (rng);
    test_special_functions<double,4> (rng);
#endif

    return 0;
}
