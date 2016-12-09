// g++ -std=c++11 -Wall -O3 -march=native -o test-linear-algebra-kernels test-linear-algebra-kernels.cpp

#include <cassert>
#include <iostream>

#include "simd_debug.hpp"

using namespace std;


template<typename T>
static vector<T> reference_multiply_lower(const vector<T> &mat, const vector<T> &v, int S, int N)
{
    int NN = (N*(N+1))/2;
    assert(mat.size() == NN*S);
    assert(v.size() == N*S);

    vector<T> ret(N*S, 0.0);
    
    for (int i = 0; i < N; i++) {
	for (int j = 0; j <= i; j++) {
	    int ij = (i*(i+1)*S)/2 + j*S;
	    for (int s = 0; s < S; s++)
		ret[i*S+s] += mat[ij+s] * v[j*S+s];
	}
    }

    return ret;
}


template<typename T>
static vector<T> reference_multiply_upper(const vector<T> &mat, const vector<T> &v, int S, int N)
{
    int NN = (N*(N+1))/2;
    assert(mat.size() == NN*S);
    assert(v.size() == N*S);

    vector<T> ret(N*S, 0.0);
    
    for (int i = 0; i < N; i++) {
	for (int j = 0; j <= i; j++) {
	    int ij = (i*(i+1)*S)/2 + j*S;
	    for (int s = 0; s < S; s++)
		ret[j*S+s] += mat[ij+s] * v[i*S+s];
	}
    }

    return ret;
}


template<typename T>
static vector<T> reference_multiply_symmetric(const vector<T> &mat, const vector<T> &v, int S, int N)
{
    int NN = (N*(N+1))/2;
    assert(mat.size() == NN*S);
    assert(v.size() == N*S);

    vector<T> ret(N*S, 0.0);
    
    for (int i = 0; i < N; i++) {
	for (int j = 0; j <= i; j++) {
	    int ij = (i*(i+1)*S)/2 + j*S;

	    for (int s = 0; s < S; s++)
		ret[i*S+s] += mat[ij+s] * v[j*S+s];

	    if (i == j)
		continue;

	    for (int s = 0; s < S; s++)
		ret[j*S+s] += mat[ij+s] * v[i*S+s];
	}
    }

    return ret;
}


// -------------------------------------------------------------------------------------------------


template<typename T, unsigned int S, unsigned int N>
void test_linear_algebra_kernels_N(std::mt19937 &rng)
{
    simd_trimatrix<T,S,N> m = random_simd_trimatrix<T,S,N> (rng);
    simd_ntuple<T,S,N> v = gaussian_random_simd_ntuple<T,S,N> (rng);
    simd_ntuple<T,S,N> x;

    // multiply_lower()
    simd_ntuple<T,S,N> w = m.multiply_lower(v);
    vector<T> wbuf = reference_multiply_lower(vectorize(m), vectorize(v), S, N);
    double epsilon = compare(vectorize(w), wbuf);
    assert(epsilon < 1.0e-6);
    
    // multiply_upper()
    w = m.multiply_upper(v);
    wbuf = reference_multiply_upper(vectorize(m), vectorize(v), S, N);
    epsilon = compare(vectorize(w), wbuf);
    assert(epsilon < 1.0e-6);
    
    // multiply_symmetric()
    w = m.multiply_symmetric(v);
    wbuf = reference_multiply_symmetric(vectorize(m), vectorize(v), S, N);
    epsilon = compare(vectorize(w), wbuf);
    assert(epsilon < 1.0e-6);

    // solve_lower()
    w = m.solve_lower(v);
    x = m.multiply_lower(w);
    epsilon = compare(vectorize(v), vectorize(x));
    assert(epsilon < 1.0e-6);

    // solve_upper()
    w = m.solve_upper(v);
    x = m.multiply_upper(w);
    epsilon = compare(vectorize(v), vectorize(x));
    assert(epsilon < 1.0e-6);

    // decholesky()
    simd_trimatrix<T,S,N> p = m.decholesky();
    w = p.multiply_symmetric(v);
    x = m.multiply_upper(v);
    x = m.multiply_lower(x);
    epsilon = compare(vectorize(w), vectorize(x));
    assert(epsilon < 1.0e-6);

    // cholesky()
    simd_trimatrix<T,S,N> m2 = p.cholesky();
    epsilon = compare(vectorize(m), vectorize(m2));
    assert(epsilon < 1.0e-6);

    // cholseky_flagged_in_place(), in case where all matrices are positive definite
    simd_trimatrix<T,S,N> m3 = p;
    simd_t<int,S> flags = m3.cholesky_in_place_checked(1.0e-3);
    int all_true = flags.compare_eq(1).test_all_ones();
    epsilon = compare(vectorize(m), vectorize(m3));
    assert(epsilon < 1.0e-6);
    assert(all_true);

    // cholseky_flagged_in_place(), with some random non positive definite matrices along for the ride

    simd_t<int,S> target_flags = uniform_random_simd_t<int,S> (rng, 0, 1);

    vector<int> v_f = vectorize(target_flags);
    vector<T> v_p = vectorize(p);

    for (unsigned int s = 0; s < S; s++) {
	if (v_f[s])
	    continue;
	
	// construct non-full-rank matrix by A A^T where A is shape (N-1,N)
	// note: _gaussian_randvec() is a helper function defined in simd_debug.hpp
	vector<T> amat = _gaussian_randvec<T> (rng, N*(N-1));
	for (int i = 0; i < N; i++) {
	    for (int j = 0; j <= i; j++) {
		T t = 0;
		for (int k = 0; k < N-1; k++)
		    t += amat[i*(N-1)+k] * amat[j*(N-1)+k];
		v_p[(i*(i+1)*S)/2 + j*S + s] = t;
	    }
	}
    }

    simd_trimatrix<T,S,N> m4 = pack_simd_trimatrix<T,S,N> (v_p);
    
    simd_t<int,S> actual_flags = m4.cholesky_in_place_checked(1.0e-2);
    simd_t<int,S> flag_agreement = actual_flags.compare_eq(target_flags);
    assert(flag_agreement.test_all_ones());

    vector<T> v_m = vectorize(m);
    vector<T> v_m4 = vectorize(m4);

    T num = 0;
    T den = 0;

    for (int s = 0; s < S; s++) {
	if (!v_f[s])
	    continue;

	for (int i = 0; i < (N*(N+1))/2; i++) {
	    T x = v_m[i*S+s];
	    T y = v_m4[i*S+s];
	    num += (x-y) * (x-y);
	    den += x*x + y*y;
	}
    }

    epsilon = (den > 0.0) ? sqrt(num/den) : 0.0;
    assert(epsilon < 1.0e-6);
}


template<typename T, unsigned int S>
void test_linear_algebra_kernels(std::mt19937 &rng)
{
    test_linear_algebra_kernels_N<T,S,1> (rng);
    test_linear_algebra_kernels_N<T,S,2> (rng);
    test_linear_algebra_kernels_N<T,S,3> (rng);
    test_linear_algebra_kernels_N<T,S,4> (rng);
    test_linear_algebra_kernels_N<T,S,5> (rng);
    test_linear_algebra_kernels_N<T,S,6> (rng);
    test_linear_algebra_kernels_N<T,S,7> (rng);
    test_linear_algebra_kernels_N<T,S,8> (rng);
}


void test_all_linear_algebra_kernels(std::mt19937 &rng)
{
    for (int iter = 0; iter < 100; iter++) {
	test_linear_algebra_kernels<float,4> (rng);
	test_linear_algebra_kernels<float,8> (rng);
    }

    cout << "test_all_linear_algebra_kernels: pass\n";
}


int main()
{
    std::random_device rd;
    std::mt19937 rng(rd());

    test_all_linear_algebra_kernels(rng);
    return 0;
}
