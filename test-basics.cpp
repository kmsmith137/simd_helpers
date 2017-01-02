#include "simd_helpers/simd_debug.hpp"

using namespace std;
using namespace simd_helpers;


// Convenient when defining templated unit tests which work for both integral and floating-point types
template<> inline constexpr int simd_helpers::machine_epsilon()  { return 0; }


// -------------------------------------------------------------------------------------------------
//
// "Basics": load / store / extract / constructors


template<typename T, unsigned int S, unsigned int N, typename std::enable_if<(N==0),int>::type = 0>
inline bool _check_extract(simd_t<T,S> x, const T *v) { return true; }

template<typename T, unsigned int S, unsigned int N, typename std::enable_if<(N>0),int>::type = 0>
inline bool _check_extract(simd_t<T,S> x, const T *v) { return _check_extract<T,S,N-1>(x,v) && (x.template extract<N-1>() == v[N-1]); }


template<typename T, unsigned int S>
inline void test_load_store_extract(std::mt19937 &rng)
{
    vector<T> v = uniform_randvec<T> (rng, S, -1000, 1000);
    simd_t<T,S> x = simd_t<T,S>::loadu(&v[0]);

    bool extract_ok = _check_extract<T,S,S>(x, &v[0]);
    assert(extract_ok);

    vector<T> w = vectorize(x);
    assert(strictly_equal(v,w));   // tests simd_t<T,S>::storeu()
}


template<typename T, unsigned int S>
inline void test_constructors(std::mt19937 &rng)
{
    T t = uniform_randvec<T> (rng, 1, -1000, 1000)[0];

    vector<T> vt = vectorize(simd_t<T,S> (t));
    vector<T> v0 = vectorize(simd_t<T,S>::zero());
    vector<T> vr = vectorize(simd_t<T,S>::range());

    for (unsigned int s = 0; s < S; s++) {
	assert(vt[s] == t);
	assert(v0[s] == 0);
	assert(vr[s] == s);
    }
}


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


// -------------------------------------------------------------------------------------------------
//
// Arithmetic ops


template<typename T, unsigned int S>
inline void test_compound_assignment_operator(std::mt19937 &rng, void (*f1)(simd_t<T,S> &, simd_t<T,S>), void (*f2)(T&, T))
{
    const T epsilon = 100 * machine_epsilon<T>();

    simd_t<T,S> x = uniform_random_simd_t<T,S> (rng, -10, 10);
    simd_t<T,S> y = uniform_random_simd_t<T,S> (rng, 1, 10);

    vector<T> vx = vectorize(x);
    vector<T> vy = vectorize(y);

    f1(x, y);
    vector<T> vz = vectorize(x);

    for (unsigned int s = 0; s < S; s++)
	f2(vx[s], vy[s]);

    assert(maxdiff(vx,vz) <= epsilon);
}


template<typename T, unsigned int S>
inline void test_binary_operator(std::mt19937 &rng, simd_t<T,S> (*f1)(simd_t<T,S>, simd_t<T,S>), T (*f2)(T, T))
{
    const T epsilon = 100 * machine_epsilon<T>();

    simd_t<T,S> x = uniform_random_simd_t<T,S> (rng, -10, 10);
    simd_t<T,S> y = uniform_random_simd_t<T,S> (rng, 1, 10);

    vector<T> vx = vectorize(x);
    vector<T> vy = vectorize(y);

    vector<T> w1 = vectorize(f1(x,y));

    vector<T> w2(S);
    for (unsigned int s = 0; s < S; s++)
	w2[s] = f2(vx[s], vy[s]);

    assert(maxdiff(w1,w2) <= epsilon);    
}


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


// -------------------------------------------------------------------------------------------------


template<typename T> inline void assign_add(T &x, T y) { x += y; }
template<typename T> inline void assign_sub(T &x, T y) { x -= y; }
template<typename T> inline void assign_mul(T &x, T y) { x *= y; }
template<typename T> inline void assign_div(T &x, T y) { x /= y; }

template<typename T> inline T binary_add(T x, T y) { return x + y; }
template<typename T> inline T binary_sub(T x, T y) { return x - y; }
template<typename T> inline T binary_mul(T x, T y) { return x * y; }
template<typename T> inline T binary_div(T x, T y) { return x / y; }


// Runs unit tests which are defined for every pair (T,S)
template<typename T, unsigned int S>
inline void test_all_TS(std::mt19937 &rng)
{
    test_load_store_extract<T,S>(rng);
    test_constructors<T,S>(rng);

    test_compound_assignment_operator(rng, assign_add< simd_t<T,S> >, assign_add<T>);   // operator+=
    test_compound_assignment_operator(rng, assign_sub< simd_t<T,S> >, assign_sub<T>);   // operator-=
    test_compound_assignment_operator(rng, assign_mul< simd_t<T,S> >, assign_mul<T>);   // operator*=
    // test_compound_assignment_operator(rng, assign_div< simd_t<T,S> >, assign_div<T>);   // operator/=

    test_binary_operator(rng, binary_add< simd_t<T,S> >, binary_add<T>);   // operator+
    test_binary_operator(rng, binary_sub< simd_t<T,S> >, binary_sub<T>);   // operator-
    test_binary_operator(rng, binary_mul< simd_t<T,S> >, binary_mul<T>);   // operator*
    // test_binary_operator(rng, binary_div< simd_t<T,S> >, binary_div<T>);   // operator/

    test_abs<T,S>(rng);
}


// Runs unit tests which are defined for every T
template<typename T>
inline void test_all_T(std::mt19937 &rng)
{
    constexpr unsigned int S = 16 / sizeof(T);

    test_all_TS<T,S> (rng);
    test_all_TS<T,2*S> (rng);
    test_merging_constructor<T,S> (rng);
}


int main(int argc, char **argv)
{
    std::random_device rd;
    std::mt19937 rng(rd());

    for (int iter = 0; iter < 1000; iter++) {
	test_all_T<int>(rng);
	test_all_T<float>(rng);
    }

    cout << "test-basics: pass\n";
    return 0;
}
