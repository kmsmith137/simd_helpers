#include "simd_helpers/simd_debug.hpp"

using namespace std;

namespace simd_helpers {
#if 0
}  // emacs pacifier
#endif

// Convenient when defining templated unit tests which work for both integral and floating-point types
template<> inline constexpr int machine_epsilon()  { return 0; }
template<> inline constexpr int64_t machine_epsilon()  { return 0; }


// -------------------------------------------------------------------------------------------------
//
// "Basics": load / store / extract / constructors


template<typename T, unsigned int S, unsigned int N, typename std::enable_if<(N==0),int>::type = 0>
inline bool _check_extract(simd_t<T,S> x, const T *v) { return true; }

template<typename T, unsigned int S, unsigned int N, typename std::enable_if<(N>0),int>::type = 0>
inline bool _check_extract(simd_t<T,S> x, const T *v) 
{ 
    return _check_extract<T,S,N-1>(x,v) && (x.template extract<N-1>() == v[N-1]) && (extract_slow(x,N-1) == v[N-1]);
}


template<typename T, unsigned int S>
inline void test_load_store_extract(std::mt19937 &rng)
{
    vector<T> v = uniform_randvec<T> (rng, S, -1000, 1000);
    simd_t<T,S> x = simd_t<T,S>::loadu(&v[0]);

    bool extract_ok = _check_extract<T,S,S>(x, &v[0]);
    assert(extract_ok);

    // tests simd_t<T,S>::storeu()
    vector<T> w = vectorize(x);
    assert(strictly_equal(v,w));

    v = uniform_randvec<T> (rng, S, -1000, 1000);
    for (unsigned int s = 0; s < S; s++)
	set_slow(x, s, v[s]);

    w = vectorize(x);
    assert(strictly_equal(v,w));
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
	assert(vb[i] == vc[i+S]);
    }
};


// -------------------------------------------------------------------------------------------------
//
// Arithmetic ops


template<typename T, unsigned int S>
inline void test_unary_operation(const char *name, std::mt19937 &rng, simd_t<T,S> (*f1)(simd_t<T,S>), T (*f2)(T), T lo, T hi, T e)
{
    const T epsilon = e * machine_epsilon<T> ();

    simd_t<T,S> x = uniform_random_simd_t<T,S> (rng, lo, hi);
    simd_t<T,S> y = f1(x);

    vector<T> vx = vectorize(x);
    vector<T> vy = vectorize(y);

    vector<T> vz(S);
    for (unsigned int s = 0; s < S; s++)
	vz[s] = f2(vx[s]);

    if (maxdiff(vy,vz) <= epsilon)
	return;

    for (unsigned int s = 0; s < S; s++) {
	cerr << "test_unary_operation(" << name << "," << type_name<T>() << "," << S << ") failed\n"
	     << "    argument: " << x << endl
	     << "    output: " << y << endl
	     << "    expected output: " << vecstr(vz) << endl
	     << "    difference: [";

	for (unsigned int s = 0; s < S; s++)
	    cerr << " " << std::abs(vy[s]-vz[s]);

	cerr << "]\n"
	     << "    epsilon: " << epsilon << endl;

	exit(1);
    }
}


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
inline void test_binary_operator(const char *name, std::mt19937 &rng, simd_t<T,S> (*f1)(simd_t<T,S>, simd_t<T,S>), T (*f2)(T, T))
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

    if (maxdiff(w1,w2) > epsilon) {
	cerr << "test_binary_operator(" << name << "," << type_name<T>() << "," << S << ") failed\n"
	     << "   operand 1: " << x << endl
	     << "   operand 2: " << y << endl
	     << "   output: " << vecstr(w1) << endl
	     << "   expected output: " << vecstr(w2) << endl
	     << "   difference: [";

	for (unsigned int s = 0; s < S; s++)
	    cerr << " " << std::abs(w1[s]-w2[s]);

	cerr << "]\n";
	exit(1);
    }
}


template<typename T, unsigned int S>
inline void test_comparison_operator(const char *name, std::mt19937 &rng, smask_t<T,S> (*f1)(simd_t<T,S>, simd_t<T,S>), bool (*f2)(T, T))
{
    simd_t<T,S> x = uniform_random_simd_t<T,S> (rng, -10, 10);
    simd_t<T,S> dx = uniform_random_simd_t<T,S> (rng, 1, 10);

    vector<T> vx = vectorize(x);
    vector<T> vdx = vectorize(dx);
    vector<T> vy = vx;

    for (unsigned int s = 0; s < S; s++) {
	if (std::uniform_real_distribution<>()(rng) < 0.33)
	    vy[s] += vdx[s];
	else if (std::uniform_real_distribution<>()(rng) < 0.5)
	    vy[s] -= vdx[s];
    }

    simd_t<T,S> y = pack_simd_t<T,S> (vy);
    vector<smask_t<T> > vc = vectorize(f1(x,y));
    
    for (unsigned int s = 0; s < S; s++) {
	smask_t<T> expected_c = f2(vx[s],vy[s]) ? -1 : 0;
	if (vc[s] == expected_c)
	    continue;
	
	cerr << "test_comparison_operator(" << name << "," << type_name<T>() << "," << S << ") failed\n";
	exit(1);
    }
}


// Used to test apply_mask(), apply_inverse_mask(), and blendv().
template<typename T, unsigned int S>
inline void test_masking_operator(const char *name, std::mt19937 &rng, simd_t<T,S> (*f1)(smask_t<T,S>, simd_t<T,S>, simd_t<T,S>), T (*f2)(bool,T,T))
{
    smask_t<T,S> mask = uniform_random_simd_t<smask_t<T>,S> (rng, -1, 0);
    simd_t<T,S> a = uniform_random_simd_t<T,S> (rng, -10000, 10000);
    simd_t<T,S> b = uniform_random_simd_t<T,S> (rng, -10000, 10000);
    simd_t<T,S> c1 = f1(mask, a, b);

    vector<smask_t<T> > vmask = vectorize(mask);
    vector<T> va = vectorize(a);
    vector<T> vb = vectorize(b);
    vector<T> vc1 = vectorize(c1);
    
    vector<T> vc2(S);
    for (unsigned int s = 0; s < S; s++) {
	if (vmask[s] == 0)
	    vc2[s] = f2(false, va[s], vb[s]);
	else if (vmask[s] == smask_t<T>(-1))
	    vc2[s] = f2(true, va[s], vb[s]);	    
	else {
	    cerr << "internal error in test_masking_operator(" << name << "," << type_name<T>() << "," << S << ")\n";
	    exit(1);
	}
    }

    if (!strictly_equal(vc1, vc2)) {
	cerr << "test_masking_operator(" << name << "," << type_name<T>() << "," << S << ") failed\n"
	     << "   mask: " << mask << "\n"
	     << "   arg 1: " << a << "\n"
	     << "   arg 2: " << b << "\n"
	     << "   output: " << c1 << "\n"
	     << "   expected output: " << vecstr(vc2) << "\n";
	exit(1);
    }
}


// -------------------------------------------------------------------------------------------------


// Tests both horizontal_sum() and sum().
template<typename T, unsigned int S>
inline void test_horizontal_sum(std::mt19937 &rng)
{
    const T epsilon = 10000. * machine_epsilon<T> ();

    simd_t<T,S> x = uniform_random_simd_t<T,S> (rng, -1000, 1000);
    vector<T> vx = vectorize(x);

    T actual_sum = x.sum();

    T expected_sum = 0;
    for (unsigned int s = 0; s < S; s++)
	expected_sum += vx[s];

    simd_t<T,S> h = x.horizontal_sum();
    vector<T> vh = vectorize(h);

    vector<T> dh(S);
    for (unsigned int s = 0; s < S; s++)
	dh[s] = vh[s] - expected_sum;
	    
    if ((std::abs(actual_sum - expected_sum) > epsilon) || (maxabs(dh) > epsilon)) {
	cerr << "test_horizontal_sum(" << type_name<T>() << "," << S << ") failed\n"
	     << "   operand: " << x << endl
	     << "   expected sum: " << expected_sum << endl
	     << "   result of horizontal_sum(): " << h << ", difference=" << vecstr(dh) << "\n"
	     << "   result of sum(): " << actual_sum << ", difference=" << (actual_sum-expected_sum) << "\n";

	exit(1);
    }
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
template<typename T> inline T unary_minus(T x) { return -x; }

template<typename T> inline T std_min(T x, T y) { return std::min(x,y); }
template<typename T> inline T std_max(T x, T y) { return std::max(x,y); }
template<typename T> inline T std_abs(T x)      { return std::abs(x); }
template<typename T> inline T std_sqrt(T x)     { return std::sqrt(x); }

template<typename T, unsigned int S> inline simd_t<T,S> simd_min(simd_t<T,S> x, simd_t<T,S> y) { return x.min(y); }
template<typename T, unsigned int S> inline simd_t<T,S> simd_max(simd_t<T,S> x, simd_t<T,S> y) { return x.max(y); }
template<typename T, unsigned int S> inline simd_t<T,S> simd_abs(simd_t<T,S> x)   { return x.abs(); }
template<typename T, unsigned int S> inline simd_t<T,S> simd_sqrt(simd_t<T,S> x)  { return x.sqrt(); }

template<typename T> inline bool cmp_eq(T x, T y) { return (x == y); }
template<typename T> inline bool cmp_ne(T x, T y) { return (x != y); }
template<typename T> inline bool cmp_gt(T x, T y) { return (x > y); }
template<typename T> inline bool cmp_ge(T x, T y) { return (x >= y); }
template<typename T> inline bool cmp_lt(T x, T y) { return (x < y); }
template<typename T> inline bool cmp_le(T x, T y) { return (x <= y); }

template<typename T, unsigned int S> inline smask_t<T,S> simd_cmp_eq(simd_t<T,S> x, simd_t<T,S> y) { return x.compare_eq(y); }
template<typename T, unsigned int S> inline smask_t<T,S> simd_cmp_ne(simd_t<T,S> x, simd_t<T,S> y) { return x.compare_ne(y); }
template<typename T, unsigned int S> inline smask_t<T,S> simd_cmp_gt(simd_t<T,S> x, simd_t<T,S> y) { return x.compare_gt(y); }
template<typename T, unsigned int S> inline smask_t<T,S> simd_cmp_ge(simd_t<T,S> x, simd_t<T,S> y) { return x.compare_ge(y); }
template<typename T, unsigned int S> inline smask_t<T,S> simd_cmp_lt(simd_t<T,S> x, simd_t<T,S> y) { return x.compare_lt(y); }
template<typename T, unsigned int S> inline smask_t<T,S> simd_cmp_le(simd_t<T,S> x, simd_t<T,S> y) { return x.compare_le(y); }

template<typename T> inline T bitwise_and(T x, T y)     { return (x & y); }
template<typename T> inline T bitwise_or(T x, T y)      { return (x | y); }
template<typename T> inline T bitwise_xor(T x, T y)     { return (x ^ y); }
template<typename T> inline T bitwise_andnot(T x, T y)  { return (x & ~y); }
template<typename T> inline T bitwise_not(T x)          { return ~x; }

template<typename T, unsigned int S> inline simd_t<T,S> simd_bitwise_and(simd_t<T,S> x, simd_t<T,S> y)     { return x.bitwise_and(y); }
template<typename T, unsigned int S> inline simd_t<T,S> simd_bitwise_or(simd_t<T,S> x, simd_t<T,S> y)      { return x.bitwise_or(y); }
template<typename T, unsigned int S> inline simd_t<T,S> simd_bitwise_xor(simd_t<T,S> x, simd_t<T,S> y)     { return x.bitwise_xor(y); }
template<typename T, unsigned int S> inline simd_t<T,S> simd_bitwise_andnot(simd_t<T,S> x, simd_t<T,S> y)  { return x.bitwise_andnot(y); }
template<typename T, unsigned int S> inline simd_t<T,S> simd_bitwise_not(simd_t<T,S> x)                    { return x.bitwise_not(); }

template<typename T, unsigned int S> inline simd_t<T,S> simd_blendv(smask_t<T,S> mask, simd_t<T,S> a, simd_t<T,S> b)              { return blendv(mask,a,b); }
template<typename T, unsigned int S> inline simd_t<T,S> simd_apply_mask(smask_t<T,S> mask, simd_t<T,S> a, simd_t<T,S> b)          { return a.apply_mask(mask); }
template<typename T, unsigned int S> inline simd_t<T,S> simd_apply_inverse_mask(smask_t<T,S> mask, simd_t<T,S> a, simd_t<T,S> b)  { return a.apply_inverse_mask(mask); }

template<typename T> inline T std_blendv(bool mask, T a, T b)              { return mask ? a : b; }
template<typename T> inline T std_apply_mask(bool mask, T a, T b)          { return mask ? a : T(0); }
template<typename T> inline T std_apply_inverse_mask(bool mask, T a, T b)  { return mask ? T(0) : a; }


// Runs unit tests which are defined for every pair (T,S)
template<typename T, unsigned int S>
inline void test_TS(std::mt19937 &rng)
{
    test_load_store_extract<T,S>(rng);
    test_constructors<T,S>(rng);

    test_compound_assignment_operator(rng, assign_add< simd_t<T,S> >, assign_add<T>);   // operator+=
    test_compound_assignment_operator(rng, assign_sub< simd_t<T,S> >, assign_sub<T>);   // operator-=

    test_binary_operator("+", rng, binary_add< simd_t<T,S> >, binary_add<T>);
    test_binary_operator("-", rng, binary_sub< simd_t<T,S> >, binary_sub<T>);

    test_unary_operation<T> ("-", rng, unary_minus< simd_t<T,S> >, unary_minus<T>, -10000, 10000, 0);
    test_unary_operation<T> ("abs", rng, simd_abs<T,S>, std_abs<T>, -10000, 10000, 0);

    test_binary_operator("min", rng, simd_min<T,S>, std_min<T>);
    test_binary_operator("max", rng, simd_max<T,S>, std_max<T>);

    test_comparison_operator("compare_eq", rng, simd_cmp_eq<T,S>, cmp_eq<T>);
    test_comparison_operator("compare_ne", rng, simd_cmp_ne<T,S>, cmp_ne<T>);
    test_comparison_operator("compare_gt", rng, simd_cmp_gt<T,S>, cmp_gt<T>);
    test_comparison_operator("compare_ge", rng, simd_cmp_ge<T,S>, cmp_ge<T>);
    test_comparison_operator("compare_lt", rng, simd_cmp_lt<T,S>, cmp_lt<T>);
    test_comparison_operator("compare_le", rng, simd_cmp_le<T,S>, cmp_le<T>);

    test_masking_operator("blendv", rng, simd_blendv<T,S>, std_blendv<T>);
    test_masking_operator("apply_mask", rng, simd_apply_mask<T,S>, std_apply_mask<T>);
    test_masking_operator("apply_inverse_mask", rng, simd_apply_inverse_mask<T,S>, std_apply_inverse_mask<T>);

    test_horizontal_sum<T,S> (rng);
}

// Unit tests which are defined for a floating-point pair (T,S)
template<typename T, unsigned int S>
inline void test_floating_point_TS(std::mt19937 &rng)
{
    test_TS<T,S> (rng);

    test_compound_assignment_operator(rng, assign_mul< simd_t<T,S> >, assign_mul<T>);   // operator*=
    test_compound_assignment_operator(rng, assign_div< simd_t<T,S> >, assign_div<T>);   // operator/=

    test_binary_operator("*", rng, binary_mul< simd_t<T,S> >, binary_mul<T>);
    test_binary_operator("/", rng, binary_div< simd_t<T,S> >, binary_div<T>);

    test_unary_operation<T> ("sqrt", rng, simd_sqrt<T,S>, std_sqrt<T>, 10.0, 1000.0, 50.0);
}

// Unit tests which are defined for an integer pair (T,S)
template<typename T, unsigned int S>
inline void test_integer_TS(std::mt19937 &rng)
{
    test_TS<T,S> (rng);

    test_binary_operator("bitwise_and", rng, simd_bitwise_and<T,S>, bitwise_and<T>);
    test_binary_operator("bitwise_or", rng, simd_bitwise_or<T,S>, bitwise_or<T>);
    test_binary_operator("bitwise_xor", rng, simd_bitwise_xor<T,S>, bitwise_xor<T>);
    test_binary_operator("bitwise_andnot", rng, simd_bitwise_andnot<T,S>, bitwise_andnot<T>);
    test_unary_operation<T> ("bitwise_not", rng, simd_bitwise_not<T,S>, bitwise_not<T>, -10000, 10000, 0);
}

template<typename T>
inline void test_floating_point_T(std::mt19937 &rng)
{
    constexpr unsigned int S = 16 / sizeof(T);

    test_floating_point_TS<T,S> (rng);
    test_floating_point_TS<T,2*S> (rng);
    test_merging_constructor<T,S> (rng);
}

template<typename T>
inline void test_integer_T(std::mt19937 &rng)
{
    constexpr unsigned int S = 16 / sizeof(T);

    test_integer_TS<T,S> (rng);
    test_integer_TS<T,2*S> (rng);
    test_merging_constructor<T,S> (rng);
}

inline void test_all(std::mt19937 &rng)
{
    test_integer_T<int> (rng);
    test_integer_T<int64_t> (rng);
    test_floating_point_T<float> (rng);
    test_floating_point_T<double> (rng);
}

}   // namespace simd_helpers


int main(int argc, char **argv)
{
    std::random_device rd;
    std::mt19937 rng(rd());

    for (int iter = 0; iter < 1000; iter++)
	simd_helpers::test_all(rng);

    cout << "test-basics: pass\n";
    return 0;
}
