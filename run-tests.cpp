#include <functional>
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


template<typename T, int S, int N, typename std::enable_if<(N==0),int>::type = 0>
inline bool _check_extract(simd_t<T,S> x, const T *v) { return true; }

template<typename T, int S, int N, typename std::enable_if<(N>0),int>::type = 0>
inline bool _check_extract(simd_t<T,S> x, const T *v) 
{ 
    return _check_extract<T,S,N-1>(x,v) && (x.template extract<N-1>() == v[N-1]) && (extract_slow(x,N-1) == v[N-1]);
}


template<typename T, int S>
inline void test_basics(std::mt19937 &rng)
{
    vector<T> v = uniform_randvec<T> (rng, S, -1000, 1000);
    simd_t<T,S> x = simd_load<T,S>(&v[0]);

    // note that if extract<>() doesn't work, there's no point printing an error message, 
    // since the print routines use extract<>(), so their output would be unreliable
    bool extract_works = _check_extract<T,S,S>(x, &v[0]);
    assert(extract_works);

    // tests simd_t<T,S>::storeu()
    vector<T> w = vectorize(x);
    assert(strictly_equal(v,w));

    // generate a new random (x,v) pair using set_slow()
    v = uniform_randvec<T> (rng, S, -1000, 1000);
    for (int s = 0; s < S; s++)
	set_slow(x, s, v[s]);

    if (!_check_extract<T,S,S>(x, &v[0])) {
	cerr << "test_basics(" << type_name<T>() << "," << S << "): set_slow() didn't work as expected\n"
	     << "   input: " << vecstr(v) << "\n"
	     << "   output: " << x << "\n";

	exit(1);
    }
}


template<typename T, int S>
inline void test_constructors(std::mt19937 &rng)
{
    T t0 = uniform_rand<T> (rng, -1000, 1000);

    simd_t<T,S> t = simd_t<T,S>(t0);
    simd_t<T,S> z = simd_t<T,S>::zero();
    simd_t<T,S> r = simd_t<T,S>::range();

    for (int s = 0; s < S; s++) {
	assert(extract_slow(t,s) == t0);
	assert(extract_slow(z,s) == 0);
	assert(extract_slow(r,s) == s);
    }
}


// Tests "merging" constructor (simd_t<T,S>, simd_t<T,S>) -> simd_t<T,2*S>
// Also tests simd_t<T,2*S>::extract_half(), which inverts the merge.
template<typename T, int S>
inline void test_merging_constructor(std::mt19937 &rng) 
{
    simd_t<T,S> a = uniform_random_simd_t<T,S> (rng, -1000, 1000);
    simd_t<T,S> b = uniform_random_simd_t<T,S> (rng, -1000, 1000);

    simd_t<T,2*S> c(a, b);
    simd_t<T,S> c0 = c.template extract_half<0> ();
    simd_t<T,S> c1 = c.template extract_half<1> ();

    for (int s = 0; s < S; s++) {
	assert(extract_slow(c,s) == extract_slow(a,s));
	assert(extract_slow(c,s+S) == extract_slow(b,s));
	assert(extract_slow(c0,s) == extract_slow(a,s));
	assert(extract_slow(c1,s) == extract_slow(b,s));
    }
}


// -------------------------------------------------------------------------------------------------
//
// Arithmetic ops


template<typename T, int S>
inline void test_unary_operation(const char *name, std::mt19937 &rng, simd_t<T,S> (*f1)(simd_t<T,S>), T (*f2)(T), T lo, T hi, T e)
{
    const T epsilon = e * machine_epsilon<T> ();

    simd_t<T,S> x = uniform_random_simd_t<T,S> (rng, lo, hi);
    simd_t<T,S> y = f1(x);

    simd_t<T,S> y_exp;
    for (int s = 0; s < S; s++)
	set_slow(y_exp, s, f2(extract_slow(x,s)));

    if (maxdiff(y,y_exp) <= epsilon)
	return;

    for (int s = 0; s < S; s++) {
	cerr << "test_unary_operation(" << name << "," << type_name<T>() << "," << S << ") failed\n"
	     << "    argument: " << x << endl
	     << "    output: " << y << endl
	     << "    expected output: " << y_exp << endl
	     << "    difference: " << (y - y_exp) << endl
	     << "    epsilon: " << epsilon << endl;

	exit(1);
    }
}


template<typename T, int S>
inline void test_compound_assignment_operator(const char *name, std::mt19937 &rng, void (*f1)(simd_t<T,S> &, simd_t<T,S>), void (*f2)(T&, T), T lo1, T hi1, T lo2, T hi2, T e)
{
    const T epsilon = e * machine_epsilon<T>();

    simd_t<T,S> x_in = uniform_random_simd_t<T,S> (rng, lo1, hi1);
    simd_t<T,S> y = uniform_random_simd_t<T,S> (rng, lo2, hi2);

    simd_t<T,S> x_out = x_in;
    f1(x_out, y);

    simd_t<T,S> x_exp;
    for (int s = 0; s < S; s++) {
	T t = extract_slow(x_in, s);
	f2(t, extract_slow(y, s));
	set_slow(x_exp, s, t);
    }

    if (maxdiff(x_out,x_exp) > epsilon) {
	cerr << "test_compound_assignment_operator(" << name << "," << type_name<T>() << "," << S << ") failed\n"
	     << "   arg 1 (input): " << x_in << "\n"
	     << "   arg 2: " << y << "\n"
	     << "   arg 1 (actual output): " << x_out << "\n"
	     << "   arg 2 (expected output): " << x_exp << "\n"
	     << "   difference: " << (x_out-x_exp) << "\n"
	     << "   epsilon: " << epsilon << "\n";

	exit(1);
    }
}


template<typename T, int S>
inline void test_binary_operator(const char *name, std::mt19937 &rng, simd_t<T,S> (*f1)(simd_t<T,S>, simd_t<T,S>), T (*f2)(T, T), T lo1, T hi1, T lo2, T hi2, T e)
{
    const T epsilon = e * machine_epsilon<T>();

    simd_t<T,S> x = uniform_random_simd_t<T,S> (rng, lo1, hi1);
    simd_t<T,S> y = uniform_random_simd_t<T,S> (rng, lo2, hi2);
    simd_t<T,S> z = f1(x, y);

    simd_t<T,S> z_exp;
    for (int s = 0; s < S; s++)
	set_slow(z_exp, s, f2(extract_slow(x,s), extract_slow(y,s)));

    if (maxdiff(z,z_exp) > epsilon) {
	cerr << "test_binary_operator(" << name << "," << type_name<T>() << "," << S << ") failed\n"
	     << "   operand 1: " << x << endl
	     << "   operand 2: " << y << endl
	     << "   output: " << z << endl
	     << "   expected output: " << z_exp << endl
	     << "   difference: " << (z-z_exp) << endl
	     << "   epsilon: " << epsilon << endl;

	exit(1);
    }
}


template<typename T, int S>
inline void test_comparison_operator(const char *name, std::mt19937 &rng, smask_t<T,S> (*f1)(simd_t<T,S>, simd_t<T,S>), bool (*f2)(T, T))
{
    simd_t<T,S> x = uniform_random_simd_t<T,S> (rng, -10, 10);
    simd_t<T,S> y;

    for (int s = 0; s < S; s++) {
	T t = extract_slow(x, s);
	if (std::uniform_real_distribution<>()(rng) < 0.33)
	    t += uniform_rand<T> (rng, 1, 10);
	else if (std::uniform_real_distribution<>()(rng) < 0.5)
	    t -= uniform_rand<T> (rng, 1, 10);
	set_slow(y, s, t);
    }

    smask_t<T,S> c = f1(x, y);
    smask_t<T,S> c_exp;

    for (int s = 0; s < S; s++) {
	bool b = f2(extract_slow(x,s), extract_slow(y,s));
	smask_t<T> m = b ? -1 : 0;
	set_slow(c_exp, s, m);
    }

    if (!strictly_equal(c, c_exp)) {
	cerr << "test_comparison_operator(" << name << "," << type_name<T>() << "," << S << ") failed\n"
	     << "    argument 1: " << x << endl
	     << "    argument 2: " << y << endl
	     << "    output: " << c << endl
	     << "    expected: " << c_exp << endl;

	exit(1);
    }
}


// Used to test apply_mask(), apply_inverse_mask(), and blendv().
template<typename T, int S>
inline void test_masking_operator(const char *name, std::mt19937 &rng, simd_t<T,S> (*f1)(smask_t<T,S>, simd_t<T,S>, simd_t<T,S>), T (*f2)(bool,T,T))
{
    smask_t<T,S> mask = uniform_random_simd_t<smask_t<T>,S> (rng, -1, 0);
    simd_t<T,S> a = uniform_random_simd_t<T,S> (rng, -10000, 10000);
    simd_t<T,S> b = uniform_random_simd_t<T,S> (rng, -10000, 10000);
    simd_t<T,S> c = f1(mask, a, b);
    simd_t<T,S> c_exp;

    for (int s = 0; s < S; s++) {
	T aa = extract_slow(a, s);
	T bb = extract_slow(b, s);

	if (extract_slow(mask,s) == 0)
	    set_slow(c_exp, s, f2(false,aa,bb));
	else if (extract_slow(mask,s) == -1)
	    set_slow(c_exp, s, f2(true,aa,bb));	    
	else {
	    cerr << "internal error in test_masking_operator(" << name << "," << type_name<T>() << "," << S << ")\n";
	    exit(1);
	}
    }

    if (!strictly_equal(c, c_exp)) {
	cerr << "test_masking_operator(" << name << "," << type_name<T>() << "," << S << ") failed\n"
	     << "   mask: " << mask << "\n"
	     << "   arg 1: " << a << "\n"
	     << "   arg 2: " << b << "\n"
	     << "   output: " << c << "\n"
	     << "   expected output: " << c_exp << "\n";
	exit(1);
    }
}



// -------------------------------------------------------------------------------------------------
//
// "Boolean reducers": these are different enough that I decided not to make a generic test routine


template<typename T, int S>
inline void test_is_all_ones(std::mt19937 &rng)
{
    simd_t<T,S> x = simd_t<T,S>(-1);
    int expected_result = 1;

    for (int s = 0; s < S; s++) {
	if (std::uniform_real_distribution<>()(rng) < 1/(1.5*S)) {
	    T t = uniform_rand<T>(rng, -10, 10);
	    set_slow(x, s, t);
	    if (t != -1)
		expected_result = 0;
	}
    }

    if (x.is_all_ones() != expected_result) {
	cerr << "test_is_all_ones(" << type_name<T>() << "," << S << ") failed\n"
	     << "   argument = " << x << "\n"
	     << "   result = " << x.is_all_ones() << "\n"
	     << "   expected result = " << expected_result << "\n";

	exit(1);
    }
}


template<typename T, int S>
inline void test_is_all_zeros(std::mt19937 &rng)
{
    simd_t<T,S> x = simd_t<T,S>::zero();
    int expected_result = 1;

    for (int s = 0; s < S; s++) {
	if (std::uniform_real_distribution<>()(rng) < 1/(1.5*S)) {
	    T t = uniform_rand<T>(rng, -10, 10);
	    set_slow(x, s, t);
	    if (t != 0)
		expected_result = 0;
	}
    }

    if (x.is_all_zeros() != expected_result) {
	cerr << "test_is_all_zeros(" << type_name<T>() << "," << S << ") failed\n"
	     << "   argument = " << x << "\n"
	     << "   result = " << x.is_all_zeros() << "\n"
	     << "   expected result = " << expected_result << "\n";

	exit(1);
    }
}


template<typename T, int S>
inline void test_is_all_zeros_masked(std::mt19937 &rng)
{
    simd_t<T,S> x = simd_t<T,S>::zero();
    smask_t<T,S> mask = uniform_random_simd_t<smask_t<T>,S> (rng, -1, 0);
    int expected_result = 1;

    for (int s = 0; s < S; s++) {
	if (std::uniform_real_distribution<>()(rng) < 1.5/S) {
	    T t = uniform_rand<T>(rng, -10, 10);
	    set_slow(x, s, t);
	    if ((t != 0) && (extract_slow(mask,s) != 0))
		expected_result = 0;
	}
    }

    if (x.is_all_zeros_masked(mask) != expected_result) {
	cerr << "test_is_all_zeros_masked(" << type_name<T>() << "," << S << ") failed\n"
	     << "   argument = " << x << "\n"
	     << "   mask = " << mask << "\n"
	     << "   result = " << x.is_all_zeros_masked(mask) << "\n"
	     << "   expected result = " << expected_result << "\n";

	exit(1);
    }
}


template<typename T, int S>
inline void test_is_all_zeros_inverse_masked(std::mt19937 &rng)
{
    simd_t<T,S> x = simd_t<T,S>::zero();
    smask_t<T,S> mask = uniform_random_simd_t<smask_t<T>,S> (rng, -1, 0);
    int expected_result = 1;

    for (int s = 0; s < S; s++) {
	if (std::uniform_real_distribution<>()(rng) < 1.5/S) {
	    T t = uniform_rand<T>(rng, -10, 10);
	    set_slow(x, s, t);
	    if ((t != 0) && (extract_slow(mask,s) == 0))
		expected_result = 0;
	}
    }

    if (x.is_all_zeros_inverse_masked(mask) != expected_result) {
	cerr << "test_is_all_zeros_inverse_masked(" << type_name<T>() << "," << S << ") failed\n"
	     << "   argument = " << x << "\n"
	     << "   mask = " << mask << "\n"
	     << "   result = " << x.is_all_zeros_inverse_masked(mask) << "\n"
	     << "   expected result = " << expected_result << "\n";

	exit(1);
    }
}


// -------------------------------------------------------------------------------------------------


// For horizontal_sum(), horizontal_max(), etc.
template<typename T, int S>
inline void test_horizontal_reducer(const char *name, std::mt19937 &rng, simd_t<T,S> (*f1)(simd_t<T,S>), T (*f2)(simd_t<T,S>), T (*f3)(T,T))
{
    const T epsilon = S * 2000 * machine_epsilon<T> ();

    simd_t<T,S> x = uniform_random_simd_t<T,S> (rng, -1000, 1000);
    simd_t<T,S> hval = f1(x);
    T sval = f2(x);

    T cval = extract_slow(x, 0);
    for (int s = 1; s < S; s++)
	cval = f3(cval, extract_slow(x,s));
    
    if ((std::abs(sval-cval) > epsilon) || (maxdiff(hval, simd_t<T,S>(cval)) > epsilon)) {
	cerr << "test_horizontal_reducer(" 
	     << name << "," << type_name<T>() << "," << S << ") failed\n"
	     << "   operand: " << x << endl
	     << "   expected result: " << cval << endl
	     << "   result of vector->scalar reducer: " << sval << "\n"
	     << "   difference=" << (sval-cval) << "\n"
	     << "   result of vector->vector reducer: " << hval << "\n"
	     << "   difference=" << (hval-cval) << "]\n"
	     << "   epsilon: " << epsilon << endl;

	exit(1);
    }
}


// -------------------------------------------------------------------------------------------------
//
// void test_align<T,S>(rng)
//    Tests all align() operations for a given (T,S)
//
// simd_t<T,S> align_slow(int A, simd_t<T,S> x, simd_t<T,S> y);
//    Slow version of align() in which A is a function argument, not a template parameter.
//
// void align_slow(int A, simd_ntuple<T,S,N> &dst, const simd_ntuple<T,S,N> &x, const simd_ntuple<T,S,N> &src)
//    Slow version of align() in which A is a function argument, not a template parameter.


template<typename T, int S, int A1=S+1, typename std::enable_if<(A1==0),int>::type = 0>
static simd_t<T,S> align_slow(int A, simd_t<T,S> x, simd_t<T,S> y)
{
    throw runtime_error("align_slow internal error");
}

template<typename T, int S, int A1=S+1, typename std::enable_if<(A1>0),int>::type = 0>
static simd_t<T,S> align_slow(int A, simd_t<T,S> x, simd_t<T,S> y)
{
    return (A == A1-1) ? (align<A1-1>(x,y)) : align_slow<T,S,A1-1>(A,x,y);
}


template<typename T, int S, int N, int A1=S+1, typename std::enable_if<(A1==0),int>::type = 0>
static void align_slow(int A, simd_ntuple<T,S,N> &dst, const simd_ntuple<T,S,N> &x, const simd_ntuple<T,S,N> &y)
{
    throw runtime_error("align_slow internal error");
}

template<typename T, int S, int N, int A1=S+1, typename std::enable_if<(A1>0),int>::type = 0>
static void align_slow(int A, simd_ntuple<T,S,N> &dst, const simd_ntuple<T,S,N> &x, const simd_ntuple<T,S,N> &y)
{
    if (A == A1-1)
	align<A1-1> (dst, x, y);
    else
	align_slow<T,S,N,A1-1> (A, dst, x, y);
}


template<typename T, int S, int N>
static void test_align_ntuple(std::mt19937 &rng)
{
    for (int A = 0; A <= S; A++) {
	simd_ntuple<T,S,N> x = uniform_random_simd_ntuple<T,S,N> (rng, 0, 100);
	simd_ntuple<T,S,N> y = uniform_random_simd_ntuple<T,S,N> (rng, 0, 100);

	simd_ntuple<T,S,N> t;
	align_slow(A, t, x, y);

	vector<T> vx = vectorize(x);
	vector<T> vy = vectorize(y);
	vector<T> vt = vectorize(t);

	for (int i = 0; i < N; i++) {
	    for (int s = 0; s < S; s++) {
		T u = vt[i*S+s];
		T v = (s+A < S) ? vx[i*S+(s+A)] : vy[i*S+(s+A-S)];
		assert(u == v);
	    }
	}
    }
}


template<typename T, int S>
static void test_align(std::mt19937 &rng)
{
    for (int A = 0; A <= S; A++) {
	simd_t<T,S> x = uniform_random_simd_t<T,S> (rng, 0, 100);
	simd_t<T,S> y = uniform_random_simd_t<T,S> (rng, 0, 100);
	simd_t<T,S> t = align_slow(A, x, y);

	vector<T> vx = vectorize(x);
	vector<T> vy = vectorize(y);
	vector<T> vt = vectorize(t);

	for (int s = 0; s < S; s++) {
	    T u = vt[s];
	    T v = (s+A < S) ? vx[s+A] : vy[s+A-S];
	    assert(u == v);
	}
    }

    test_align_ntuple<T,S,1> (rng);
    test_align_ntuple<T,S,2> (rng);
    test_align_ntuple<T,S,3> (rng);
    test_align_ntuple<T,S,4> (rng);
}


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
//
// Upsample/downsample


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


// Test upsample for a given (S,N) pair.
template<int S, int N>
static void test_upsample(std::mt19937 &rng)
{
    test_upsample1<int,S,N> (rng);
    test_upsample1<float,S,N> (rng);
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


// Test downsample for a given (S,N) pair.
template<int S, int N>
static void test_downsample(std::mt19937 &rng)
{
    test_downsample1<int,S,N> (rng, [](int x, int y) { return x+y; }, "+");
    test_downsample1<float,S,N> (rng, [](float x, float y) { return x+y; }, "+");

    test_downsample1<int,S,N,simd_max<int,S>> (rng, [](int x, int y) { return max(x,y); }, "max");
    test_downsample1<float,S,N,simd_max<float,S>> (rng, [](float x, float y) { return max(x,y); }, "max");
    
    test_downsample1<int,S,N,simd_bitwise_or<int,S>> (rng, [](int x, int y) { return x|y; }, "|");
}


// -------------------------------------------------------------------------------------------------



template<typename T>
static vector<T> reference_multiply_lower(const vector<T> &mat, const vector<T> &v, int S, int N)
{
    int NN = (N*(N+1))/2;
    assert((int)mat.size() == NN*S);
    assert((int)v.size() == N*S);

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
    assert((int)mat.size() == NN*S);
    assert((int)v.size() == N*S);

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
    assert((int)mat.size() == NN*S);
    assert((int)v.size() == N*S);

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


template<typename T>
void reference_transpose(T *dst, const T *src, int S, int N)
{
    // Number of spectator indices
    int M = S/N;

    // Shape: (N,M,N)
    for (int i = 0; i < N; i++)
	for (int j = 0; j < M; j++)
	    for (int k = 0; k < N; k++)
		dst[i*M*N+j*N+k] = src[k*M*N+j*N+i];
}


template<typename T>
void reference_btranspose(T *dst, const T *src, int S, int N)
{
    // Number of spectator indices
    int M = S/N;

    // Shape: (N,N,M)
    for (int i = 0; i < N; i++)
	for (int j = 0; j < N; j++)
	    for (int k = 0; k < M; k++)
		dst[i*M*N+j*M+k] = src[j*M*N+i*M+k];
}


template<typename T, int S, int N>
void test_transpose(std::mt19937 &rng)
{
    vector<T> src = uniform_randvec<T> (rng, S*N, 0, 1000000);
    vector<T> dst1(S*N);
    vector<T> dst2(S*N);

    simd_ntuple<T,S,N> t;
    t.loadu(&src[0]);
    simd_transpose(t);
    t.storeu(&dst1[0]);

    reference_transpose(&dst2[0], &src[0], S, N);

    if (!strictly_equal(S*N, &dst1[0], &dst2[0])) {
	cerr << "test_transpose(T=" << type_name<T>() << ",S=" << S << ",N=" << N << ") failed\n";
	exit(1);
    }
}


template<typename T, int S, int N>
void test_btranspose(std::mt19937 &rng)
{
    vector<T> src = uniform_randvec<T> (rng, S*N, 0, 1000000);
    vector<T> dst1(S*N);
    vector<T> dst2(S*N);

    simd_ntuple<T,S,N> t;
    t.loadu(&src[0]);
    simd_btranspose(t);
    t.storeu(&dst1[0]);

    reference_btranspose(&dst2[0], &src[0], S, N);

    if (!strictly_equal(S*N, &dst1[0], &dst2[0])) {
	cerr << "test_btranspose(T=" << type_name<T>() << ",S=" << S << ",N=" << N << ") failed\n";
	exit(1);
    }
}


// -------------------------------------------------------------------------------------------------


template<typename T, int S, int N>
void test_linear_algebra_kernels_N(std::mt19937 &rng)
{
    double epsilon0 = 10. * machine_epsilon<T> ();

    simd_trimatrix<T,S,N> m = random_simd_trimatrix<T,S,N> (rng);
    simd_ntuple<T,S,N> v = gaussian_random_simd_ntuple<T,S,N> (rng);
    simd_ntuple<T,S,N> x;

    // multiply_lower()
    simd_ntuple<T,S,N> w = m.multiply_lower(v);
    vector<T> wbuf = reference_multiply_lower(vectorize(m), vectorize(v), S, N);
    double epsilon = compare(vectorize(w), wbuf);
    assert(epsilon < epsilon0);
    
    // multiply_upper()
    w = m.multiply_upper(v);
    wbuf = reference_multiply_upper(vectorize(m), vectorize(v), S, N);
    epsilon = compare(vectorize(w), wbuf);
    assert(epsilon < epsilon0);
    
    // multiply_symmetric()
    w = m.multiply_symmetric(v);
    wbuf = reference_multiply_symmetric(vectorize(m), vectorize(v), S, N);
    epsilon = compare(vectorize(w), wbuf);
    assert(epsilon < epsilon0);

    // solve_lower()
    w = m.solve_lower(v);
    x = m.multiply_lower(w);
    epsilon = compare(vectorize(v), vectorize(x));
    assert(epsilon < epsilon0);

    // solve_upper()
    w = m.solve_upper(v);
    x = m.multiply_upper(w);
    epsilon = compare(vectorize(v), vectorize(x));
    assert(epsilon < epsilon0);

    // decholesky()
    simd_trimatrix<T,S,N> p = m.decholesky();
    w = p.multiply_symmetric(v);
    x = m.multiply_upper(v);
    x = m.multiply_lower(x);
    epsilon = compare(vectorize(w), vectorize(x));
    assert(epsilon < epsilon0);

    // cholesky()
    simd_trimatrix<T,S,N> m2 = p.cholesky();
    epsilon = compare(vectorize(m), vectorize(m2));
    assert(epsilon < epsilon0);

    // cholseky_flagged_in_place(), in case where all matrices are positive definite
    simd_trimatrix<T,S,N> m3 = p;
    smask_t<T,S> flags = m3.cholesky_in_place_checked(1.0e-3);
    epsilon = compare(vectorize(m), vectorize(m3));
    assert(epsilon < epsilon0);
    assert(flags.is_all_ones());

    // cholseky_flagged_in_place(), with some random non positive definite matrices along for the ride

    smask_t<T,S> target_flags = uniform_random_simd_t<smask_t<T>,S> (rng, -1, 0);

    vector<smask_t<T> > v_f = vectorize(target_flags);
    vector<T> v_p = vectorize(p);

    for (int s = 0; s < S; s++) {
	if (v_f[s])
	    continue;
	
	// construct non-full-rank matrix by A A^T where A is shape (N-1,N)
	// note: gaussian_randvec() is defined in simd_debug.hpp
	vector<T> amat = gaussian_randvec<T> (rng, N*(N-1));
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
    
    smask_t<T,S> actual_flags = m4.cholesky_in_place_checked(1.0e-2);
    int flags_agree = actual_flags.compare_eq(target_flags).is_all_ones();
    assert(flags_agree);

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
    assert(epsilon < epsilon0);
}


template<typename T, int S>
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


// -------------------------------------------------------------------------------------------------


template<int S>
inline void test_float16_kernels(std::mt19937 &rng)
{
    vector<float> v(2*S, -1.0);

    simd_ntuple<float,S,2> x = uniform_random_simd_ntuple<float,S,2> (rng, 0.0, 1.0);
    simd_store_float16(&v[0], x);
    
    simd_ntuple<float,S,2> y;
    simd_load_float16(y, &v[0]);

    float epsilon = maxdiff(x,y);
    assert(epsilon < 1.0e-3);
}


// -------------------------------------------------------------------------------------------------


// Runs unit tests which are defined for every pair (T,S)
template<typename T, int S>
inline void test_TS(std::mt19937 &rng)
{
    test_basics<T,S>(rng);
    test_constructors<T,S>(rng);
    test_align<T,S>(rng);

    test_compound_assignment_operator<T,S> ("+=", rng, 
					    [](simd_t<T,S> &x, simd_t<T,S> y) { x += y; }, 
					    [](T &x, T y) { x += y; },
					    -10, 10, -10, 10, 20);

    test_compound_assignment_operator<T,S> ("-=", rng, 
					    [](simd_t<T,S> &x, simd_t<T,S> y) { x -= y; }, 
					    [](T &x, T y) { x -= y; },
					    -10, 10, -10, 10, 20);

    test_compound_assignment_operator<T,S> ("*=", rng, 
					    [](simd_t<T,S> &x, simd_t<T,S> y) { x *= y; }, 
					    [](T &x, T y) { x *= y; },
					    -10, 10, -10, 10, 200);

    test_binary_operator<T,S> ("+", rng, 
			       [] (simd_t<T,S> x, simd_t<T,S> y) { return x+y; }, 
			       [](T x, T y) { return x+y; },
			       -10, 10, -10, 10, 20);

    test_binary_operator<T,S> ("-", rng, 
			       [] (simd_t<T,S> x, simd_t<T,S> y) { return x-y; }, 
			       [](T x, T y) { return x-y; },
			       -10, 10, -10, 10, 20);

    test_binary_operator<T,S> ("*", rng, 
			       [] (simd_t<T,S> x, simd_t<T,S> y) { return x*y; }, 
			       [](T x, T y) { return x*y; },
			       -10, 10, -10, 10, 200);

    test_binary_operator<T,S> ("min", rng, 
			       [] (simd_t<T,S> x, simd_t<T,S> y) { return x.min(y); }, 
			       [](T x, T y) { return min(x,y); },
			       -10, 10, -10, 10, 0);

    test_binary_operator<T,S> ("max", rng, 
			       [] (simd_t<T,S> x, simd_t<T,S> y) { return x.max(y); }, 
			       [](T x, T y) { return max(x,y); },
			       -10, 10, -10, 10, 0);

    test_unary_operation<T,S> ("-", rng, 
			       [](simd_t<T,S> t) { return -t; },
			       [](T t) { return -t; }, 
			       -10000, 10000, 0);

    test_unary_operation<T,S> ("abs", rng, 
			       [](simd_t<T,S> t) { return t.abs(); },
			       [](T t) { return std::abs(t); },
			       -10000, 10000, 0);

    test_comparison_operator<T,S> ("compare_eq", rng, 
				   [](simd_t<T,S> x, simd_t<T,S> y) { return x.compare_eq(y); }, 
				   [](T x, T y) { return x == y; });

    test_comparison_operator<T,S> ("compare_ne", rng, 
				   [](simd_t<T,S> x, simd_t<T,S> y) { return x.compare_ne(y); }, 
				   [](T x, T y) { return x != y; });

    test_comparison_operator<T,S> ("compare_gt", rng, 
				   [](simd_t<T,S> x, simd_t<T,S> y) { return x.compare_gt(y); }, 
				   [](T x, T y) { return x > y; });

    test_comparison_operator<T,S> ("compare_ge", rng, 
				   [](simd_t<T,S> x, simd_t<T,S> y) { return x.compare_ge(y); }, 
				   [](T x, T y) { return x >= y; });

    test_comparison_operator<T,S> ("compare_lt", rng, 
				   [](simd_t<T,S> x, simd_t<T,S> y) { return x.compare_lt(y); }, 
				   [](T x, T y) { return x < y; });

    test_comparison_operator<T,S> ("compare_le", rng, 
				   [](simd_t<T,S> x, simd_t<T,S> y) { return x.compare_le(y); }, 
				   [](T x, T y) { return x <= y; });

    test_masking_operator<T,S> ("blendv", rng, 
				[](smask_t<T,S> m, simd_t<T,S> x, simd_t<T,S> y) { return blendv(m,x,y); },
				[](bool m, T x, T y) { return m ? x : y; });

    test_masking_operator<T,S> ("apply_mask", rng, 
				[](smask_t<T,S> m, simd_t<T,S> x, simd_t<T,S> y) { return x.apply_mask(m); },
				[](bool m, T x, T y) { return m ? x : 0; });

    test_masking_operator<T,S> ("apply_inverse_mask", rng, 
				[](smask_t<T,S> m, simd_t<T,S> x, simd_t<T,S> y) { return x.apply_inverse_mask(m); },
				[](bool m, T x, T y) { return m ? 0 : x; });

    test_horizontal_reducer<T,S> ("sum", rng,
				  [](simd_t<T,S> x) { return x.horizontal_sum(); },
				  [](simd_t<T,S> x) { return x.sum(); },
				  [](T x, T y) { return x+y; });

    test_horizontal_reducer<T,S> ("max", rng,
				  [](simd_t<T,S> x) { return x.horizontal_max(); },
				  [](simd_t<T,S> x) { return x.max(); },
				  [](T x, T y) { return std::max(x,y); });
}


// Unit tests which are defined for a floating-point pair (T,S)
template<typename T, int S>
inline void test_floating_point_TS(std::mt19937 &rng)
{
    test_TS<T,S> (rng);

    test_compound_assignment_operator<T,S> ("/=", rng, 
					    [](simd_t<T,S> &x, simd_t<T,S> y) { x /= y; }, 
					    [](T &x, T y) { x /= y; },
					    -100, 100, 1, 10, 200);

    test_binary_operator<T,S> ("/", rng, 
			       [] (simd_t<T,S> x, simd_t<T,S> y) { return x/y; }, 
			       [](T x, T y) { return x/y; },
			       -100, 100, 1, 10, 200);

    test_unary_operation<T,S> ("sqrt", rng, 
			       [](simd_t<T,S> t) { return t.sqrt(); },
			       [](T t) { return std::sqrt(t); },
			       10.0, 1000.0, 50.0);
}


// Unit tests which are defined for an integer pair (T,S)
template<typename T, int S>
inline void test_integer_TS(std::mt19937 &rng)
{
    test_TS<T,S> (rng);

    test_is_all_ones<T,S> (rng);
    test_is_all_zeros<T,S> (rng);
    test_is_all_zeros_masked<T,S> (rng);
    test_is_all_zeros_inverse_masked<T,S> (rng);

    test_binary_operator<T,S> ("&", rng, 
			       [](simd_t<T,S> x, simd_t<T,S> y) { return x & y; }, 
			       [](T x, T y) { return x & y; },
			       -100, 100, -100, 100, 0);

    test_binary_operator<T,S> ("|", rng, 
			       [](simd_t<T,S> x, simd_t<T,S> y) { return x | y; }, 
			       [](T x, T y) { return x | y; },
			       -100, 100, -100, 100, 0);

    test_binary_operator<T,S> ("^", rng, 
			       [](simd_t<T,S> x, simd_t<T,S> y) { return x ^ y; }, 
			       [](T x, T y) { return x ^ y; },
			       -100, 100, -100, 100, 0);

    test_binary_operator<T,S> ("<<", rng, 
			       [](simd_t<T,S> x, simd_t<T,S> y) { return x << y; }, 
			       [](T x, T y) { return x << y; },
			       0, 100, 0, 10, 0);

    test_binary_operator<T,S> (">>", rng, 
			       [](simd_t<T,S> x, simd_t<T,S> y) { return x >> y; }, 
			       [](T x, T y) { return x >> y; },
			       0, 10000000, 0, 10, 0);

    test_binary_operator<T,S> ("bitwise_andnot", rng, 
			       [](simd_t<T,S> x, simd_t<T,S> y) { return x.bitwise_andnot(y); }, 
			       [](T x, T y) { return x & ~y; },
			       -100, 100, -100, 100, 0);

    test_unary_operation<T,S> ("~", rng,
			       [](simd_t<T,S> x) { return x.bitwise_not(); },
			       [](T t) { return ~t; },
			       -100, 100, 0);

    test_compound_assignment_operator<T,S> ("&=", rng, 
					    [](simd_t<T,S> &x, simd_t<T,S> y) { x &= y; }, 
					    [](T &x, T y) { x &= y; },
					    -100, 100, -100, 100, 0);

    test_compound_assignment_operator<T,S> ("|=", rng, 
					    [](simd_t<T,S> &x, simd_t<T,S> y) { x |= y; }, 
					    [](T &x, T y) { x |= y; },
					    -100, 100, -100, 100, 0);

    test_compound_assignment_operator<T,S> ("^=", rng, 
					    [](simd_t<T,S> &x, simd_t<T,S> y) { x ^= y; }, 
					    [](T &x, T y) { x ^= y; },
					    -100, 100, -100, 100, 0);

    test_compound_assignment_operator<T,S> ("<<=", rng, 
					    [](simd_t<T,S> &x, simd_t<T,S> y) { x <<= y; }, 
					    [](T &x, T y) { x <<= y; },
					    0, 100, 0, 10, 0);

    test_compound_assignment_operator<T,S> (">>=", rng, 
					    [](simd_t<T,S> &x, simd_t<T,S> y) { x >>= y; }, 
					    [](T &x, T y) { x >>= y; },
					    0, 10000000, 0, 10, 0);
}


template<typename T>
inline void test_floating_point_T(std::mt19937 &rng)
{
    constexpr int S = 16 / sizeof(T);

    test_floating_point_TS<T,S> (rng);
#ifdef __AVX__
    test_floating_point_TS<T,2*S> (rng);
    test_merging_constructor<T,S> (rng);
#endif
}

template<typename T>
inline void test_integer_T(std::mt19937 &rng)
{
    constexpr int S = 16 / sizeof(T);

    test_integer_TS<T,S> (rng);
#ifdef __AVX__
    test_integer_TS<T,2*S> (rng);
    test_merging_constructor<T,S> (rng);
#endif
}

inline void test_all(std::mt19937 &rng)
{
    test_integer_T<int> (rng);
    test_integer_T<int64_t> (rng);
    test_floating_point_T<float> (rng);
    test_floating_point_T<double> (rng);

    // Upsampling is implemented for (D <= S).
    test_upsample<4,1> (rng);
    test_upsample<4,2> (rng);
    test_upsample<4,4> (rng);

    // Downsampling is implemented for general D.
    test_downsample<4,1> (rng);
    test_downsample<4,2> (rng);
    test_downsample<4,4> (rng);
    test_downsample<4,8> (rng);
    test_downsample<4,12> (rng);
    test_downsample<4,16> (rng);

#ifdef __AVX__
    test_convert<float,double,4> (rng);
    test_convert<double,float,4> (rng);
    test_upconvert<double,float,4,2> (rng);
    test_downconvert<float,double,4,2> (rng);

    // Upsampling is implemented for (D <= S).
    test_upsample<8,1> (rng);
    test_upsample<8,2> (rng);
    test_upsample<8,4> (rng);
    test_upsample<8,8> (rng);

    // Downsampling is implemented for general D.
    test_downsample<8,1> (rng);
    test_downsample<8,2> (rng);
    test_downsample<8,4> (rng);
    test_downsample<8,8> (rng);
    test_downsample<8,16> (rng);
    test_downsample<8,24> (rng);
    test_downsample<8,32> (rng);

    test_linear_algebra_kernels<float,8> (rng);
    test_linear_algebra_kernels<double,4> (rng);
#endif

#ifdef __AVX2__
    // FIXME transpose kernels currently assume AVX2
    test_transpose<float,8,4> (rng);
    test_transpose<float,8,8> (rng);
    test_btranspose<float,8,2> (rng);
#endif

    test_linear_algebra_kernels<float,4> (rng);
    test_linear_algebra_kernels<double,2> (rng);

#if defined(__AVX__) && defined(__F16C__)
    test_float16_kernels<8> (rng);
#endif
}


}   // namespace simd_helpers


int main(int argc, char **argv)
{
    string avx2 = "no";
    string f16c = "no";
    string avx = "no";
    string sse4_2 = "no";

#ifdef __AVX2__
    avx2 = "yes";
#endif

#ifdef __F16C__
    f16c = "yes";
#endif

#ifdef __AVX__
    avx = "yes";
#endif

#ifdef __SSE4_2__
    sse4_2 = "yes";
#endif

    string arch = "avx2=" + avx2 + ", f16c=" + f16c + ", avx=" + avx + ", sse4_2=" + sse4_2;
    cout << "simd_helpers: starting tests (" << arch << ")\n";

    std::random_device rd;
    std::mt19937 rng(rd());

    for (int iter = 0; iter < 1000; iter++)
	simd_helpers::test_all(rng);
    
    cout << "simd_helpers: all tests passed\n";
    return 0;
}
