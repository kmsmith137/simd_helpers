#include <vector>
#include <iostream>
#include <unistd.h>
#include <sys/time.h>
#include "simd_helpers.hpp"
#include "simd_helpers/simd_debug.hpp"
#include "simd_helpers/downsample.hpp"

using namespace std;
using namespace simd_helpers;


// -------------------------------------------------------------------------------------------------


inline double time_diff(const struct timeval &tv1, const struct timeval &tv2)
{
    return (tv2.tv_sec - tv1.tv_sec) + 1.0e-6 * (tv2.tv_usec - tv1.tv_usec);
}

inline struct timeval get_time()
{
    struct timeval ret;
    if (gettimeofday(&ret, NULL) < 0)
	throw std::runtime_error("gettimeofday() failed");
    return ret;
}


void warm_up_cpu()
{
    // A throwaway computation which uses the CPU for ~10^9
    // clock cycles.  The details (usleep, xor) are to prevent the
    // compiler from optimizing it out!
    //
    // Empirically, this makes timing results more stable (without it,
    // the CPU seems to run slow for the first ~10^9 cycles or so.)

    long n = 0;
    for (long i = 0; i < 1000L * 1000L * 1000L; i++)
	n += (i ^ (i-1));
    usleep(n % 2);
}


// -------------------------------------------------------------------------------------------------


inline void time_trivial(float *zero, int niter)
{
    struct timeval tv0 = get_time();
    __m256 dummy = _mm256_loadu_ps(zero);
    __m256 t = _mm256_loadu_ps(zero);

    for (int iter = 0; iter < niter; iter++) {
	dummy = _mm256_xor_ps(dummy, t);
	t = _mm256_xor_ps(dummy, t);
    }

    _mm256_storeu_ps(zero, dummy);
    struct timeval tv1 = get_time();

    double dt_ns = 1.0e9 * time_diff(tv0,tv1) / (8*float(niter));
    cout << "time_trivial: " << dt_ns << " ns" << endl;
}


template<typename T, int S, int N>
void time_downsample(T *zero, int niter)
{
    struct timeval tv0 = get_time();
    
    simd_t<T,S> dummy;
    dummy.loadu(zero);
    
    simd_ntuple<T,S,N> t;
    t.loadu(zero);

    for (int iter = 0; iter < niter; iter++) {
	simd_t<T,S> u = downsample(t);
	dummy ^= u;
	t ^= u;
    }

    simd_store(zero, dummy);
    struct timeval tv1 = get_time();

    double dt_ns = 1.0e9 * time_diff(tv0,tv1) / (S*N*float(niter));
    cout << "time_downsample<" << type_name<T>() << "," << S << "," << N << ">: " << dt_ns << " ns" << endl;
}


template<typename T, int S, int N>
void time_new_downsample(T *zero, int niter)
{
    struct timeval tv0 = get_time();
    
    simd_t<T,S> dummy;
    dummy.loadu(zero);
    
    simd_ntuple<T,S,N> t;
    t.loadu(zero);

    for (int iter = 0; iter < niter; iter++) {
	simd_t<T,S> u = simd_downsample(t);
	dummy ^= u;
	t ^= u;
    }

    simd_store(zero, dummy);
    struct timeval tv1 = get_time();

    double dt_ns = 1.0e9 * time_diff(tv0,tv1) / (S*N*float(niter));
    cout << "time_new_downsample<" << type_name<T>() << "," << S << "," << N << ">: " << dt_ns << " ns" << endl;
}


int main(int argc, char **argv)
{
    warm_up_cpu();
    
    vector<float> zero(64, 0.0);
    vector<int> izero(64, 0);

    // time_downsample<float,4,1> (&zero[0], 1 << 30);
    time_downsample<float,4,2> (&zero[0], 1 << 30);
    time_downsample<float,4,4> (&zero[0], 1 << 29);
    // time_downsample<float,8,1> (&zero[0], 1 << 30);
    time_downsample<float,8,2> (&zero[0], 1 << 29);
    time_downsample<float,8,4> (&zero[0], 1 << 28);
    time_downsample<float,8,8> (&zero[0], 1 << 27);

    // time_new_downsample<float,4,1> (&zero[0], 1 << 30);
    time_new_downsample<float,4,2> (&zero[0], 1 << 30);
    time_new_downsample<float,4,4> (&zero[0], 1 << 29);
    // time_new_downsample<float,8,1> (&zero[0], 1 << 30);
    time_new_downsample<float,8,2> (&zero[0], 1 << 29);
    time_new_downsample<float,8,4> (&zero[0], 1 << 28);
    time_new_downsample<float,8,8> (&zero[0], 1 << 27);

    // time_new_downsample<int,4,1> (&izero[0], 1 << 30);
    time_new_downsample<int,4,2> (&izero[0], 1 << 30);
    time_new_downsample<int,4,4> (&izero[0], 1 << 29);
    // time_new_downsample<int,8,1> (&izero[0], 1 << 30);
    time_new_downsample<int,8,2> (&izero[0], 1 << 29);
    time_new_downsample<int,8,4> (&izero[0], 1 << 28);
    time_new_downsample<int,8,8> (&izero[0], 1 << 27);
    
    return 0;
}
