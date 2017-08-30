#include <vector>
#include <iostream>
#include <unistd.h>
#include <sys/time.h>
#include "simd_helpers.hpp"
#include "simd_helpers/simd_debug.hpp"
#include "simd_helpers/downsample.hpp"
#include "simd_helpers/upsample.hpp"

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


template<typename T, int S, int N>
void time_old_downsample(T *zero, int niter)
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
    cout << "time_old_downsample<" << type_name<T>() << "," << S << "," << N << ">: " << dt_ns << " ns" << endl;
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


// -------------------------------------------------------------------------------------------------


    
template<typename T, int S, int N>
void time_old_upsample(T *zero, int niter)
{
    struct timeval tv0 = get_time();
    
    simd_t<T,S> dummy;
    dummy.loadu(zero);

    simd_ntuple<T,S,N> t;

    for (int iter = 0; iter < niter; iter++) {
	upsample(t, dummy);
	dummy ^= t.vertical_xor();
    }
    
    simd_store(zero, dummy);
    struct timeval tv1 = get_time();

    double dt_ns = 1.0e9 * time_diff(tv0,tv1) / (S*N*float(niter));
    cout << "time_old_upsample<" << type_name<T>() << "," << S << "," << N << ">: " << dt_ns << " ns" << endl;
}


template<typename T, int S, int N>
void time_new_upsample(T *zero, int niter)
{
    struct timeval tv0 = get_time();
    
    simd_t<T,S> dummy;
    dummy.loadu(zero);

    simd_ntuple<T,S,N> t;

    for (int iter = 0; iter < niter; iter++) {
	simd_upsample(t, dummy);
	dummy ^= t.vertical_xor();
    }
    
    simd_store(zero, dummy);
    struct timeval tv1 = get_time();

    double dt_ns = 1.0e9 * time_diff(tv0,tv1) / (S*N*float(niter));
    cout << "time_new_upsample<" << type_name<T>() << "," << S << "," << N << ">: " << dt_ns << " ns" << endl;
}


// -------------------------------------------------------------------------------------------------


void time_downsample()
{
    vector<float> zero(64, 0.0);
    vector<int> izero(64, 0);

    time_old_downsample<float,4,2> (&zero[0], 1 << 30);
    time_old_downsample<float,4,4> (&zero[0], 1 << 29);
    time_old_downsample<float,8,2> (&zero[0], 1 << 29);
    time_old_downsample<float,8,4> (&zero[0], 1 << 28);
    time_old_downsample<float,8,8> (&zero[0], 1 << 27);

    time_new_downsample<float,4,2> (&zero[0], 1 << 30);
    time_new_downsample<float,4,4> (&zero[0], 1 << 29);
    time_new_downsample<float,8,2> (&zero[0], 1 << 29);
    time_new_downsample<float,8,4> (&zero[0], 1 << 28);
    time_new_downsample<float,8,8> (&zero[0], 1 << 27);

    time_new_downsample<int,4,2> (&izero[0], 1 << 30);
    time_new_downsample<int,4,4> (&izero[0], 1 << 29);
    time_new_downsample<int,8,2> (&izero[0], 1 << 29);
    time_new_downsample<int,8,4> (&izero[0], 1 << 28);
    time_new_downsample<int,8,8> (&izero[0], 1 << 27);
}


template<int S, int N>
void time_upsample(int niter)
{
    vector<float> zero(64, 0.0);
    vector<int> izero(64, 0.0);
    
    time_old_upsample<int,S,N> (&izero[0], niter);
    time_new_upsample<int,S,N> (&izero[0], niter);
    
    time_old_upsample<float,S,N> (&zero[0], niter);
    time_new_upsample<float,S,N> (&zero[0], niter);
}

    
void time_upsample()
{
    time_upsample<4,2> (1 << 30);
    time_upsample<4,4> (1 << 29);
    time_upsample<8,2> (1 << 29);
    time_upsample<8,4> (1 << 28);
    time_upsample<8,8> (1 << 27);
}


int main(int argc, char **argv)
{
    warm_up_cpu();
    
    // time_downsample();
    time_upsample();
    
    return 0;
}
