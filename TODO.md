- Needs a lot more documentation!  (some example .cpp files would help a lot)

- Aligned/streaming load/store flags will be implemented soon!
  It might be useful to move my memory bandwidth profiling code to this github repo.

- Fused multiply/adds would be good to implement soon.
  (I'd like to play with these and see if they can improve some of my existing kernels!)

- Half-float loads/stores.  (Needed soon for bonsai)

- Random loose end: scalar-vector ops, e.g. (T * simd_t<T,S>), are not currently unit tested.

- I think more syntactic sugar would be nice.
  Random example: min(x,y) can be a synonym for x.min(y)

- Upsampling/downsampling kernels for int64_t and double.

- Does it make sense to implement something for integer division?
  There are no "real" simd instructions for integer division (at least in AVX2).  
  Maybe the best option is to extract every element of the simd_t, and do a scalar integer division?
  There is an assembly instruction for scalar integer division, right?

- Could write a 'make testx' target which runs the unit tests with multiple combinations of cpu flags
  (downgraded from -march=native), e.g. to test non-AVX2 kernels on an AVX machine.

- Generally speaking, the non-AVX2 kernels are not very well-optimized, but I'm not sure how much of a priority this is.

- Lots more integer types are possible (int8, uint8, int16, uint16, uint32, uint64)

- In spite of the number of lines of boilerplate here, there is a lot missing when compared to the intel manuals!
