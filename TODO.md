- Needs a lot more documentation!  (some example .cpp files would help a lot)

- Aligned/streaming load/store flags will be implemented soon!
  It might be useful to move my memory bandwidth profiling code to this github repo.

- Fused multiply/adds would be good to implement soon.
  (I'd like to play with these and see if they can improve some of my existing kernels!)

- Half-float loads/stores.  (Needed soon for bonsai)

- Align operations.  (Needed soon for bonsai)

- Horizontal reducing min/max.  (Needed soon for bonsai)

- Random loose end: scalar-vector ops, e.g. (T * simd_t<T,S>), are not currently unit tested.

- I think more syntactic sugar would be nice.
  Random example: min(x,y) can be a synonym for x.min(y)

- Not all upsampling/downsampling kernels have been implemented.  So far we only have
     - upsampling: float32, int32
     - downsampling: float32

- For an integer type T, simd_t<T,S>::operator*() wraps the simplest possible multiplication
  intrinsic, but there are other possibilities.  (E.g. _mm_mul_epi32() or _mm_mul_epu32()
  in addition to _mm_mullo_epi32() which corresponds to operator*())  These should be
  wrapped somehow as well.

- Does it make sense to implement something for integer division?
  There are no "real" simd instructions for integer division (at least in AVX2).  Maybe the 
  best option is to extract every element of the simd_t, and do a scalar integer division?
  (There is an assembly instruction for scalar integer division, right?)

- Comparison operators could have boolean template arguments to override the "quiet ordered" default.

- Could write a 'make testx' target which runs the unit tests with multiple combinations of cpu flags
  (downgraded from -march=native), e.g. to test non-AVX2 kernels on an AVX2 machine.

- Generally speaking, the non-AVX2 kernels are not very well-optimized, but I'm not sure how much of a priority this is.
  When documentation exists, it should say somewhere that 256-bit integer types are dubious without AVX2, and may be slower than 128-bit types.

- Lots more integer types are possible (int8, uint8, int16, uint16, uint32, uint64)

- In spite of the number of lines of boilerplate here, there is a lot missing when compared to the intel manuals!

- I currently assume the AVX instruction set.  This restriction could be removed, but not sure if it's worth the effort!

