### simd_helpers

This is a helper library for writing fast x86 SIMD kernels.

It is currently in an "initial hacking" stage, and will be hard to use until it gets some cleanup and documentation!
Let me know if this should be a higher priority.

Compilation instructions:
```
ln -s site/Makefile.local.norootprivs Makefile.local   # see comment in Makefile for discussion
make
./test-linear-algebra-kernels    # currently unit tests must be run individually by hand
./test-udsample
make install
```
