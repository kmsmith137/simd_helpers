### simd_helpers

This is a helper library for writing fast x86 SIMD kernels.

It is currently in an "initial hacking" stage, and will be hard to use until it gets some cleanup and documentation!
Let me know if this should be a higher priority.

Compilation instructions:
```
ln -s site/Makefile.local.noprootprivs Makefile.local   # see comment in Makefile for discussion
make
./test-linear-algebra-kernels    # currently the only unit test
make install
```
