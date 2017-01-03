### simd_helpers

This is a helper library for writing fast x86 SIMD kernels.

This is version 2.  This version is required for the `fast_clippers` and `fast_transforms`
branches of rf_pipelines.  For the `fast_detrender` branch, you want version 1.

This library is currently in an "initial hacking" stage, and will be hard to use until it gets more documentation!
Let me know if this should be a higher priority.

Compilation instructions:
```
# First you need to create a file Makefile.local, which defines Makefile variables
#   INCDIR   install dir for C++ headers
#   CPP      c++ compiler command line (should probably include flags -std=c++11 -O3 -march=native)
#
# You can probably just symlink Makefile.local to one of the following:
#   site/Makefile.local.norootprivs     (specifies INCDIR=$HOME/include)
#   site/Makefile.local.rootprivs       (specifies INCDIR=/usr/local/include)

# Assuming you're not installing as root
ln -s site/Makefile.local.norootprivs Makefile.local

# Compiles everything
make

# Currently unit tests must be run by hand
./run-tests

# Installs .hpp files to $(INCDIR)
make install
```
