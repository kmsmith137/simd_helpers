### simd_helpers

This is a helper library for writing fast x86 SIMD kernels.

It is currently in an "initial hacking" stage, and will be hard to use until it gets some cleanup and documentation!
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

# Currently unit tests must be run individually by hand
./test-basics
./test-linear-algebra-kernels
./test-udsample

# Installs .hpp files to $(INCDIR)
make install
```
