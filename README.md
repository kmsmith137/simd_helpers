### simd_helpers

This is a helper library for writing fast x86 SIMD kernels.

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

make test      # optional but recommended
make install   # Installs .hpp files to $(INCDIR)
```

You may find that compilation fails, with the error message below, followed by a very long
list of compiler errors.
```
Either you're compiling on an old machine, or you forgot the -march=native compiler flag.  
If this is an old machine, see note in simd_helpers/README.md.
```
This error message is generated if your CPU appears to not have the SSE 4.2 instruction set.
If you're using a machine more recent than 2008, then you probably just need to include `-march=native`
in your compiler flags.

If you are getting this error because you have an old (pre-2008) machine, then send me an email
and it will probably be easy for me to get simd_helpers working on your machine.  The only reason
I haven't done it yet is that I don't have access to a machine old enough to use for testing!






