# Makefile.local must define the following variables
#   INCDIR   install dir for C++ headers
#   CPP      c++ compiler command line (should probably include -std=c++11 -O3 -march=native)
#
# You can probably just symlink Makefile.local to one of the following:
#   site/Makefile.local.norootprivs     (specifies INCDIR=$HOME/include)
#   site/Makefile.local.rootprivs       (specifies INCDIR=/usr/local/include)

include Makefile.local

ifndef CPP
$(error Fatal: Makefile.local must define CPP variable)
endif

ifndef INCDIR
$(error Fatal: Makefile.local must define INCDIR variable)
endif


INCFILES_TOP=simd_helpers.hpp

INCFILES_SUB=simd_helpers/simd_t.hpp \
	simd_helpers/simd_t_implementations.hpp \
	simd_helpers/simd_ntuple.hpp \
	simd_helpers/simd_trimatrix.hpp \
	simd_helpers/simd_debug.hpp \
	simd_helpers/udsample.hpp

TESTFILES=test-basics \
	test-linear-algebra-kernels \
	test-udsample


all: $(TESTFILES)

# FIXME needs 'make test'

clean:
	rm -f *~ simd_helpers/*~ .gitignore~ $(TESTFILES)

install:
	mkdir -p $(INCDIR)
	mkdir -p $(INCDIR)/simd_helpers
	cp -f $(INCFILES_TOP) $(INCDIR)
	cp -f $(INCFILES_SUB) $(INCDIR)/simd_helpers

uninstall:
	for f in $(INCFILES_TOP) $(INCFILES_SUB); do rm -f $(INCDIR)/$$f; done
	rmdir $(INCDIR)/simd_helpers

test-basics: test-basics.cpp $(INCFILES_TOP) $(INCFILES_SUB)
	$(CPP) -o $@ $<

test-linear-algebra-kernels: test-linear-algebra-kernels.cpp $(INCFILES_TOP) $(INCFILES_SUB)
	$(CPP) -o $@ $<

test-udsample: test-udsample.cpp $(INCFILES_TOP) $(INCFILES_SUB)
	$(CPP) -o $@ $<
