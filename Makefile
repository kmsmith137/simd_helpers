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

# Note: if more files are added here, 'make uninstall' will need to be updated
INCFILES_TOP=simd_helpers.hpp

INCFILES_SUB=simd_helpers/core.hpp \
	simd_helpers/simd_int32.hpp \
	simd_helpers/simd_int64.hpp \
	simd_helpers/simd_float16.hpp \
	simd_helpers/simd_float32.hpp \
	simd_helpers/simd_float64.hpp \
	simd_helpers/simd_ntuple.hpp \
	simd_helpers/simd_trimatrix.hpp \
	simd_helpers/simd_debug.hpp \
	simd_helpers/align.hpp \
	simd_helpers/cast.hpp \
	simd_helpers/convert.hpp \
	simd_helpers/downsample.hpp \
	simd_helpers/exp.hpp \
	simd_helpers/log.hpp \
	simd_helpers/log_add.hpp \
	simd_helpers/median.hpp \
	simd_helpers/quantize.hpp \
	simd_helpers/sort.hpp \
	simd_helpers/transpose.hpp \
	simd_helpers/upsample.hpp \
	simd_helpers/udsample.hpp \
	simd_helpers/downsample_max.hpp \
	simd_helpers/downsample_bitwise_or.hpp

TESTFILES=run-tests \
	test-convert \
	test-median \
	test-quantize \
	test-sort \
	test-special-functions \
	test-transpose \
	test-udsample

all: $(TESTFILES) time-kernels

clean:
	rm -f *~ simd_helpers/*~ .gitignore~ $(TESTFILES) .touchfile*

test: $(TESTFILES) .touchfile_test .touchfile_test_convert .touchfile_test_quantize .touchfile_test_udsample

.touchfile_test: run-tests
	./run-tests && touch $@

.touchfile_test_convert: test-convert
	./test-convert && touch $@

.touchfile_test_median: test-median
	./test-median && touch $@

.touchfile_test_quantize: test-quantize
	./test-quantize && touch $@

.touchfile_test_sort: test-sort
	./test-sort && touch $@

.touchfile_test_special_functions: test-special-functions
	./test-special-functions && touch $@

.touchfile_test_transpose: test-transpose
	./test-transpose && touch $@

.touchfile_test_udsample: test-udsample
	./test-udsample && touch $@

install:
	mkdir -p $(INCDIR)
	mkdir -p $(INCDIR)/simd_helpers
	cp -f $(INCFILES_TOP) $(INCDIR)
	cp -f $(INCFILES_SUB) $(INCDIR)/simd_helpers

uninstall:
	rm -f $(INCDIR)/simd_helpers.hpp $(INCDIR)/simd_helpers/*.hpp
	if [ -e $(INCDIR)/simd_helpers ]; then rmdir $(INCDIR)/simd_helpers; fi

run-tests: run-tests.cpp $(INCFILES_TOP) $(INCFILES_SUB)
	$(CPP) -o $@ $<

test-convert: test-convert.cpp $(INCFILES_TOP) $(INCFILES_SUB)
	$(CPP) -o $@ $<

test-median: test-median.cpp $(INCFILES_TOP) $(INCFILES_SUB)
	$(CPP) -o $@ $<

test-quantize: test-quantize.cpp $(INCFILES_TOP) $(INCFILES_SUB)
	$(CPP) -o $@ $<

test-sort: test-sort.cpp $(INCFILES_TOP) $(INCFILES_SUB)
	$(CPP) -o $@ $<

test-special-functions: test-special-functions.cpp $(INCFILES_TOP) $(INCFILES_SUB)
	$(CPP) -o $@ $<

test-transpose: test-transpose.cpp $(INCFILES_TOP) $(INCFILES_SUB)
	$(CPP) -o $@ $<

test-udsample: test-udsample.cpp $(INCFILES_TOP) $(INCFILES_SUB)
	$(CPP) -o $@ $<

time-kernels: time-kernels.cpp $(INCFILES_TOP) $(INCFILES_SUB)
	$(CPP) -o $@ $<
