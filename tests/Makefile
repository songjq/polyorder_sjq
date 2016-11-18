CXX = g++ -std=gnu++11

#MACOS = macos
#ifeq ($(MACOS), macos)
#	EXTRA_LIB_FLAGS = -framework Accelerate
#endif

LIBORDER = ../build/lib/liborder.a
LIBCHEB = ../build/lib/libcheb.a

LIB_FLAGS = $(LIBORDER) $(LIBCHEB) \
			-larmadillo -lblitz -lfftw3 -lm -lmat -lndarray \
			$(EXTRA_LIB_FLAGS)

OPT = -O2

#ARMAFINAL = -DARMA_NO_DEBUG

#WARN = -Wall
## Uncomment the above line to enable all compilation warings.

#DEBUG = -g
## Uncomment the above line to enable debug mode

#BENCHMARK = -pg
## Uncomment the above line to enable bechmark mode.

HEADERS = -I../include -I/Users/lyx/Develop/cheb++/include
CXXFLAGS = $(BENCHMARK) $(DEBUG) $(WARN) $(ARMAFINAL) $(OPT) $(HEADERS)

#all: test_armadillo test_etdrk4_1d test_etdrk4_2d \
#	 test_etdrk4_3d test_armadillo_fftw test_armadillo_blitz \
#	 test_config test_unitcell test_grid test_field test_density \
#	 test_propagator test_blitz test_fieldax test_quadrature4 \
#	 test_etdrk4_pbc_1d test_pseudospectral test_rqm4
all: test_etdrk4_2d
     #test_armadillo test_etdrk4_1d \
	 #test_armadillo_fftw test_armadillo_blitz 
	 #test_config test_unitcell test_grid test_field test_density \
	 #test_propagator test_blitz test_fieldax test_quadrature4 \
	 #test_etdrk4_pbc_1d test_pseudospectral test_rqm4

test_DiffMat: test_DiffMat.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIB_FLAGS)

test_armadillo: test_armadillo.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIB_FLAGS)

test_pseudospectral: test_pseudospectral.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIB_FLAGS)

test_etdrk4_1d: test_etdrk4_1d.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIB_FLAGS)

test_etdrk4_2d: test_etdrk4_2d.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIB_FLAGS)

test_etdrk4_3d: test_etdrk4_3d.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIB_FLAGS)

test_etdrk4_pbc_1d: test_etdrk4_pbc_1d.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIB_FLAGS)

test_rqm4: test_rqm4.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIB_FLAGS)

test_armadillo_fftw: test_armadillo_fftw.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIB_FLAGS)

test_armadillo_blitz: test_armadillo_blitz.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIB_FLAGS)

test_config: test_config.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIB_FLAGS)

test_unitcell: test_unitcell.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIB_FLAGS)

test_grid: test_grid.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIB_FLAGS)

test_field: test_field.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIB_FLAGS)

test_density: test_density.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIB_FLAGS)

test_propagator: test_propagator.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIB_FLAGS)

test_blitz: test_blitz.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIB_FLAGS)

test_fieldax: test_fieldax.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIB_FLAGS)

test_quadrature4: test_quadrature4.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIB_FLAGS)

.PHONY: clean
#clean:
#	rm -f test_armadillo test_etdrk4_1d \
#	 test_etdrk4_3d test_armadillo_fftw test_armadillo_blitz \
#	 test_config test_unitcell test_grid test_field test_density \
#	 test_propagator test_blitz test_fieldax test_quadrature4 \
#	 test_etdrk4_pbc_1d test_pseudospectral test_rqm4

clean:
	rm -f test_etdrk4_2d
	 #test_armadillo test_etdrk4_1d test_etdrk4_2d \
	 #test_armadillo_fftw test_armadillo_blitz \
	 #test_config test_unitcell test_grid test_field test_density \
	 #test_propagator test_blitz test_fieldax test_quadrature4 \
	 #test_etdrk4_pbc_1d test_pseudospectral test_rqm4
