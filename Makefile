BASEDIR = build
LOUTDIR = $(BASEDIR)/lib
LOUT = $(LOUTDIR)/liborder.a
MNANE = polyorder
MOUTDIR = $(BASEDIR)/bin
MOUT = $(MOUTDIR)/$(MNANE)
ODIR = $(BASEDIR)/src
SDIR = src
MODIR = $(BASEDIR)/main
MSDIR = scft

CXX = g++ -std=gnu++11

MACOS = macos
#ifeq ($(MACOS), macos)
#	EXTRA_LIB_FLAGS = -framework Accelerate
#endif

LIBCHEB = ~/Develop/cheb++/build/lib/libcheb.a
LIB_FLAGS = -larmadillo $(LIBCHEB) $(EXTRA_LIB_FLAGS)

OPT = -O2

#ARMAFINAL = -DARMA_NO_DEBUG

#WARN = -Wall
## Uncomment the above line to enable all compilation warings.

#DEBUG = -g
## Uncomment the above line to enable debug mode

#BENCHMARK = -pg
## Uncomment the above line to enable bechmark mode.

LIB_HEADERS = -Iinclude -I/Users/songjq/Develop/cheb++/include

CXXFLAGS = $(BENCHMARK) $(DEBUG) $(WARN) $(ARMAFINAL) $(OPT) $(LIB_HEADERS)

# here, $(LOUT) should before any others to link successfully
MLIBS =  $(LOUT) $(LIB_FLAGS) -lm -lblitz -lndarray -lfftw3 -lmx -lmex -lmat

_OBJS = common.o \
		Config.o \
		Grid.o \
		UnitCell.o \
		Field.o \
		FieldAX.o \
		Yita.o \
		Propagator.o \
		Density.o \
		DensitySmall.o \
		Simpson.o \
		Quadrature4.o \
	   	PseudoSpectral.o \
	   	RQM4.o \
	   	Etdrk4_PBC.o \
	   	Etdrk4.o \
		Model_ABW.o \
		Model_AS.o \
		Model_AB_S.o \
		Model_A_B.o \
		Model_AB.o \
		Model_AB_C.o \
		Helper.o \
	   	scft.o

OBJS = $(patsubst %,$(ODIR)/%,$(_OBJS))

_MOBJS = $(MNANE).o

MOBJS = $(patsubst %,$(MODIR)/%,$(_MOBJS))

$(ODIR)/%.o : $(SDIR)/%.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(MODIR)/%.o : $(MSDIR)/%.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(MOUT) : $(MOBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(MLIBS)

$(LOUT) : $(OBJS)
	ar -r $(LOUT) $^

.PHONY: all
all: $(LOUT)

.PHONY: lib
lib: $(LOUT)

.PHONY: main
main: $(MOUT)

.PHONY: clean
clean:
	rm -f $(ODIR)/*.o $(TODIR)/*.o $(MODIR)/*.o $(MOUT) $(LOUT)
