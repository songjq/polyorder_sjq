/**
 * common.h
 * Created at 2011.6.6
 *
 * It is a file containing constants, structs used across Polyorder.
 *
 * History
 * 2012.4.2
 *   1. From now on, the history of this file is tracked by Mercurial.
 *   2. Package name: polyorder.
 * 2011.6.6
 *   1. original version
 *
 * Copyright (C) 2012 Yi-Xin Liu <lyx@fudan.edu.cn>
 *
 * This file is part of Polyorder
 *
 * Polyorder is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License
 * as published by the Free Software Foundation, either version 3
 * of the License, or (at your option) any later version.
 *
 * Polyorder is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Polyorder.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef polyorder_common_h
#define polyorder_common_h

#include <string>
using std::string;

// Current MUDPACK 5.0.1 is single precision
// Use this macro to define the precision,
// in case the MUDPACK changes its precision, or
// we modify MUDPACK's precision.
// MUD_INT is defined to compromise with f2c converted MUDPACK lib
#define MUD_FLOAT float
#define MUD_INT long int

// http://en.wikipedia.org/wiki/Pi.
const double PI=3.14159265358979323846264338327950288;
const double SMALL=1e-10;
const double LARGE=1e99;

const string DEFAULT_DATA_FILE="data";
const string DEFAULT_Q_FILE="q";
const string DEFAULT_PARAM_FILE="param";

/*
 * We intend to explicitly list all models instead of just saying that
 * this is a linear block copolymer model.
 * We believe to build a general linear block copolymer model is much
 * more time consuming than to build each specific linear block copolymer
 * model separately.
 * All availabe models are included in models.h. Therefore one can obtain
 * these models by just "#include models.h".
 */
enum class MaskType {
    COS,
    EXP,
    TANH,
};
enum class EnsembleType {
    canonical,
    grandCanonical,
};
enum class ModelType {
    AB,         // AB diblock copolymers
    ABW,        // AB diblock copolymers + wall particles
    //A_gC,        // A homopolymer + grafted homopolymer C on two parallel flats
    A_B,        // A + B polymer blends
    AS,         // A polymer + implicit solvent
    A_S,        // A polymer + explicit solvent
    AB_S,       // AB diblock copolymers + explicit solvent
    g_AB_S,     // AB diblock copolymers + explicit solvent in grand canonical ensemble
    AB_C,       // AB + C polymer blends, C homopolymer is grafted brush
    ABC,        // ABC triblock copolymer
    ABm,        // A-Bm miktoarm polymers, m is the number of B blocks.
    BmABm,      // Bm-A-Bm Ladder like block copolymers
    BABm,       // B-A-Bm mixing diblock and miktoarm copolymers
    STAR        // One common point 0 ~ m arms star polymers
};

enum class ETDRK4SCHEME {
    COX,
    KROGSTAD
};

/* Algorithms for solving Modified diffusion equations */
enum class AlgorithmMDEType {
    OS2,    // PseudoSpectral
    RQM4,   // RQM4
    ETDRK4  // ETDRK4 or ETDRK4_PBC
};

/* Algorithms for updating SCFT equations */
enum class AlgorithmSCFTType {
    EM,         // Explicit Euler scheme
    ANDERSON,   // Anderson mixing
    SIS,        // Semi-implicit Seidel scheme
    ETD         // Exponential time difference
};

/* Algorithms for searching for optimum cell size */
enum class AlgorithmCellType {
    SINGLE,     // a single cell size calculation, i.e. no optimization.
    MANUAL,     // manually batch cell size
    BRENT,      // optimization by Brent's method
    VARIABLE    // optimization by variable cell method
};

/* Algorithms for conducting contour integration */
enum class AlgorithmContourType {
    TRAPEZOIDAL,  // Trapezoidal rule
    SIMPSON       // Simpson's rule, fall back to other 4th order quadratures.
};

/**
 * A list of confinement types.
 * NONE: not confined.
 * CUBE: including all kinds of parallelepiped (1D, 2D, 3D)
 *       1D CUBE: confined in a line segment.
 *       2D CUBE: confined in a slit
 *       3D CUBE: confined in a slab
 * DISK: confined in disk (2D) by circumference.
 * CYLINDER: confined in cylinder (3D) by cylindrical surface.
 * SPHERE: confined in sphere (3D) by spherical surface.
*/
enum class ConfineType {
    NONE,
    CUBE,
    DISK,
    CYLINDER,
    SPHERE
};

enum class GridType{
    REGULAR,  /* equal-spaced grid */
    CHEBYSHEV_GAUSS_LOBATTO  /* Chebyshev-Gauss-Lobatto grid */
};

enum class CrystalSystemType{
                    // case-insensitive, accepted input words
    LAMELLAR,       // 1D: LAM, lamellar, lamella, lamellae
    SQUARE,         // 2D: square, sq
    RECTANGULAR,    // 2D: rect, rectangle, rectangular
    HEXAGONAL,      // 2D/3D: HEX, hexagonal, hexagon
    OBLIQUE,        // 2D: ob, oblique
    CUBIC,          // 3D: cube, cubic
    TETRAGONAL,     // 3D: tetragonal, tetragon
    ORTHORHOMBIC,   // 3D: orthorhombic
    TRIGONAL,       // 3D: trigonal
    MONOCLINIC,     // 3D: monoclinic
    TRICLINIC,      // 3D: triclinic
    UNKNOWN         // Wrong input
};

enum class GridInitType{
                    // case-insensitive, accepted input words
    RANDOM_INIT,    // ran, random
    CONSTANT_INIT,  // const, constant
    DATA_INIT,      // data
    FILE_INIT,      // file
    PATTERN_INIT    // pattern
};

struct CellParam{
    double a; // length of unit vector a in real space
    double b; // length of unit vector b in real space
    double c; // length of unit vector c in real space
    double alpha; // angle between b and c
    double beta; // angle between c and a
    double gamma; // angle between a and b
};

////////////////////////////////////////////////////////////////////////
// Following are meant to change in the future

enum PhasePattern{
                    // only 2 component systems are supported
    LAM1_PATTERN,   // 2D or 3D, 1 period, type I, along y axis
    LAM2_PATTERN,   // 1D, 2D or 3D, 1 period, type II, along z axis
    LAM3_PATTERN,   // 2D or 3D, 1 period, type III, along x axis
    LAM4_PATTERN,   // 2D or 3D, 2 periods, type I, along y axis
    LAM5_PATTERN,   // 2D or 3D, 2 periods, type II, along z axis
    LAM6_PATTERN,   // 2D or 3D, 2 periods, type III, along x axis
    HEX1_PATTERN,   // 2D, 1 period, type I , one circle in center
    HEX2_PATTERN,   // 2D, 1 period, type II, 4 half circles on 4 edges
    HEX3_PATTERN,   // 2D, 2 periods, type I, two circles in center
    HEX4_PATTERN,   // 2D, 2 periods, type II, one circle in center
    GYROID_PATTERN, // 3D
    HCP_PATTERN,    // 3D
    NUM_PHASE_PATTERN
};
const string PhasePatterns[NUM_PHASE_PATTERN]={"LAM1","LAM2","LAM3","LAM4","LAM5","LAM6","HEX1","HEX2","HEX3","HEX4","GYROID","HCP"};

struct MudpackSig{
    int dim;
    int nx;
    int ny;
    int nz;
    MUD_FLOAT dx;
    MUD_FLOAT dy;
    MUD_FLOAT dz;
    MUD_FLOAT xa;
    MUD_FLOAT yc;
    MUD_FLOAT ze;
    MUD_FLOAT xb;
    MUD_FLOAT yd;
    MUD_FLOAT zf;
    MUD_FLOAT beta; // for Monoclinic unit cell mud3cr
    MUD_FLOAT *xdata; // for sigx_mud2, sigx_mud3, cof_cr_mud2, cof_cr_mud3
    MUD_FLOAT *ydata; // for sigy_mud2, sigy_mud3, cof_cr_mud2, cof_cr_mud3
    MUD_FLOAT *zdata; // for sigz_mud3, cof_cr_mud3
    MUD_FLOAT *lambda_data; // for lambda_mud2, lambda_mud3, cof_cr_mud3
};
extern MudpackSig gmudsig; // a better name should be gmudsa

struct SCFTParam{
    string base_dir; // the base path for storing files
    string data_file; // to store all information except q_file
    string param_file; // store model parameters
    int min_iter;   // Minimum number of iterations
	int max_iter;	// Maximum number of iterations
    bool is_display;
    bool is_save_data;
    bool is_save_q;
	int display_interval;	// every interval_print relaxation steps print to screen
    int record_interval;     // the interval for calculating energy (_H), residual error (_residual_error), and imcompressibility (_icomp)
	int save_interval;	// save data to file every interval_save relaxation steps
	double thresh_H;	// terminating signal, when difference of two H smaller than threshH
//	double thresh_density;	// terminating signal, when difference of two density small than threshDensity
	double thresh_residual;	// terminating signal, when residual error is small than threshResidual
	double thresh_incomp;	// terminating signal, when incompressibility is small than thresh_incomp
};

struct BatchParam{
	string name;	// the variable to be swept in batch
	double value;	// the current value of the batch variable
	double min;		// the minimum value of batch variable
	double step;	// the step for increasing batch variable
	double max;	    // the maximum value of batch variable
				    // i.e. batchVar's range is batchVar:batchVarStep:batchVarMax
};

enum MudpackType{
    MUD_SP, // separable
    MUD_SP4, // 4th order discretization
    MUD_SA, // self-adjoint
    MUD_NSP, // non-separable
    MUD_NSP4, // 4th order
    MUD_H, // hybrid
    MUD_H4, // 4th order
    MUD_CR, // cross term
    MUD_CR4, // 4th order
    MUD_HCR, // H + CR
    MUD_HCR4,// 4th order
    NUM_MUDPACK_TYPE
};

const string MudpackTypes[NUM_MUDPACK_TYPE]={"sp","sp4","sa","nsp","nsp4","h","h4","cr","cr4","hcr","hcr4"};

#endif

