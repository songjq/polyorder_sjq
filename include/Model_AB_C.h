/**
 * Model_AB_C.h
 * Created at 2014.6.6
 *
 * Model_AB_C is a class derived from Model describing a
 *           linear AB diblock copolymer + C homopolymer brush
 *           confined in a Slab.
 * The fields are updated, the propagators is calculated,
 * and the densities are regenerated.
 *
 *
 * Copyright (C) 2014 Yi-Xin Liu <lyx@fudan.edu.cn>
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

#ifndef polyorder_model_ab_c_h
#define polyorder_model_ab_c_h

#include <string>
#include <iostream>

#include "armadillo"

#include "Model.h"
#include "Config.h"
#include "Propagator.h"
#include "fields.h"
#include "densities.h"
#include "updaters.h"
#include "Cheb.h"

using std::string;

class Model_AB_C:public Model{
public:
    Model_AB_C(){}
    Model_AB_C(const string config_file);

    void init();
    void reset(const string& config_data);
    //void resetInStringABC(int s, int Lx, int Ly, int Lz, blitz::Array<double, 3> &w1, blitz::Array<double, 3> &w2, blitz::Array<double, 3> &w3);

    void update();
    double Hw() const;
    double Hs() const;
    double H() const;
    double incomp() const;
    double residual_error() const;
    double density_error() const;
    //blitz::Array<double, 4> output_data(); //added by songjq for string method in 20161008

    void display() const;
    void display_parameters() const;
    void save(const string file);
    void save_model(const string file);
    void save_field(const string file);
    void save_density(const string file);
    void save_q(const string file);
    //void input_AField(blitz::Array<double, 3>); //added by songjq for string method in 20161008
    //void input_BField(blitz::Array<double, 3>); //added by songjq for string method in 20161008
    //void input_CField(blitz::Array<double, 3>); //added by songjq for string method in 20161008
    //void release_memory_string();  //to be compatible with Model_AB

    ~Model_AB_C();

private:
    void init_delta();
    void init_field();
    void init_random_field();
    void init_constant_field();
    void init_file_field();
    void init_pattern_field();
    //void init_data_field(); //added by songjq for string method in 20161008
    void init_density();
    void init_propagator();
    void release_memory();

private:
    bool is_compressible;
    /*C denotes brush chain while A is free chain chain. so fC was defined to be volume fraction of brush chain with fC=1-fA*/
    double C = 1.0;
    double eps = 1.0; //length of brush chain
    double QAB, QC;
    double sigma;
    double fA, fB, fC;  // fA and fB is relative volume fraction of copolymer with fA+fB=1.0;fC is volume fraction of brush homopolymer in whole system       
    double aA, aB, aC;      // segment length
    double chiNab, chiNac, chiNbc;        // Flory-Huggins interation parameters
    double dsA, dsB, dsC;    // dsA=dsB and dsC is time step of free and brush chain 
    double lamA, lamB, lamC, lamYita;
    double LamA, LamB, LamC; // coefficients for short-range potential
    arma::uword sA, sB, sC;   //here sA+sB and sC means length of free and brush chains
    //blitz::Array<double, 3> waa, wbb, wcc; //added by songjq for string method in 20161008

    Field *wA, *wB, *wC, *wH;     // will not be initialized for Anderson mixing, *wH timed multiplied by a different factors is short-range potential for A, B, C
    FieldAX *wAx, *wBx, *wCx;   // will not be initialized for non-Anderson mixing
    Yita *yita;         // will not be initialized for compressible model
    Density *phiA, *phiB, *phiC;
    Density *phiAB_end, *phiAB_middle;
    Density *phiC_begin, *phiC_middle, *phiC_end;
    Propagator *qA, *qB, *qC, *qAc, *qBc, *qCc;
    blitz::Array<double, 3> delta;

    Updater *ppropupA, *ppropupB, *ppropupC;
};

#endif

