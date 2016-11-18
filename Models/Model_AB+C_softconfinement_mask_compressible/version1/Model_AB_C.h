/**
 * Model_AB_C.h
 * Created at 2014.6.6
 *
 * Model_AB_C is a class derived from Model describing a
 *           linear AB diblock copolymer + C homopolymer brush
 *           confined in a Slab. 
 * The length ration of brush chain over copolymer chain is eps,
 *  namely N_AB = N and N_C = eps * N
 *
 * DBC + mask + compressible. delta function is described by (diff(\phi_W))^2,
 * where \phi_W is density function of mask. compressible model is used other that
 * incompressible model.  
 *
 * The fields are updated, the propagators is calculated,
 * and the densities are regenerated.
 *
 *
 * Copyright (C) 2016 Jun-Qing Song <13110440010@fudan.edu.cn>
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

    void update();
    double Hw() const;
    double Hs() const;
    double H() const;
    double incomp() const;
    double residual_error() const;
    double density_error() const;

    void display() const;
    void display_parameters() const;
    void save(const string file);
    void save_model(const string file);
    void save_field(const string file);
    void save_density(const string file);
    void save_q(const string file);

    ~Model_AB_C();

private:
    void init_delta();
    void init_field();
    void init_random_field();
    void init_constant_field();
    void init_file_field();
    void init_pattern_field();
    void init_density();
    void init_propagator();
    void release_memory();

private:
    bool is_compressible;
    /*C denotes brush chain while A is free chain chain. so fC was defined to be volume fraction of brush chain with fC=1-fA*/
    double C = 1.0;
    double phi; //volume fraction of all polymers
    double d; // width of mask;
    double XiN; //strength of the harmonic energy penalty for local density fluctuation in compressible model
    double eps = 1.0; //length of copolymer chain is N_AB = N, length of brush chain is N_C = eps * N
    double QAB, QC;
    //double Fc, Eab, Eacb, S_C, S_AB, S_ABconf, S_ABtrans, S_AB_pure, S_AB_incomp;
    //double Fc;
    double sigma;
    double fA, fB, fC;  // fA and fB is relative volume fraction of copolymer with fA+fB=1.0;fC is volume fraction of brush homopolymer in whole system       
    double aA, aB, aC;      // segment length
    double chiNab, chiNac, chiNbc, chiNaw, chiNbw, chiNcw;        // Flory-Huggins interation parameters
    double dsA, dsB, dsC;    // dsA=dsB and dsC is time step of free and brush chain 
    double lamA, lamB, lamC, lamYita;
    arma::uword sA, sB, sC;   //here sA+sB and sC means length of free and brush chains

    Field *wA, *wB, *wC;     // will not be initialized for Anderson mixing
    FieldAX *wAx, *wBx, *wCx;   // will not be initialized for non-Anderson mixing
    Yita *yita;         // will not be initialized for compressible model
    Density *phiA, *phiB, *phiC, *phiW; // phiW is density of mask
    //Density *phiAB_end, *phiAB_middle;
    //Density *phiC_begin, *phiC_middle, *phiC_end;
    Propagator *qA, *qB, *qC, *qAc, *qBc, *qCc;
    blitz::Array<double, 3> delta, phiw; //phiw is a matrix for phiW

    Updater *ppropupA, *ppropupB, *ppropupC;
};

#endif

