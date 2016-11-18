/**
 * Model_AB.h
 * Created at 2014.6.6
 *
 * Model_A_B is a class derived from Model describing a
 *           linear A homopolymer + B homopolymer brush
 * confined in a Slab.
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

#ifndef polyorder_model_a_b_h
#define polyorder_model_a_b_h

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

class Model_A_B:public Model{
public:
    Model_A_B(){}
    Model_A_B(const string config_file);

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

    ~Model_A_B();

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
    double QA, QC;
    double E, S_C, S_A, S_Atrans, S_Aconf;
    double sigma;
    double fA, fC;       
    double aA, aC;      // segment length
    double chiN;        // Flory-Huggins interation parameters
    double dsA, dsC;    // dsA and dsB is time step of brush and free chain 
    double lamA, lamC, lamYita;
    arma::uword sA, sC;   //here sA and sB means length of brush and free chains

    Field *wA, *wC;     // will not be initialized for Anderson mixing
    FieldAX *wAx, *wCx;   // will not be initialized for non-Anderson mixing
    Yita *yita;         // will not be initialized for compressible model
    Density *phiA, *phiC;
    Density *phiA_end, *phiA_middle;
    Density *phiC_begin, *phiC_end, *phiC_middle;
    Propagator *qA, *qC, *qAc, *qCc;
    blitz::Array<double, 3> delta;

    Updater *ppropupA, *ppropupC;
};

#endif

