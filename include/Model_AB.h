/**
 * Model_AB.h
 * Created at 2014.6.6
 *
 * Model_AB is a class derived from Model describing a
 *           linear A-B diblock copolymer melt
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

#ifndef polyorder_model_ab_h
#define polyorder_model_ab_h

#include <string>
#include <iostream>

#include "armadillo"

#include "Model.h"
#include "Config.h"
 #include "Propagator.h"
#include "fields.h"
#include "densities.h"
#include "updaters.h"

using std::string;

class Model_AB:public Model{
public:
    Model_AB(){}
	Model_AB(const string config_file);

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

    //double Eab, Eh, Sab; //AB interaction energy, surface energy associated with dots, enthopy

    ~Model_AB();

private:
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
    bool is_induced; //induced by chemical pattern or dot
    string confine_mold; //judge NBC_by_PBC or not
    double fA, fB;      // fA=1-fB=NA/N
    double aA, aB;      // segment length
    double chiN;        // Flory-Huggins interation parameters
    double dsA, dsB;
    double lamA, lamB, lamYita;
    double lamH;        //strength of the pattern
    arma::uword sA, sB;
    double fA_shiftvalue;  //added by songjq in 20141103

    Field *wA, *wB;     // will not be initialized for Anderson mixing
    FieldAX *wAx, *wBx;   // will not be initialized for non-Anderson mixing
    Yita *yita;         // will not be initialized for compressible model
    Density *phiA, *phiB;
    Propagator *qA, *qB, *qAc, *qBc;
    Updater *ppropupA, *ppropupB;
    Field *wH;      //chemical pattern or dot on up&bottom substrates
};

#endif

