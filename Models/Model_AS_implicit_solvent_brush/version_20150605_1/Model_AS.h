/**
 * Model_AS.h
 * Created at 2014.6.6
 *
 * Model_AS is a class derived from Model describing a
 *           homopolymer in implicit solvent
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

#ifndef polyorder_model_as_h
#define polyorder_model_as_h

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

class Model_AS:public Model{
public:
    Model_AS(){}
	Model_AS(const string config_file);

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

    ~Model_AS();

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
    double C = 1.0;      //constant used to obtain dimensionless variables
    double v;            //excluded_volume
    double aA;           //segment length
    double alpha = 0.001;
    double graft_density;
    arma::uword sA;       
    double dsA;
    double lamA;
    Field *wA;
    Density *phiA;
    Propagator *qA, *qAc;
    blitz::Array<double, 3> delta;

    Updater *ppropupA;
    Cheb *cheb;
};


#endif

