/**
 * scft.h
 * Created at 2011.6.6
 *
 * scft is the base class for performing SCFT simulation of
 * polymer systems. It sets the general framework to perform
 * SCFT simulation. scft class is model independent.
 * That means we can perform scft using this class for
 * any models given that the specific model class is provided.
 *
 * HISTORY:
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

#ifndef polyorder_scft_h
#define polyorder_scft_h

#include <cstdlib>
#include <ctime>
#include <cmath>
#include <complex>
#include <iostream>

#include <blitz/array.h>

#include "CMatFile.h" // Matlab MAT-file processor
#include "common.h"
#include "Config.h"
#include "Helper.h"
#include "Model.h"

using std::string;

class scft{
public:
	scft(const string config_file, Model* pmodel);
    void run();

private:
    Config _cfg;
    //SCFTParam _scftp;
    //BatchParam _batchp;
    Model* _pmodel;
    arma::uword _num_scft;  // number of scft simulations has been run.

    // Number of steps that actually has been run, starting from 1.
	int _iter;
    // max number of recorded iters
    int _num_iters;
    blitz::Array<double, 1> _t;  // array of _iter
    blitz::Array<double, 1> _time;  // array of CPU time for each SCFT cycle
    blitz::Array<double, 1> _residual_error;  // array of errors
    blitz::Array<double, 1> _H;  // array of energy
    blitz::Array<double, 1> _incomp;  // array of incompressibility

    double _curr_a, _curr_b, _curr_c;
    // For lattice size optimization algorithm: Brent's method
    double aa, ba, ca, Faa, Fba, Fca;
    double amin, bmin, cmin, Fmin;
    arma::uword max_iter_cell;
    arma::vec as, bs, cs, Fs;
    string prev_data_file;

	void init();
    // calculating H, residual error, incomp
    void init_record();
    void reset_model(double a, double b, double c);
	void relax();
    double solve_scft(double a, double b, double c);  // F is returned
    // save model and scft parameters
    void save_param();
    // save scft parameters
	void save_scft(const string file);
    // save calculated values: residual error, H, and incomp
    void save_record(const string file);
	bool save_data(const string s, const int i_save);
    // save final cell optimization result
    void save_cell(const string file);
	void display(const double t) const;

    /** Lattice size optimization algorithm: Brent's method **/
    // Find aa, ba, and ca where F(aa) > F(ba) and F(ca) > F(ba)
    void bracket();
    void optimization_Brent();
    void update_cell(const double a, double& b, double& c);
};

#endif

