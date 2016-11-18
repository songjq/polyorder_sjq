/**
 * Etdrk4_PBC.h
 * Created at 2014.4.22 by Jun-Qing Song.
 * Refractored at 2014.9.15 by Yi-Xin Liu.
 *
 * Etdrk4 is derived from Updater implementing
 * ETDRK4 algorithms for solving propagation equations
 * (modified diffusion function) with periodic boundary conditions.
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

#ifndef Polyorder_Etdrk4_PBC_h
#define Polyorder_Etdrk4_PBC_h

#include <blitz/array.h>
#include <cmath>

#include "fftw3.h"
#include "Updater.h"
#include "Grid.h"
#include "Propagator.h"
#include "UnitCell.h"

using namespace blitz;

class Etdrk4_PBC:public Updater{
public:
	Etdrk4_PBC() {};
	Etdrk4_PBC(const Etdrk4_PBC &rhs);
    Etdrk4_PBC(const UnitCell &uc,
               const int Lx, const int Ly, const int Lz,
               const double ds,
               const int M=32);
	void solve(Propagator &, const Grid &);
    Etdrk4_PBC *clone() const;
	~Etdrk4_PBC();

private:
	fftw_complex *_fftw_in;
	fftw_complex *_fftw_out;
	fftw_plan _p_forward;
	fftw_plan _p_backward;
	Array<double,3> _Etdrk4_L;

    int _M;  // number of circular contour points
	Array<double,3> _E;
	Array<double,3> _E2;
	Array<double,3> _Q;
	Array<double,3> _f1;
	Array<double,3> _f2;
	Array<double,3> _f3;

	void init_fftw();
	void init_coefficient(const double);
};

#endif

