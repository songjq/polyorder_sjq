/**
 * PseudoSpectral.h
 * Created at 2011.6.20
 *
 * PseudoSpectral is derived from Updater implementing
 * pseudospectral algorithm for solving propagation equations
 * (modified diffusion function).
 *
 * HISTORY:
 * 2012.3.31
 *   1. From now on, the history of this file is tracked by Mercurial.
 *   2. Package name: polyorder.
 * 2011.6.21
 *   1. test Passed.
 * 2011.6.20
 *   1. original version
 *
 * Copyright (C) 2012-2014 Yi-Xin Liu <lyx@fudan.edu.cn>
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

#ifndef polyorder_pseudospectral_h
#define polyorder_pseudospectral_h

#include <blitz/array.h>

#include "Updater.h"
#include "Grid.h"
#include "Propagator.h"
#include "UnitCell.h"
#include "fftw3.h"

class PseudoSpectral:public Updater{
public:
    PseudoSpectral(){}
    PseudoSpectral(const PseudoSpectral &rhs);

    PseudoSpectral(const UnitCell &uc,
                   const int Lx, const int Ly, const int Lz,
                   const double ds);
    PseudoSpectral(const blitz::Array<double,3> laplace);

    void solve(Propagator &, const Grid &);
    PseudoSpectral *clone() const;
    ~PseudoSpectral();

private:
    blitz::Array<double,3> _laplace;
    fftw_complex *_fftw_in;
    fftw_complex *_fftw_out;
    fftw_plan _p_forward;
    fftw_plan _p_backward;

    void init_fftw();
};

#endif

