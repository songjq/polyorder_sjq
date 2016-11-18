/**
 * RQM4.h
 * Created at 2014.9.16 by Yi-Xin Liu
 *
 * RQM4 is derived from Updater implementing
 * 4th-order operator splitting algorithm for solving propagation equations
 * (modified diffusion function).
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

#ifndef polyorder_rqm4_h
#define polyorder_rqm4_h

#include <blitz/array.h>

#include "Updater.h"
#include "Grid.h"
#include "Propagator.h"
#include "UnitCell.h"
#include "fftw3.h"

class RQM4:public Updater{
public:
    RQM4(){}
    RQM4(const RQM4 &rhs);
    RQM4(const UnitCell &uc, const int Lx, const int Ly, const int Lz,
         const double ds);

    void solve(Propagator &, const Grid &);
    RQM4 *clone() const;
    ~RQM4();

private:
    blitz::Array<double,3> _laplace;   // = exp(-ds*k^2)
    blitz::Array<double,3> _laplace2;  // = exp(-0.5*ds*k^2)
    fftw_complex *_fftw_in;
    fftw_complex *_fftw_out;
    fftw_plan _p_forward;
    fftw_plan _p_backward;

    void init_fftw();
};

#endif

