/**
 * Quadrature4.h
 * Created at 2014.9.15 by Yi-Xin Liu
 *
 * A set of 4-th order quadrature algorithms for contour integration.
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

#ifndef polyorder_quadrature4_h
#define polyorder_quadrature4_h

#include <blitz/array.h>

#include "Updater.h"
#include "Propagator.h"

class Quad4_Closed:public Updater{
public:
	void solve(blitz::Array<double,3> data,
               const Propagator &q,
               const Propagator &qc,
               const double cc) const;

    Quad4_Closed *clone() const;
};

class Quad4_Open:public Updater{
public:
    void solve(blitz::Array<double,3> data,
               const Propagator &q,
               const Propagator &qc,
               const double cc) const;

    Quad4_Open *clone() const;
};

class Quad4_Semiopen_Left:public Updater{
public:
    void solve(blitz::Array<double,3> data,
               const Propagator &q,
               const Propagator &qc,
               const double cc) const;

    Quad4_Semiopen_Left *clone() const;
};

class Quad4_Semiopen_Right:public Updater{
public:
    void solve(blitz::Array<double,3> data,
               const Propagator &q,
               const Propagator &qc,
               const double cc) const;

    Quad4_Semiopen_Right *clone() const;
};

#endif

