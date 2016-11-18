/**
 * Updater.h
 * Created at 2011.6.20
 *
 * Updater is the base class providing interfaces for
 * the various algorithms to update Density, Field, and Propagator.
 *
 * HISTORY:
 * 2012.4.2
 *   1. From now on, the history of this file is tracked by Mercurial.
 *   2. Package name: polyorder.
 * 2011.6.20
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

#ifndef polyorder_updater_h
#define polyorder_updater_h

#include <iostream>
#include <blitz/array.h>

class Grid;
class Propagator;
class Density;

class Updater{
public:
    Updater(){}
    // Why not use pure virtual?
    // Because for different type of Object (Propagator, Density, and Field),
    // only one type of solve() is necessary.
    // Consider to split these Updater interface into three interfaces:
    //      PropagatorUpdater
    //      DensityUpdater
    //      FieldUpdater
    // for Propagator
    virtual void solve(Propagator &q, const Grid &w) {}
    // for Density
	virtual void solve(blitz::Array<double,3> data, const Propagator &q,
                       const Propagator &qc, const double cc) const {}
    // for Field
    virtual void solve(Grid &w, const Grid &data) const {}
    virtual void solve(Grid &w, const Grid &data) {}

    //virtual void set_data(const Grid&) = 0;
    //virtual void set_data(const blitz::Array<double,1>) = 0;
    //virtual void set_data(const blitz::Array<double,2>) = 0;
    //virtual void set_data(const blitz::Array<double,3>) = 0;
    virtual Updater *clone() const = 0;
    virtual ~Updater(){}
};

#endif

