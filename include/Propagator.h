/**
 * Propagator.h
 * Created at 2011.6.10
 *
 * Propagator is derived from Grid which implements
 * the virtual update function for propagators.
 *
 * HISTORY:
 * 2012.3.31
 *   1. From now on, the history of this file is tracked by Mercurial.
 *   2. Package name: polyorder.
 * 2011.6.20
 *   1. add Updater support
 * 2011.6.10
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

#ifndef polyorder_propagator_h
#define polyorder_propagator_h

#include <blitz/array.h>

#include "Grid.h"
#include "Updater.h"
#include "PseudoSpectral.h"

using std::string;

class Propagator:public Grid{
public:
    Propagator();

    Propagator(const string name, const Config &cfg,
               const int len, const double ds,
               Updater *up=0);
    Propagator(const string name, const Config &cfg,
               const int len, const double ds, const Grid &data,
               Updater *up=0);
    Propagator(const string name, const Config &cfg,
               const int len, const double ds,
               const blitz::Array<double,3> data,
               Updater *up=0);

    double operator() (int s, int ix, int iy, int iz);
    const double operator() (int s, int ix, int iy, int iz) const;
    const blitz::Array<double,3> operator[] (const int i) const;
    blitz::Array<double,3> operator[] (const int i);
    const blitz::Array<double,3> get_tail() const;
    void set_head(const Grid &u);
    void set_head(const blitz::Array<double,3> bu);

    blitz::Array<double,4> qs();
    blitz::Array<double,4> qs() const;
    void update(const Grid &w);
    void update(const Grid &w,const Updater *up);
    double Q(const int s);
    double Qt();
    int len() const;
    double ds() const;
    void save(const string file);
    ~Propagator();
private:
    int _length;
    double _ds;
    blitz::Array<double,4> _qs;
    Updater *_updater;

    void init();
    void init(const Grid &data);
    void set_updater(Updater *up);
};

#endif
