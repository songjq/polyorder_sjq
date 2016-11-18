/**
 * DensitySmall.h
 * Created at 2011.6.16
 *
 * DensitySmall is derived from Grid which implemented
 * the virtual update function for updating density fields
 * of small molecules in particular.
 *
 * HISTORY:
 * 2012.4.1
 *   1. From now on, the history of this file is tracked by Mercurial.
 *   2. Package name: polyorder.
 * 2011.6.16
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

#ifndef polyorder_density_small_h
#define polyorder_density_small_h

#include "Grid.h"

//#include <blitz/Array.h>

using std::string;

class DensitySmall:public Grid{
public:
    DensitySmall(){}
//    DensitySmall(const DensitySmall &rhs):Grid(rhs),_cc(rhs.cc()){}

    DensitySmall(const string name, const Config &cfg, const double cc);
    DensitySmall(const string name, const Config &cfg, const double val,
                 const double cc);
    DensitySmall(const string name, const Config &cfg,
                 const blitz::Array<double,3> data,
                 const double cc);
    DensitySmall(const string name, const Config &cfg, const string file,
                 const double cc);
    DensitySmall(const string name, const Config &cfg, const double cc,
                 const double low, const double high, const size_t seed=0);

    DensitySmall & operator= (const Grid &rhs);

    void update(const Grid &g);
    void update(const Grid &g,const double cc);
    void set_cc(const double cc);
    double cc() const;
private:
    double _cc; // the expected density value

    const double Q(const Grid &g) const; // to be called by update() only
    void _update(const Grid &g);
};

#endif

