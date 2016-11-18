/**
 * Density.h
 * Created at 2011.6.10
 *
 * Density is derived from Grid which implemented
 * the virtual update function for updating density fields.
 *
 * HISTORY:
 * 2012.4.1
 *   1. From now on, the history of this file is tracked by Mercurial.
 *   2. Package name: polyorder.
 * 2011.6.20
 *   1. set the default _cc=1.0
 * 2011.6.17
 *   1. Add Updater support
 * 2011.6.16
 *   1. in operator=, add return *this;
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

#ifndef polyorder_density_h
#define polyorder_density_h

#include <blitz/array.h>

#include "Grid.h"
#include "Propagator.h"
#include "Updater.h"

using std::string;

class Density:public Grid{
public:
    Density();
    Density(const Density &rhs);

    Density(const string name, const Config &cfg, const Updater *up=0);

    Density(const string name, const Config &cfg, const double val,
            const Updater *up=0);
    Density(const string name, const Config &cfg,
            const blitz::Array<double,3> data,
            const Updater *up=0);
    Density(const string name, const Config &cfg, const string file,
            const Updater *up=0);

    Density(const string name, const Config &cfg,
            const double low, const double high,
            const size_t seed=0, const Updater *up=0);

    Density & operator= (const Density &rhs);
    Density & operator= (const Grid &rhs);


    void update(const Propagator &q, const Propagator &qc);
    void update(const Propagator &q, const Propagator &qc, const double cc);
    void update(const Propagator &q, const Propagator &qc, const Updater *up);
    void update(const Propagator &q, const Propagator &qc,
                const double cc, const Updater *up);
    void set_cc(const double cc);
    double cc() const;
    ~Density();
private:
    double _cc;
    Updater *_updater;

    void _update(const Propagator &q,const Propagator &qc);
    void set_updater(const Updater *up);
};

#endif
