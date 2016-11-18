/**
 * Yita.h
 * Created at 2011.6.16
 *
 * Yita is derived from Grid which implemented
 * the virtual update function for updating the imcompressible field.
 *
 * HISTORY:
 * 2012.4.1
 *   1. From now on, the history of this file is tracked by Mercurial.
 *   2. Package name: polyorder.
 * 2011.6.10
 *   1. original version
 * 2011.6.16
 *   1. in operator=, add return *this;
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

#ifndef polyorder_yita_h
#define polyorder_yita_h

#include <blitz/array.h>

#include "Grid.h"

using std::string;

class Yita:public Grid{
public:
    Yita(){}
    //Yita(const Yita &rhs):Grid(rhs),_lambda(rhs._lambda){}

    Yita(const string name, const Config &cfg, const double lam);
    Yita(const string name, const Config &cfg,
         const double val, const double lam);
    Yita(const string name, const Config &cfg,
         const blitz::Array<double,3> data,
         const double lam);
    Yita(const string name, const Config &cfg, const string file,
         const double lam);
    Yita(const string name, const Config &cfg, const double lam,
         const double low, const double high, const size_t seed=0);

    Yita & operator= (const Grid &rhs);

    void update(const Grid &u);
    void update(const blitz::Array<double,3> bu);
    void set_lambda(const double lam);
    double lambda() const;
private:
    double _lambda; // relaxation parameter
};

#endif
