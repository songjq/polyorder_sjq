/**
 * Field.h
 * Created at 2011.6.10
 *
 * Field is derived from Grid which implemented the
 * virtual update function for updating fields.
 *
 * HISTORY:
 * 2012.4.1
 *   1. From now on, the history of this file is tracked by Mercurial.
 *   2. Package name: polyorder.
 * 2011.6.16
 *   2. in operator=, add return *this;
 * 2011.6.10
 *   1. original version
 *   2. all functionalities work properly
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


#ifndef polyorder_field_h
#define polyorder_field_h

#include <blitz/array.h>

#include "Grid.h"

using std::string;

class Field:public Grid{
public:
    Field(){}
//    Field(const Field &rhs):Grid(rhs),_lambda(rhs._lambda){}

    Field(const string name, const Config &cfg,const double lam);
    Field(const string name, const Config &cfg, const double val,
          const double lam);
    Field(const string name, const Config &cfg,
          const blitz::Array<double,3> data,
          const double lam);
    Field(const string name, const Config &cfg, const string file,
          const double lam);
    Field(const string name, const Config &cfg, const double lam,
          const double low, const double high, const size_t seed=0);

    Field & operator= (const Grid &rhs);

    void update(const Grid &u);
    void update(const blitz::Array<double,3> bu);
    void set_lambda(const double lam);
    double lambda() const;
    double Q(const double N) const;
    blitz::Array<double, 3> ws();
    blitz::Array<double, 3> ws() const;
private:
    double _lambda; // relaxation parameter
};

#endif
