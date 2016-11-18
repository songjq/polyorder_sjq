/**
 * FieldE.h
 * Created at 2011.6.21
 *
 * FieldE abstracts the electrostatic fields.
 *
 * HISTORY:
 * 2012.4.2
 *   1. From now on, the history of this file is tracked by Mercurial.
 *   2. Package name: polyorder.
 * 2011.6.21
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


#ifndef polyorder_fielde_h
#define polyorder_fielde_h

#include <blitz/array.h>

#include "Updater.h"
#include "Grid.h"
#include "MUD2D.h"
#include "MUD3D.h"

using std::string;

class FieldE:public Grid{
public:
    FieldE(){}
    FieldE(const FieldE &rhs):Grid(rhs),_eps_type(rhs._eps_type),_lambda(rhs._lambda),_cc(rhs._cc),_updater(0){set_updater(rhs._updater);}

    FieldE(const string name,const int eps_type,const Config &cfg,const double lam,const double cc,const Updater *up=0):Grid(name,cfg),_eps_type(eps_type),_lambda(lam),_cc(cc),_updater(0){set_updater(up);}

    FieldE(const string name,const int eps_type,const Config &cfg,const double val,const double lam,const double cc,const Updater *up=0):Grid(name,cfg,val),_eps_type(eps_type),_lambda(lam),_cc(cc),_updater(0){set_updater(up);}

    FieldE(const string name,const int eps_type,const Config &cfg,const blitz::Array<double,3> data,const double lam,const double cc,const Updater *up=0):Grid(name,cfg,data),_eps_type(eps_type),_lambda(lam),_cc(cc),_updater(0){set_updater(up);}

    FieldE(const string name,const int eps_type,const Config &cfg,const string file,const double lam,const double cc,const Updater *up=0):Grid(name,cfg,file),_eps_type(eps_type),_lambda(lam),_cc(cc),_updater(0){set_updater(up);}

    FieldE(const string name,const int eps_type,const Config &cfg,const double lam,const double cc,const double low,const double high,const size_t seed=0,const Updater *up=0):Grid(name,cfg,low,high,seed),_eps_type(eps_type),_lambda(lam),_cc(cc),_updater(0){set_updater(up);}

    FieldE & operator= (const FieldE &rhs){if(&rhs != this){Grid::operator=(rhs);set_updater(rhs._updater);}return *this;}
    FieldE & operator= (const Grid &rhs){Grid::operator=(rhs);return *this;}

    void set_eps(const Grid &eps);

    void update(const Grid &rho){update(rho,_updater);}
    void update(const Grid &rho,Updater *up);

private:
    int _eps_type;
    double _lambda;
    double _cc; // here, _cc may be N/eps
    Updater *_updater;

    void set_updater(const Updater *up);
};

#endif

