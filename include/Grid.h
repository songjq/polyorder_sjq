/**
 * Grid.h
 * Created at 2011.6.6

 * Grid is the base class abstrating the discrete simulation cell
 * associated with values. It can be derived to represent fields,
 * densities, and propagators.
 *
 * HISTORY:
 * 2012.3.31
 *   1. From now on, the history of this file is tracked by Mercurial.
 *   2. Package name: polyorder.
 * 2011.6.16
 *   1. in operator=, add return *this
 * 2011.6.9
 *   1. all functionalities are tested and work properly.
 *   2. overload operator +,-,*,+=,-=,*=, fully tested.
 * 2011.6.6
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

#ifndef polyorder_grid_h
#define polyorder_grid_h

#include <cstdlib>
#include <ctime>
#include <cmath>
#include <string>
#include <iostream>

#include <blitz/array.h>

#include "lyxDef.h" // Typedefs(UlLong, Doub, UInt) for ran.h
#include "CMatFile.h" // Matlab MAT-file processor
#include "ran.h" // Numerical Recipes v3.0

#include "common.h"
#include "Config.h" // Reading configuration file
#include "UnitCell.h"

using std::string;

class Grid{
public:
    Grid(){}
    // must define the copy constructor since `_data(rhs._data)`
    // does not copy data from `rhs` to `this`
    Grid(const Grid &rhs);

    Grid(const UnitCell &uc, const int Lx, const int Ly, const int Lz);
    Grid(const UnitCell &uc, const int Lx, const int Ly, const int Lz,
         const double val);
    Grid(const UnitCell &uc, const int Lx, const int Ly, const int Lz,
         const blitz::Array<double,3> data);

	Grid(const string name);
	Grid(const string name, const Config &cfg);
	Grid(const string name, const Config &cfg, const double val);
	Grid(const string name, const Config &cfg,
         const blitz::Array<double,3> data);

	Grid(const string name, const Config &cfg, const string file);
	Grid(const string name, const Config &cfg,
         const double low, const double high, const size_t seed=0);

    Grid & operator= (const Grid &rhs);

    blitz::Array<double,3> data();
    const blitz::Array<double,3> data() const;

    double & operator() (int ix, int iy, int iz);
    const double & operator() (int ix, int iy, int iz) const;
    // this hides the complicated internal structure of array for 2D space.
    double & operator() (int ix, int iy);
    const double & operator() (int ix, int iy) const;
    // this hides the complicated internal structure of array for 1D space.
    double & operator() (int ix);
    const double & operator() (int ix) const;

    Grid & operator+= (const Grid &rhs);
    Grid & operator+= (const double rhs);
    Grid & operator-= (const Grid &rhs);
    Grid & operator-= (const double rhs);
    Grid & operator*= (const Grid &rhs);
    Grid & operator*= (const double rhs);
    Grid & operator/= (const Grid &rhs);
    Grid & operator/= (const double rhs);

    void save(const string file);

    const string name() const;
    void set_name(const string name);
    const UnitCell uc() const;
    const int dim() const;
    const int Lx() const;
    const int Ly() const;
    const int Lz() const;
    const double lx() const;
    const double ly() const;
    const double lz() const;
    const double dx() const;
    const double dy() const;
    const double dz() const;
    //const GridType grid_type() const{return _grid_type;}
    const GridInitType grid_init_type() const;
    //const PhasePattern phase_pattern() const{return _phase_pattern;}
    const double sum() const;
    const double abs_sum() const;
    const double mean() const;
    const double abs_mean() const;
    const double min() const;
    const double max() const;

    const double quadrature() const;
    const double quadrature(const blitz::Array<double, 3> data) const;
    const double abs_quadrature() const;
    const double abs_quadrature(const blitz::Array<double, 3> data) const;

    const Grid diff2() const;
    const Grid diff2_1d() const;
    const Grid diff2_2d() const;
    const Grid diff2_hex_2d() const;
    const Grid diff2_3d() const;
    const Grid diff2_hex_3d() const;
    const Grid diff2_mono_3d() const;
    const Grid diffx() const;
    const Grid diffx_1d() const;
    const Grid diffx_2d() const;
    const Grid diffx_3d() const;
    const Grid diffy() const;
    const Grid diffy_2d() const;
    const Grid diffy_3d() const;
    const Grid diffz() const;

    const int ngrids() const;
    const int size() const;
    const double volume() const;
    const size_t seed() const;

    virtual void update(){}

protected:
    string _name;
    UnitCell _uc;
    int _Lx,_Ly,_Lz; // 1D: _Ly=_Lz=1; 2D: _Lz=1
    ConfineType _ctype;
    GridType _gtypex, _gtypey, _gtypez;
    GridInitType _grid_init_type;
    //PhasePattern _phase_pattern;
    blitz::Array<double,3> _data;
    size_t _seed;

private:
	void init(const Config &cfg);

    const double quadrature_1d(const blitz::Array<double, 3> data) const;
    const double quadrature_2d(const blitz::Array<double, 3> data) const;
    const double quadrature_3d(const blitz::Array<double, 3> data) const;
    const double mean(const blitz::Array<double, 3> data) const;

    void set_value(string file);
    void set_value(const size_t seed, const double low, const double high);
};

Grid operator+ (const Grid &,const Grid &);
Grid operator+ (const double,const Grid &);
Grid operator+ (const Grid &,const double);
Grid operator- (const Grid &,const Grid &);
Grid operator- (const double,const Grid &);
Grid operator- (const Grid &,const double);
Grid operator* (const Grid &,const Grid &);
Grid operator* (const double,const Grid &);
Grid operator* (const Grid &,const double);
Grid operator/ (const Grid &,const Grid &);
Grid operator/ (const double,const Grid &);
Grid operator/ (const Grid &,const double);

Grid exp(const Grid &);
Grid log(const Grid &);

#endif

