/**
 * UnitCell.h
 * Created at 2012.4.5
 *
 * UnitCell is the base class abstrating the discrete simulation cell
 * associated with values. It can be derived to represent fields,
 * densities, and propagators.
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

#ifndef polyorder_unitcell_h
#define polyorder_unitcell_h

#include <string>
#include <blitz/array.h>
#include <blitz/tinyvec-et.h>

#include "common.h"
#include "Config.h" // Reading configuration file
#include "CMatFile.h"

class UnitCell{
public:
    UnitCell(){}
    UnitCell(const UnitCell &rhs);

    UnitCell(const Config &cfg);
    UnitCell(const int dim,const CrystalSystemType cs,const CellParam &cp);

    void update(const CellParam &);
    void save(const string,int,int,int) const;

    UnitCell & operator= (const UnitCell &rhs);

    CrystalSystemType type_key() const;
    string type() const;
    const int dim() const;
    const double a() const;
    const double b() const;
    const double c() const;
    const double alpha() const;
    const double beta() const;
    const double gamma() const;
    const blitz::TinyVector<double,3> l() const;
    const double lx() const;
    const double ly() const;
    const double lz() const;
    const blitz::Array<double,3> x(int Lx,int Ly,int Lz) const;
    const blitz::Array<double,3> y(int Lx,int Ly,int Lz) const;
    const blitz::Array<double,3> z(int Lx,int Ly,int Lz) const;
    const blitz::TinyMatrix<double,3,3> h() const;
    const blitz::TinyMatrix<double,3,3> g() const;

    const blitz::Array<double,3> calc_kLaplacian(int Na,int Nb,int Nc,
                                                 double ds) const;
    const blitz::Array<double,3> calc_kLaplacian(blitz::TinyVector<int,3> vN,
                                                 double ds) const;
    const blitz::Array<double,3> calc_k2_orthogonal(const int Lx,
                                                    const int Ly,
                                                    const int Lz) const;
    const blitz::Array<double,3> calc_kLaplacian_orthogonal(const int Lx,
                                                            const int Ly,
                                                            const int Lz,
                                                            const double ds
                                                            ) const;

private:
    int _dim; // actual space dimension
    CrystalSystemType _crystal_system_type;
    CellParam _cell_param;
    blitz::TinyVector<double,3> _a,_b,_c;
    blitz::TinyVector<double,3> _l;
    blitz::TinyMatrix<double,3,3> _h,_g;

	void init();
	void init(const Config &cfg);
    void calc_shape_matrix1();
    void calc_shape_matrix2();
    void calc_shape_matrix3();
    double calc_k2(const blitz::TinyVector<double,3>,int,int,int) const;
};

#endif

