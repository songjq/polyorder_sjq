/**
 * FMG2D.h
 * Created at 2011.6.21
 *
 * FMG2D is derived from fmg for solving linearized Poisson-Boltzmann 
 * equation (PBE) with constant dielectric constant (CDC) 
 * periodic boundary condition (PBC) in two-dimension (2D) space 
 * using full multigrid algorithm (FMG). 
 *
 * Finite diference method is adopt to discretize the PBE. 
 * The implementaion is based on Numercial Recipes's mglin.h
 * 
 * It uses red-black Gauss-Seidel as the smoothing operator, 
 * bilinear interpolation for MG prologation (P), 
 * tricubic interpolation for FMG prolongation and 
 * full-weighting for Restriction (R).
 * 
 * General usage:
 *		fmg2d *pmg;
 *		pmg=new fmg2d(g,1,2,2); // (Grid &, ncycle, NPRE, NPOST)
 *		pmg->mglin(u);
 *		delete pmg;
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

#ifndef polyorder_fmg_2d_h
#define polyorder_fmg_2d_h

#include <cstdlib>
#include <cmath>
#include <string>
#include <iostream>

#include "lyxDef.h"
#include "CNDArray.h" // Typedef (UlLong, Doub, UInt); Class (NRvector, NRmatrix)

class fmg2d:public fmg{
private:
	Int n,ng,ncycle;
	Doub lx,ly; // a rectangular domain with width lx and  height ly
	MatDoub *uj,*uj1;
	NRvector<NRmatrix<Doub> *> rho;

	void interp_bilinear(MatDoub_O &uf, MatDoub_I &uc);
	void interp_bicubic(MatDoub_O &uf, MatDoub_I &uc);
	void addint(MatDoub_O &uf, MatDoub_I &uc, MatDoub_O &res);
	void slvsml(MatDoub_O &u, MatDoub_I &rhs);
	void relax(MatDoub_IO &u, MatDoub_I &rhs);
	void resid(MatDoub_O &res, MatDoub_I &u, MatDoub_I &rhs);
	void rstrct(MatDoub_O &uc, MatDoub_I &uf);
	void mg(Int j, MatDoub_IO &u, MatDoub_I &rhs);

public:
	Int NPRE,NPOST;

	Cfmg_pbe_2d():n(0),ng(0),ncycle(0),NPRE(1),NPOST(1),uj(NULL),uj1(NULL){}
	Cfmg_pbe_2d(MatDoub_IO &rho0, const Doub llx, const Doub lly, const Int ncycle);
	void mglin(MatDoub_IO &u,double &resnorm);
	~Cfmg_pbe_2d();
};

#endif

