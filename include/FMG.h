/**
 * FMG.h
 * Created at 2011.6.21
 *
 * FMG is derived from Updater. 
 * Itself is again a base class abstracting the basic interface for 
 * Full Multigrid Algorithm.
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

#ifndef polyorder_fmg_h
#define polyorder_fmg_h

class FMG:public Updater{
public:
	Cfmg_pbe_2d():n(0),ng(0),ncycle(0),NPRE(1),NPOST(1),uj(NULL),uj1(NULL){}
	Cfmg_pbe_2d(MatDoub_IO &rho0, const Doub llx, const Doub lly, const Int ncycle);
	void mglin(MatDoub_IO &u,double &resnorm);
	~Cfmg_pbe_2d();

private:
    int _ncycle;
    int _npre;
    int _npost;
    int _n;

    double _lx;
    double _ly;
    double _lz;
	NRvector<NRmatrix<Doub> *> rho;

	void interp_bilinear(MatDoub_O &uf, MatDoub_I &uc);
	void interp_bicubic(MatDoub_O &uf, MatDoub_I &uc);
	void addint(MatDoub_O &uf, MatDoub_I &uc, MatDoub_O &res);
	void slvsml(MatDoub_O &u, MatDoub_I &rhs);
	void relax(MatDoub_IO &u, MatDoub_I &rhs);
	void resid(MatDoub_O &res, MatDoub_I &u, MatDoub_I &rhs);
	void rstrct(MatDoub_O &uc, MatDoub_I &uf);
	void mg(Int j, MatDoub_IO &u, MatDoub_I &rhs);
};

#endif

