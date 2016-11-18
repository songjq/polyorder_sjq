/**
 * MUD2D.h
 * Created at 2011.6.23
 *
 * MUD2D is a C++ wrapper for MUDPACK mud2sp and mud2sa 
 * to solve separable elliptic equations and self-adjoint 
 * elliptic equations.
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

#ifndef polyorder_mud2d_h
#define polyorder_mud2d_h

#include "MUD.h"
#include "Grid.h"

using std::string;

class MUD2D:public MUD{
public:
    typedef void (*COF_FUNC)(MUD_FLOAT*,MUD_FLOAT*,MUD_FLOAT*,MUD_FLOAT*);
    typedef void (*BND_FUNC)(MUD_INT*,MUD_FLOAT*,MUD_FLOAT*,MUD_FLOAT*);
    typedef double (*SIG_FUNC)(MUD_FLOAT*,MUD_FLOAT*);
    typedef void (*COF_CR_FUNC)(MUD_FLOAT*,MUD_FLOAT*,MUD_FLOAT*,MUD_FLOAT*,MUD_FLOAT*,MUD_FLOAT*,MUD_FLOAT*,MUD_FLOAT*);
    typedef void (*BND_CR_FUNC)(MUD_INT*,MUD_FLOAT*,MUD_FLOAT*,MUD_FLOAT*,MUD_FLOAT*,MUD_FLOAT*);

    MUD2D(){}
    MUD2D(const MUD2D &);
    MUD2D(string t,int Lx,int Ly,double xb,double yd,double xa=0.0,double yc=0.0,int fft2mg_mode=0);
    void init(); // after set all states, must call this function

    void set_cof(COF_FUNC pcofx,COF_FUNC pcofy){_pcofx=pcofx;_pcofy=pcofy;}
    void set_cof(COF_CR_FUNC pcof_cr){_pcof_cr=pcof_cr;}
    void set_sig(SIG_FUNC psigx,SIG_FUNC psigy){_psigx=psigx;_psigy=psigy;}
    void set_xlmbda(SIG_FUNC xlmbda){_xlmbda=xlmbda;}
    void set_boundary_function(BND_FUNC pbnd){_pboundary=pbnd;}
    void set_boundary_function(BND_CR_FUNC pbnd_cr){_pboundary_cr=pbnd_cr;}
    void set_boundary_type(int nxa=0,int nxb=0,int nyc=0,int nyd=0){
        _iparm[1] = nxa;
        _iparm[2] = nxb;
        _iparm[3] = nyc;
        _iparm[4] = nyd;
    }
    void set_guess(bool is_guess=false){
        _is_guess = is_guess;
        if(_is_guess)
            _iparm[11] = 1;
        else
            _iparm[11] = 0;
    }
    void set_maxcy(int maxcy=1){_iparm[12]=maxcy;}
    void set_tolmax(MUD_FLOAT tolmax=0.0){_fparm[4]=tolmax;}
    void set_grid(int Lx,int Ly,double xb,double yd,double xa=0.0,double yc=0.0);
    void set_sig_parm();
    void set_cof_cr_parm();
    void set_method(int imethod=0);

    void solve(Grid &, const Grid &);
    void solve(Grid &, const Grid &) const{}
    void set_data(const blitz::Array<double,2> sig_data_fft);
    void set_data(const Grid& cof_cr_data_fft);

    void display() const;
    MUD2D *clone() const{return new MUD2D(*this);}
    ~MUD2D(){}

private:
    // coefficients functions for mud2sp
    COF_FUNC _pcofx;
    COF_FUNC _pcofy;
    // coefficients functions for mud2sa
    SIG_FUNC _psigx;
    SIG_FUNC _psigy;
    SIG_FUNC _xlmbda;
    // coefficient function for mud2cr
    COF_CR_FUNC _pcof_cr;
    // boundary function pointer for mud2sp and mud2sa
    BND_FUNC _pboundary;
    BND_CR_FUNC _pboundary_cr;

};

#endif

