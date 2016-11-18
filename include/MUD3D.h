/**
 * MUD3D.h
 * Created at 2012.4.16
 *
 * MUD3D is a C++ wrapper for MUDPACK mud3sp and mud3sa 
 * to solve separable elliptic equations in 3D space.
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

#ifndef polyorder_mud3d_h
#define polyorder_mud3d_h

#include "MUD.h"
#include "Grid.h"

using std::string;

class MUD3D:public MUD{
public:
    typedef void (*COF_FUNC)(MUD_FLOAT*,MUD_FLOAT*,MUD_FLOAT*,MUD_FLOAT*);
    typedef void (*BND_FUNC)(MUD_INT*,MUD_FLOAT*,MUD_FLOAT*,MUD_FLOAT*,MUD_FLOAT*);
    typedef double (*SIG_FUNC)(MUD_FLOAT*,MUD_FLOAT*,MUD_FLOAT*);
    typedef void (*COF_CR_FUNC)(MUD_FLOAT *,MUD_FLOAT *,MUD_FLOAT *,MUD_FLOAT *,MUD_FLOAT *,MUD_FLOAT *,MUD_FLOAT *,MUD_FLOAT *,MUD_FLOAT *,MUD_FLOAT *);
    typedef void (*CRS_CR_FUNC)(MUD_FLOAT*,MUD_FLOAT*,MUD_FLOAT*,MUD_FLOAT*);
    typedef void (*BND_CR_FUNC)(MUD_INT*,MUD_FLOAT*,MUD_FLOAT*,MUD_FLOAT*,MUD_FLOAT*,MUD_FLOAT*,MUD_FLOAT*);

    MUD3D(){}
    MUD3D(const MUD3D &);
    MUD3D(const string &t,int Lx,int Ly,int Lz,double xb,double yd,double zf,double xa=0.0,double yc=0.0,double ze=0.0,int fft2mg_mode=0);
    void init(); // after set all states, must call this function

    void set_cof(COF_FUNC pcofx,COF_FUNC pcofy,COF_FUNC pcofz){
        _pcofx=pcofx;_pcofy=pcofy;_pcofz=pcofz;}
    void set_cof(COF_CR_FUNC pcof_cr,CRS_CR_FUNC pcrsxy,CRS_CR_FUNC pcrsxz,CRS_CR_FUNC pcrsyz){
        _pcof_cr=pcof_cr;_pcrsxy=pcrsxy;_pcrsxz=pcrsxz;_pcrsyz=pcrsyz;}
    void set_sig(SIG_FUNC psigx,SIG_FUNC psigy,SIG_FUNC psigz){
        _psigx=psigx;_psigy=psigy;_psigz=psigz;}
    void set_xlmbda(SIG_FUNC xlmbda){_xlmbda=xlmbda;}
    void set_boundary_function(BND_FUNC pbnd){_pboundary=pbnd;}
    void set_boundary_function(BND_CR_FUNC pbnd_cr){_pboundary_cr=pbnd_cr;}
    void set_boundary_type(int nxa=0,int nxb=0,int nyc=0,int nyd=0,int nze=0,int nzf=0){
        _iparm[1] = nxa;
        _iparm[2] = nxb;
        _iparm[3] = nyc;
        _iparm[4] = nyd;
        _iparm[5] = nze;
        _iparm[6] = nzf;
    }
    void set_guess(bool is_guess=false){
        _is_guess = is_guess;
        if(_is_guess)
            _iparm[16] = 1;
        else
            _iparm[16] = 0;
    }
    void set_maxcy(int maxcy=1){_iparm[17]=maxcy;}
    void set_tolmax(MUD_FLOAT tolmax=0.0){_fparm[6]=tolmax;}
    void set_grid(int Lx,int Ly,int Lz,double xb,double yd,double zf,double xa=0.0,double yc=0.0,double ze=0.0);
    void set_cr_extras(int icrosxy=1,int icrosxz=0,int icrosyz=0,double tol=1.0e-6,int maxit=8){
        // after read the mud3cr.c source code
        // tol must > 0.0, = 0.0 is not allowed.
        // This is inconsistent with the documentation mud3cr.d
        _icros[0]=icrosxy;_icros[1]=icrosxz;_icros[2]=icrosyz;
        _tol=tol;_maxit=maxit;_rmax.resize(_maxit);}
    void set_sig_parm();
    void set_cof_cr_parm();
    void set_method(int imethod=0);

    void solve(Grid &, const Grid &);
    void solve(Grid &, const Grid &) const{}
    void set_data(const blitz::Array<double,3> sig_data_fft);
    void set_data(const Grid& cof_cr_data_fft);

    void display() const;
    MUD3D *clone() const{return new MUD3D(*this);}
    ~MUD3D(){}

private:
    // coefficients functions for mud3sp
    COF_FUNC _pcofx;
    COF_FUNC _pcofy;
    COF_FUNC _pcofz;
    // coefficients functions for mud3sa
    SIG_FUNC _psigx;
    SIG_FUNC _psigy;
    SIG_FUNC _psigz;
    SIG_FUNC _xlmbda;
    // boundary function pointer for mud3sp and mud3sa
    BND_FUNC _pboundary;
    // coefficients functions for mud3cr
    COF_CR_FUNC _pcof_cr;
    CRS_CR_FUNC _pcrsxy;
    CRS_CR_FUNC _pcrsxz;
    CRS_CR_FUNC _pcrsyz;
    // boundary function pointer for mud3cr
    BND_CR_FUNC _pboundary_cr;

    // For mud3cr only
    MUD_INT _icros[3]; // input
    MUD_FLOAT _tol; // input
    MUD_INT _maxit; // input
    MUD_INT _iouter; // output
    blitz::Array<MUD_FLOAT,1> _rmax; // output
};

#endif

