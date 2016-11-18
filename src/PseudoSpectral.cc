/**
 * PseudoSpectral.cc
 * Created at 2011.6.20
 *
 * Implementation of PseudoSpectral.h.
 *
 * Copyright (C) 2012-2014 Yi-Xin Liu <lyx@fudan.edu.cn>
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

#include "PseudoSpectral.h"

PseudoSpectral::PseudoSpectral(const PseudoSpectral &rhs):
                               _laplace(rhs._laplace){
    init_fftw();
}

PseudoSpectral::PseudoSpectral(const UnitCell &uc,
                               const int Lx, const int Ly, const int Lz,
                               const double ds):_laplace(Lx,Ly,Lz){
    _laplace = uc.calc_kLaplacian(Lx, Ly, Lz, ds);
    init_fftw();
}

PseudoSpectral::PseudoSpectral(const blitz::Array<double,3> laplace):
                               _laplace(laplace.shape()){
    _laplace = laplace;
    init_fftw();
}

PseudoSpectral* PseudoSpectral::clone() const{
    return new PseudoSpectral(*this);
}

void PseudoSpectral::init_fftw(){
    int Lx = _laplace.rows();
    int Ly = _laplace.cols();
    int Lz = _laplace.depth();
    _fftw_in = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*Lx*Ly*Lz));
    _fftw_out = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*Lx*Ly*Lz));
    _p_forward = fftw_plan_dft_3d(Lx, Ly, Lz, _fftw_in, _fftw_out,
                                  FFTW_FORWARD, FFTW_ESTIMATE);
    _p_backward = fftw_plan_dft_3d(Lx, Ly, Lz, _fftw_out, _fftw_in,
                                   FFTW_BACKWARD, FFTW_ESTIMATE);
}

PseudoSpectral::~PseudoSpectral(){
    fftw_free(_fftw_in);
    fftw_free(_fftw_out);
    fftw_destroy_plan(_p_forward);
    fftw_destroy_plan(_p_backward);
}

void PseudoSpectral::solve(Propagator &q, const Grid &w){
    int Lx = q.Lx();
    int Ly = q.Ly();
    int Lz = q.Lz();
    double ds = q.ds();
    blitz::Array<double, 4> qs(q.qs());
    int ngrids = Lx * Ly * Lz;

    blitz::Array<double, 3> expq(Lx, Ly, Lz);
    expq = blitz::exp(-0.5 * ds * w.data());

    blitz::Range all = blitz::Range::all();
    blitz::Array<complex<double>,3> bfftw_in(
                            reinterpret_cast<complex<double> *>(_fftw_in),
                            blitz::shape(Lx, Ly, Lz),
                            blitz::neverDeleteData);
    blitz::Array<complex<double>,3> bfftw_out(
                            reinterpret_cast<complex<double> *>(_fftw_out),
                            blitz::shape(Lx, Ly, Lz),
                            blitz::neverDeleteData);

    // q(r,j), j=s0,s0+1,...,s1
    for (int s=0; s<q.len()-1; s++){
        blitz::Array<double,3> q1(qs(s, all, all, all));
        blitz::Array<double,3> q2(qs(s+1, all, all, all));

        bfftw_in = blitz::zip(expq*q1, 0, complex<double>());
        fftw_execute(_p_forward);  // fftw_in => fftw_out

        bfftw_out *= _laplace;

        fftw_execute(_p_backward);  // fftw_out => fftw_in

        q2 = (blitz::real(bfftw_in) * expq) / ngrids;
    }
}

