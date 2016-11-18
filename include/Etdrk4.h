/**
 * Etdrk4.h
 * Created at 2014.5.26
 *
 * Etdrk4 is derived from Updater implementing
 * ETDRK4 algorithms for solving propagation equations
 * (modified diffusion function).
 *
 * Copyright (C) 2014 Yi-Xin Liu <lyx@fudan.edu.cn>
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

#ifndef polyorder_etdrk4_h
#define polyorder_etdrk4_h

#include "common.h"
#include "Updater.h"
#include "UnitCell.h"
#include "Grid.h"
#include "Propagator.h"

#include "Boundary.h"
#include "armadillo"
#include "fftw3.h"

class Etdrk4:public Updater{
public:
    Etdrk4(){}
    Etdrk4(const Etdrk4 &rhs);
    Etdrk4(const Config &cfg, const double ds,
           const ETDRK4SCHEME stype=ETDRK4SCHEME::KROGSTAD,
           const arma::uword M=16);
    Etdrk4(const UnitCell &uc, const arma::uword dim,
           const arma::uword Lx, const arma::uword Ly, const arma::uword Lz,
           const double ds,
           const ConfineType ctype,
           const Boundary &lb, const Boundary &rb,
           const ETDRK4SCHEME stype=ETDRK4SCHEME::KROGSTAD,
           const arma::uword M=16);  

    void solve(Propagator &q, const Grid &w);

    void solve(const arma::colvec &u, const arma::colvec &w,
               const arma::uword Ns, arma::mat &q);  // 1D col major
    void solve(const arma::rowvec &u, const arma::rowvec &w,
               const arma::uword Ns, arma::mat &q);  // 1D row major

    void solve(const arma::mat &u, const arma::mat &w,
               const arma::uword Ns, arma::cube &q);  // 2D

    void solve(const arma::cube &u, const arma::cube &w,
               const arma::uword Ns, arma::field<arma::cube> &q);  // 3D

    Etdrk4 *clone() const;
    void save(const string file="etdrk4_coeff.mat");
    ~Etdrk4();

private:
    UnitCell _uc;
    arma::uword _dim;
    arma::uword _Lx, _Ly, _Lz;
    double _ds;
    ConfineType _ctype;
    Boundary _lb, _rb;
    ETDRK4SCHEME _stype;
    arma::uword _M;

    /* For PBC only */
    arma::cx_mat arma_in;
    arma::cx_mat arma_out;
    fftw_complex *_fftw_in;
    fftw_complex *_fftw_out;
    fftw_plan _p_forward;
    fftw_plan _p_backward;

    /* ETDRK4 coefficients for 1D */
    // Cox-Matthews: E1, E2, Q, f1, f2, f3
    // Krogstad: E1, E2, Q, f1, f2, f3, f4, f5
    arma::mat E1, E2, Q, f1, f2, f3, f4, f5;

    /** ETDRK4 coefficients for 2D and 3D
     *
     * 2D: the slice index of Cube is naturally the index of dimension
     * For example, the index range of x direction is [0, Lx-1]. y direction
     * is the confining direction with Chebyshev grid, whose index range
     * is [0, Ly-1]. For RBC-RBC, the Cheyshev differential matrix D's shape is
     * (Ly x Ly). Thus, the shape of cube is (Ly x Ly x Lx).
     *
     * 3D: the slice index of Cube is a combination of
     *     indexes of two dimensions
     * For example, the index range of x and y directions are [0, Lx-1] and
     * [0, Ly-1]. z direction is the confining direction with Chebyshev grid,
     * whose index range is [0, Lz-1]. For RBC-RBC, the Cheyshev
     * differential matrix D's shape is (Lz x Lz). Thus, the shape of cube
     * is (Ly x Ly x (Lx*Ly)), and the index range of slice is [0, Lx*Ly-1].
     * Assume i in [0, Lx-1], j in [0, Ly-1], k in [0, Lx*Ly-1], we have the
     * relation k = i + Lx * j.
     */
    arma::cube mE1, mE2, mQ, mf1, mf2, mf3, mf4, mf5;

    void init_fftw();
    arma::mat phi(const unsigned int l, /* phi_l */
                  const arma::mat &A,
                  const double t,
                  const arma::uword M);
    void init_coefficients();
    void init_coefficients_size(const arma::uword N);
    void init_coefficients_size(const arma::uword N, const arma::uword Nc);
    void init_coefficients_1d();
    void init_coefficients_2d();
    void init_coefficients_3d();

    arma::mat scheme_cox(arma::colvec &u, const arma::colvec &w,
                         const arma::uword Ns);  // 1D col major
    arma::mat scheme_cox(arma::rowvec &u, const arma::rowvec &w,
                         const arma::uword Ns);  // 1D row major
    arma::mat scheme_krogstad(arma::colvec &u, const arma::colvec &w,
                              const arma::uword Ns);  // 1D col major
    arma::mat scheme_krogstad(arma::rowvec &u, const arma::rowvec &w,
                              const arma::uword Ns);  // 1D row major

    arma::cube scheme_cox(arma::mat &u, const arma::mat &w,
                         const arma::uword Ns);  // 2D
    arma::cube scheme_krogstad(arma::mat &u, const arma::mat &w,
                              const arma::uword Ns);  // 2D

    arma::field<arma::cube> scheme_cox(arma::cube &u, const arma::cube &w,
                                       const arma::uword Ns);
    arma::field<arma::cube> scheme_krogstad_armafft(arma::cube &u,
                                                    const arma::cube &w,
                                                    const arma::uword Ns);
    arma::field<arma::cube> scheme_krogstad_fftw_raw(arma::cube &u,
                                                     const arma::cube &w,
                                                     const arma::uword Ns);
    arma::field<arma::cube> scheme_krogstad(arma::cube &u, const arma::cube &w,
                                            const arma::uword Ns);  // FFTW

    void calc_coefficients(const arma::mat &L, const double t,
                           const arma::uword M);
};

#endif

