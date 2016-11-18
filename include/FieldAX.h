/**
 * FieldAX.h
 * Created at 2014.6.10
 *
 * FieldAX is derived from Grid which implemented the
 * virtual update function for updating FieldAXs, specially designed for
 * Anderson mixing algorithm.
 * Here, we follow Matsen's ramping up scheme to maximize the efficiency of
 * Anderson mixing.
 *
 * References:
 *      1. Stasiak, P.; Matsen, M. W. EPJE, 2011, 34, 110.
 *      2. Matsen, M. W. EPJE, 2009, 30, 361.
 *      3. Thompson, R.; Rasmussen, K.; Lookman, T. JCP, 2004, 120, 31.
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


#ifndef polyorder_FieldAX_h
#define polyorder_FieldAX_h

#include <blitz/array.h>
#include "armadillo"

#include "Grid.h"

using std::string;

class FieldAX:public Grid{
public:
    FieldAX(){}
    //FieldAX(const FieldAX &rhs):Grid(rhs),_lambda(rhs._lambda){}

    FieldAX(const string name, const Config &cfg, const double lam,
            arma::uword n_history);
    FieldAX(const string name, const Config &cfg, const double val,
            const double lam, arma::uword n_history);
    FieldAX(const string name, const Config &cfg,
            const blitz::Array<double,3> data,
            const double lam, arma::uword n_history);
    FieldAX(const string name, const Config &cfg, const string file,
            const double lam, arma::uword n_history);
    FieldAX(const string name, const Config &cfg, const double lam,
            arma::uword n_history,
            const double low, const double high, const size_t seed=0);

    FieldAX & operator= (const Grid &rhs);
    FieldAX & operator= (const FieldAX &rhs);

    /**
     * U_{ij} = B_{00} - B{(i+1)0} - B{0(j+1)} + B_{(i+1)(j+1)}
     * for i, j = 0, 1, ..., nn
     * where nn = min(k, n).
     */
    arma::mat calc_U() const;
    /**
     * V_{i} = B_{00} - B_{(i+1)0}    for i = 0, 1, ..., (n-1)
     */
    arma::vec calc_V() const;

    void pre_update(const Grid &u);
    void update(const arma::vec &C);

    void update(const Grid &u);
    void update(const blitz::Array<double,3> bu);
    void update(const arma::cube &wt);
    void set_lambda(const double lam);
    const double lambda() const;
    const arma::uword n_iteration() const;
    const arma::uword n_history() const;
    const arma::uword current_index() const;
    const arma::cube current_w() const;
    const double current_w_quadrature() const;
    const double current_w_abs_quadrature() const;
    const double current_w2_quadrature() const;
    const arma::cube current_d() const;
    const double current_d_quadrature() const;
    const double current_d_abs_quadrature() const;
    const double current_d2_quadrature() const;
    double Q(const double N) const;
private:
    double _lambda;  // relaxation parameter
    // blitz::Array<double,3> _data;  // member of Grid, storing w^{k}
    arma::uword n;    // length of history, w(i), i = 0, 1, 2, ..., n
    /**
     * Current position index for w and d.
     * To avoid data hard-copy, we update the history vector, w and d, using a
     * mechanism that the newest data replaces the oldest one.
     * The initial storage of w and d arranges as the newer data possesses
     * smaller index:
     *      w(i) -> w^{k-i}, i = 0, 1, ..., n
     * where k is the iteration number.
     * For example, for n = 3, the history looks like
     *      w^{k}, w^{k-1}, w^{k-2}, w^{k-3}
     * By iterate a step, now the iteration number is (k+1), and w(k-3) is
     * no longer needed. Update the above data history to
     *      w^{k}, w^{k-1}, w^{k-2}, w^{k+1}
     * And by stepping further, the iteration number is (k+2), thus w(k-2) is
     * no longer needed. Update the above data history similarly to
     *      w^{k, w^{k-1}, w^{k+2}, w^{k+1}
     * and so on.
     * Obviously, we need an indicator to indicate where our latest data
     * is located. Here, cur_pos serve as such an indicator. Below is how we
     * can obtain the specified w by a measure of how old the data is
     *      w(i) = w^{k-(i-cur_pos)}            for i >= cur_pos
     *      w(i) = w^{k-(i-cur_pos+n+1)}        for i < cur_pos
     * The storage of d is the same as w.
     */
    arma::uword cur_pos;
    /**
     * We use a ramping up algorithm to gradually increase the number of
     * history upto the the maximun number n.
     * k is the number of iterations performed.
     * The actual Anderson mixing history number is
     *      min(k, n)
     */
    arma::uword k;
    arma::field<arma::cube> w;
    /**
     * d^{k} = w_transition^{k} - w^{k}
     * where w_transition is the field calculated from density, for AB diblock
     * copolymer,
     *    w_A,transition = chi*N*phi_B - 0.5 * (w_A + w_B)
     *    w_B,transition = chi*N*phi_A - 0.5 * (w_A + w_B)
     */
    arma::field<arma::cube> d;  // history of w_transition - w_old
    /**
     * A symmetric matrix for storing scalars
     *    B_{ij}^{k} = sum d^{k-i} * d^{k-j}      for i, j = 0, 1, ..., n
     * where sum is a summation over all grid points.
     * To find B_{ij}^{k+1}, we have
     *    B_{ij}^{k+1} = sum d^{k+1-i} * d^{k+1-j}
     *                 = sum d^{k-(i-1)} * d^ {k-(j-1)}
     * Therefore,
     *    B_{00}^{k+1} = sum d^{k+1} * d^{k+1}
     *    B_{i0}^{k+1} = sum d^{k-(i-1)} * d^{k+1}    for i = 1, ..., n
     *    B_{0j}^{k+1} = sum d^{k+1} * d^{k-(j-1)}    for j = 1, ..., n
     *    B_{ij}^{k+1} = sum d^{k-(i-1)} * d^{k-(j-1)}
     *                 = B_{(i-1)(j-1)}^{k}           for i, j = 1, ..., n
     * It tells that most of elements in B_{ij}^{k+1} can be obtained from
     * B_{ij}^{k} directly, thus save most of computational efforts, which
     * reduce complexity from O(n^2*M) to O(nM).
     */
    arma::mat B;

    arma::uword find_index(const arma::uword j) const;

    void update_w0();
    void update_d(const arma::cube &wt);
    void calc_d(const arma::cube &wt);
    void update_w(const arma::cube &W, const arma::cube &D);
    void update_B();
    // C = U^{-1}V
    arma::vec calc_C(const arma::mat &U, const arma::vec &V) const;
    /**
     * W^{k} = w^{k} + sum_{i=1}{n} C_{i-1} * (w^{k-i} - w^{k})
     */
    arma::cube calc_W(const arma::vec &C) const;
    /**
     * D^{k} = d^{k} + sum_{i=1}{n} C_{i-1} * (d^{k-i} - d^{k})
     */
    arma::cube calc_D(const arma::vec &C) const;
};

#endif
