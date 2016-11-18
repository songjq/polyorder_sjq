#include <blitz/array.h>
#include "armadillo"

#include "Grid.h"
#include "FieldAX.h"

#include <cmath>  // for pow

using std::string;

FieldAX::FieldAX(const string name, const Config &cfg, const double lam,
                 arma::uword n_history):
                    Grid(name, cfg), _lambda(lam),
                    n(n_history), cur_pos(n), w(n+1), d(n+1), B(n+1,n+1), k(0){
    update_w0();
}

FieldAX::FieldAX(const string name, const Config &cfg, const double val,
                 const double lam, arma::uword n_history):
                    Grid(name, cfg, val), _lambda(lam),
                    n(n_history), cur_pos(n), w(n+1), d(n+1), B(n+1,n+1), k(0){
    update_w0();
}

FieldAX::FieldAX(const string name, const Config &cfg,
                 const blitz::Array<double,3> data,
                 const double lam,
                 arma::uword n_history):
                    Grid(name, cfg, data), _lambda(lam),
                    n(n_history), cur_pos(n), w(n+1), d(n+1), B(n+1,n+1), k(0){
    update_w0();
}

FieldAX::FieldAX(const string name, const Config &cfg, const string file,
                 const double lam, arma::uword n_history):
                    Grid(name, cfg, file), _lambda(lam),
                    n(n_history), cur_pos(n), w(n+1), d(n+1), B(n+1,n+1), k(0){
    update_w0();
}

FieldAX::FieldAX(const string name, const Config &cfg, const double lam,
                 arma::uword n_history,
                 const double low, const double high,
                 const size_t seed /* seed = 0 */):
                    Grid(name, cfg, low, high, seed), _lambda(lam),
                    n(n_history), cur_pos(n), w(n+1), d(n+1), B(n+1,n+1), k(0){
    update_w0();
}

FieldAX& FieldAX::operator= (const Grid &rhs){
    Grid::operator= (rhs);
    return *this;
}

FieldAX& FieldAX::operator= (const FieldAX &rhs){
    Grid::operator= (rhs);
    _lambda = rhs._lambda;
    n = rhs.n;
    w = rhs.w;
    d = rhs.d;
    B = rhs.B;
    return *this;
}

void FieldAX::set_lambda(const double lam) {
    _lambda = lam;
}

const double FieldAX::lambda() const{
    return _lambda;
}

const arma::uword FieldAX::n_iteration() const{
    return k;
}

const arma::uword FieldAX::n_history() const{
    return n;
}
const arma::uword FieldAX::current_index() const{
    return cur_pos;
}

const arma::cube FieldAX::current_w() const{
    return w(cur_pos);
}

const double FieldAX::current_w_quadrature() const{
    arma::cube cur_w = w(cur_pos);
    blitz::Array<double, 3> b_col(cur_w.memptr(),
                                  blitz::shape(_Lx, _Ly, _Lz),
                                  blitz::neverDeleteData,
                                  blitz::ColumnMajorArray<3>());
    return quadrature(b_col);  // Grid::quadrature
}

const double FieldAX::current_w_abs_quadrature() const{
    arma::cube cur_w = arma::abs(w(cur_pos));
    blitz::Array<double, 3> b_col(cur_w.memptr(),
                                  blitz::shape(_Lx, _Ly, _Lz),
                                  blitz::neverDeleteData,
                                  blitz::ColumnMajorArray<3>());
    return quadrature(b_col);  // Grid::quadrature
}

const double FieldAX::current_w2_quadrature() const{
    arma::uword pos = cur_pos + 1;
    if(pos > n)
        pos = 0;
    arma::cube w2 = w(pos) % w(pos);
    blitz::Array<double, 3> b_col(w2.memptr(),
                                  blitz::shape(_Lx, _Ly, _Lz),
                                  blitz::neverDeleteData,
                                  blitz::ColumnMajorArray<3>());
    return quadrature(b_col);  // Grid::quadrature
}

const arma::cube FieldAX::current_d() const{
    return d(cur_pos);
}

const double FieldAX::current_d_quadrature() const{
    arma::cube cur_d = d(cur_pos);
    blitz::Array<double, 3> b_col(cur_d.memptr(),
                                  blitz::shape(_Lx, _Ly, _Lz),
                                  blitz::neverDeleteData,
                                  blitz::ColumnMajorArray<3>());
    return quadrature(b_col);  // Grid::quadrature
}

const double FieldAX::current_d_abs_quadrature() const{
    arma::cube cur_d = arma::abs(d(cur_pos));
    blitz::Array<double, 3> b_col(cur_d.memptr(),
                                  blitz::shape(_Lx, _Ly, _Lz),
                                  blitz::neverDeleteData,
                                  blitz::ColumnMajorArray<3>());
    return quadrature(b_col);  // Grid::quadrature
}

const double FieldAX::current_d2_quadrature() const{
    arma::uword pos = cur_pos + 1;
    if(pos > n)
        pos = 0;
    arma::cube d2 = d(pos) % d(pos);
    blitz::Array<double, 3> b_col(d2.memptr(),
                                  blitz::shape(_Lx, _Ly, _Lz),
                                  blitz::neverDeleteData,
                                  blitz::ColumnMajorArray<3>());
    return quadrature(b_col);  // Grid::quadrature
}

void FieldAX::update(const arma::cube &wt){
    //w.print("w =");
    update_d(wt);
    //d.print("d =");
    update_B();
    //B.print("B =");

    arma::mat U = calc_U();
    //U.print("U =");
    arma::vec V = calc_V();
    //V.print("V =");
    arma::vec C = calc_C(U, V);
    //C.print("C =");
    arma::cube W = calc_W(C);
    arma::cube D = calc_D(C);

    /** VERY IMPORTANT
     * We update the cur_pos before updating w because the very first w
     * is updated during initialization of FieldAX instance.
     * Note 1: this update should be placed after update_d.
     * Note 2: Since cur_pos is an uword (unsigned integer), if cur_pos = 0
     *         and perform cur_pos--, it will return a very large positive
     *         integer.
     * This should be done after calc_W but before update_w.
     */
    if(cur_pos > 0)
        cur_pos--;
    else
        cur_pos = n;  // cur_pos = 0

    update_w(W, D);

    /** VERY IMPORTANT **/
    k++;  // increase the interation number after updating.
}

void FieldAX::pre_update(const Grid &u){
    blitz::Array<double, 3> b_col(_Lx, _Ly, _Lz,
                                  blitz::ColumnMajorArray<3>());
    b_col = u.data();
    arma::cube wt(b_col.data(), _Lx, _Ly, _Lz);

    //w.print("w =");
    update_d(wt);
    //d.print("d =");
    update_B();
    //B.print("B =");
}

void FieldAX::update(const arma::vec &C){
    arma::cube W = calc_W(C);
    arma::cube D = calc_D(C);

    /** VERY IMPORTANT
     * We update the cur_pos before updating w because the very first w
     * is updated during initialization of FieldAX instance.
     * Note 1: this update should be placed after update_d.
     * Note 2: Since cur_pos is an uword (unsigned integer), if cur_pos = 0
     *         and perform cur_pos--, it will return a very large positive
     *         integer.
     * This should be done after calc_W but before update_w.
     */
    if(cur_pos > 0)
        cur_pos--;
    else
        cur_pos = n;  // cur_pos = 0

    update_w(W, D);

    /** VERY IMPORTANT **/
    k++;  // increase the interation number after updating.
}

/**
 * TEST PASSED, 2014.06.25.
 */
void FieldAX::update(const Grid &u){
    blitz::Array<double, 3> b_col(_Lx, _Ly, _Lz,
                                  blitz::ColumnMajorArray<3>());
    b_col = u.data();
    arma::cube wt(b_col.data(), _Lx, _Ly, _Lz);

    update(wt);
}

void FieldAX::update(const blitz::Array<double,3> bu){
    blitz::Array<double, 3> b_col(_Lx, _Ly, _Lz,
                                  blitz::ColumnMajorArray<3>());
    b_col = bu;
    arma::cube wt(b_col.data(), _Lx, _Ly, _Lz);

    update(wt);
}

double FieldAX::Q(const double N) const{
    return blitz::mean(blitz::exp(-_data / N));
}




/***********************************************************************
 *                        Private member functions                     *
 ***********************************************************************/

/**
 * Find the index for d^{k-j} or w^{k-j} in the d and w array:
 *          w(i) = w^{k-j}
 *          d(i) = d^{k-j}
 */
arma::uword FieldAX::find_index(const arma::uword j) const{
    arma::uword i = j + cur_pos;
    if(i > n)
        i -= (n+1);
    return i;
}

/**
 * In current version, this function is only used for updating
 * the initial field w^{0} => w(n), where n is FieldAX::n,
 * see header file for details of n.
 */
void FieldAX::update_w0(){
    blitz::Array<double, 3> b_col(_Lx, _Ly, _Lz,
                                  blitz::ColumnMajorArray<3>());
    b_col = _data;
    arma::cube w0(b_col.data(), _Lx, _Ly, _Lz);
    w(n) = w0;
}

/**
 * Update fields history w.
 * Store newest field data in w(cur_pos).
 */
void FieldAX::update_w(const arma::cube &W, const arma::cube &D){
    double lam = 1.0 - pow(_lambda, k+1);
    //cout<<"lam = "<<lam<<endl;
    /*
    if(k < n)
        lam = 0.1;
    else
        lam = 1.0;
    */
    arma::cube ww = W + lam * D;
    //ww.print("w(cur_pos) =");
    w(cur_pos) = ww;

    // transfer back to field's blitz current data
    blitz::Array<double, 3> b_col(ww.memptr(),
                                  blitz::shape(_Lx, _Ly, _Lz),
                                  blitz::neverDeleteData,
                                  blitz::ColumnMajorArray<3>());
    _data = b_col;
}

/**
 * d^{0} = w_transition^{0} - w^{0}
 * Therefore, d shares the same cur_pos with w.
 */
void FieldAX::update_d(const arma::cube &wt){
    d(cur_pos) = wt - w(cur_pos);
}

/**
 * When k < n, all index i, j > k is not relavent.
 * To save a copy of original matrix:
 * 1. We first copy original data and then calculate the new data.
 * 2. When copying original data, do copy in backward manner.
 */
void FieldAX::update_B(){
    // Update B^{k+1} by copying data from B^{k}
    // to save most of computatoinal cost.
    // Reduce Complexity from O(n^2*M) to O(nM).
    for(arma::uword i=n; i>=1; i--)
        for(arma::uword j=n; j>=i; j--)
            B(i, j) = B(i-1, j-1);

    // Update new data
    arma::uword nn = n;
    if(k < n)
        nn = k;
    for(arma::uword j=0; j<=nn; j++){
        arma::uword kk = find_index(j);
        arma::cube dd = d(cur_pos) % d(kk);
        blitz::Array<double, 3> b_col(dd.memptr(),
                                      blitz::shape(_Lx, _Ly, _Lz),
                                      blitz::neverDeleteData,
                                      blitz::ColumnMajorArray<3>());
        B(0, j) = quadrature(b_col);  // Grid::quadrature
    }

    // Construct symmetric matrix from upper triangle.
    B = arma::symmatu(B);
}

/**
 * U_{ij} = B_{00} - B{(i+1)0} - B{0(j+1)} + B_{(i+1)(j+1)}
 * for i, j = 0, 1, ..., nn-1
 * where nn = min(k, n).
 */
arma::mat FieldAX::calc_U() const{
    arma::uword nn = n;
    if(k < n)
        nn = k;
    arma::mat U(nn, nn);

    //Note: when nn = k = 0, no cycle is needed, and empty U will be returned.
    for(arma::uword i=0; i<nn; i++)
        for(arma::uword j=0; j<nn; j++)
            U(i, j) = B(0, 0) - B(i+1, 0) - B(0, j+1) + B(i+1, j+1);

    return U;
}

/**
 * V_{i} = B_{00} - B_{(i+1)0}    for i = 0, 1, ..., nn-1
 * where nn = min(k, n).
 */
arma::vec FieldAX::calc_V() const{
    arma::uword nn = n;
    if(k < n)
        nn = k;
    arma::vec V(nn);

    //Note: when nn = k = 0, no cycle is needed, and empty V will be returned.
    for(arma::uword i=0; i<nn; i++)
        V(i) = B(0, 0) - B(i+1, 0);

    return V;
}

/**
 * C = U^{-1}V
 */
arma::vec FieldAX::calc_C(const arma::mat &U, const arma::vec &V) const{
    if(U.is_empty() || V.is_empty())
        return arma::vec();
    return arma::solve(U, V);
}

/**
 * W^{k} = w^{k} + sum_{i=1}{n} C_{i-1} * (w^{k-i} - w^{k})
 */
arma::cube FieldAX::calc_W(const arma::vec &C) const{
    arma::cube W = w(cur_pos);
    //W.print("W, w(cur_pos) =");
    //C.print("C =");
    // Note: C.n_elem = n, therefore we use <=.
    // No calculation is performed for empty C (C.n_elem=0).
    for(arma::uword i=1; i<=C.n_elem; i++){
        arma::uword kk = find_index(i);
        W += C(i-1) * (w(kk) - w(cur_pos));
        //cout<<"i = "<<i<<" kk ="<<kk<<endl;
        //w(kk).print("w(kk) =");
        //cout<<"C(i-1) = "<<C(i-1)<<endl;
        //W.print("W +=");
    }
    return W;
}

/**
 * D^{k} = d^{k} + sum_{i=1}{n} C_{i-1} * (d^{k-i} - d^{k})
 */
arma::cube FieldAX::calc_D(const arma::vec &C) const{
    arma::cube D = d(cur_pos);
    //D.print("D, d(cur_pos) =");
    // Note: C.n_elem = n, therefore we use <=.
    // No calculation is performed for empty C (C.n_elem=0).
    for(arma::uword i=1; i<=C.n_elem; i++){
        arma::uword kk = find_index(i);
        D += C(i-1) * (d(kk) - d(cur_pos));
        //d(kk).print("d(kk) =");
        //D.print("D +=");
    }
    return D;
}
