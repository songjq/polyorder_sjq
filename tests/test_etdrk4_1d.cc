#include <iostream>
#include "armadillo"
#include "CMatFile.h"
#include "Cheb.h"

#include "Etdrk4.h"
#include "Config.h"
#include "UnitCell.h"
#include "Propagator.h"
#include "Field.h"

#include "blitz/array.h"

using namespace std;
using namespace arma;

void save_mat(colvec &x, colvec &u0, colvec &w, mat &q){
    mwSize N = (mwSize) u0.n_elem;
    mwSize Nc = (mwSize) q.n_cols;
    mwSize Nr = (mwSize) q.n_rows;
    mwSize n_bytes_1d = N * sizeof(double);
    mwSize n_bytes_2d = Nr * Nc * sizeof(double);
    mwSize dim1[1] = {N};
    mwSize dim2[2] = {Nr, Nc};

    CMatFile mat;
    mat.matInit("test_etdrk4_1d.mat", "u");
    if(!mat.queryStatus()){
        mat.matPut("x", x.memptr(), n_bytes_1d, 1, dim1,
                mxDOUBLE_CLASS, mxREAL);
        mat.matPut("u0", u0.memptr(), n_bytes_1d, 1, dim1,
                mxDOUBLE_CLASS, mxREAL);
        mat.matPut("w", w.memptr(), n_bytes_1d, 1, dim1,
                mxDOUBLE_CLASS, mxREAL);
        mat.matPut("q", q.memptr(), n_bytes_2d, 2, dim2,
                mxDOUBLE_CLASS, mxREAL);
    }
}

void test(Boundary &lb, Boundary &rb){
    uword N_test = 1;

    uword dim = 1;
    double lx = 10.0;
    Config cfg("param.ini");
    UnitCell uc(cfg);

    uword Lx = 64;
    uword Ly = 1;
    uword Lz = 1;

    uword Ns = 100 + 1;
    double ds = 1.0 / (Ns - 1);
    ConfineType ctype = ConfineType::CUBE;
    //Following lines are default values
    //ETDRK4SCHEME stype = ETDRK4SCHEME::KROGSTAD
    //uword M = 16;
    //Etdrk4 et(uc, dim, Lx, Ly, Lz, ds, ctype, lb, rb);
    Etdrk4 et(cfg, ds);

    Cheb cheb(Lx);
    colvec x = cheb.x();
    x = 0.5 * (x + 1) * lx;
    colvec sech = 1. / cosh(0.75 * (2.0*x - lx));
    colvec w = 1.0 - 2 * sech % sech;
    colvec u0 = ones<colvec>(Lx);
    mat q;

    // q's shape is Lx x Ns
    // slice q with q.cols()
    // For N_test = 10000, runtime 44.0768 seconds
    wall_clock timer;
    timer.tic();
    for(uword i=0; i<N_test; i++)
        et.solve(u0, -w, Ns, q);
    double n_secs = timer.toc();
    cout<<"colvec took "<<n_secs<<" seconds"<<endl;

    // q's shape is Ns x Lx
    // slice q with q.rows()
    // For N_test = 10000, runtime 44.8272 seconds
    /*
    rowvec cu0 = u0.t();
    rowvec cw = w.t();
    timer.tic();
    for(uword i=0; i<N_test; i++)
        et.solve(cu0, -cw, Ns, q);
    n_secs = timer.toc();
    cout<<"rowvec took "<<n_secs<<" seconds"<<endl;
    */

    //x.print("x =");
    //u0.print("u0 =");
    //w.print("w =");
    //q.col(Ns-1).print("q[Ns] =");

    //save_mat(x, u0, w, q);
}

void test_propagator(){
    uword N_test = 1;

    Config cfg("param.ini");
    UnitCell uc(cfg);
    uword Lx = cfg.Lx();
    uword Ly = cfg.Ly();
    uword Lz = cfg.Lz();
    double lx = uc.a();
    cout<<"lx = "<<lx<<endl;

    uword Ns = 100 + 1;
    double ds = 1.0 / (Ns - 1);
    //Following lines are default values
    //ETDRK4SCHEME stype = ETDRK4SCHEME::KROGSTAD
    //uword M = 16;
    Etdrk4 et(cfg, ds);

    Cheb cheb(Lx);
    colvec x = cheb.x();
    x = 0.5 * (x + 1) * lx;
    colvec sech = 1. / cosh(0.75 * (2.0*x - lx));
    colvec w = 1.0 - 2 * sech % sech;
    w.print("w =");

    Grid gw("wA", cfg);
    cout<<"wA.dim = "<<gw.dim()<<endl;
    cout<<"wA.Lx = "<<gw.Lx()<<endl;
    cout<<"wA.Ly = "<<gw.Ly()<<endl;
    cout<<"wA.Lz = "<<gw.Lz()<<endl;
    for(uword i=0; i<Lx; i++)
        gw(i) = w(i);
    cout<<"gw = "<<gw.data()<<endl;

    Grid one(uc, Lx, Ly, Lz, 1.0);
    Propagator pq("qA", cfg, Ns, ds, one, &et);
    cout<<"pq.len() = "<<pq.len()<<endl;
    cout<<"pq[0] = "<<pq[0]<<endl;

    pq.update(gw);
    cout<<"pq[-1] = "<<pq[Ns-1]<<endl;
}

int main(){
    Boundary dbc = Boundary(0, 1, 0);
    Boundary nbc = Boundary(1, 0, 0);
    Boundary rbc = Boundary(1, 1, 0);
    //test(rbc, rbc);
    test_propagator();

    return 0;
}
