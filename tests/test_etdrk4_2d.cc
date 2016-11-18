#include <iostream>
#include "armadillo"
#include "CMatFile.h"
#include "Cheb.h"

#include "Etdrk4.h"
#include "Config.h"
#include "UnitCell.h"
#include "common.h"

using namespace std;
using namespace arma;

void save_mat(colvec &x, colvec &y, mat &u0, mat &w, cube &q){
    mwSize Nx = (mwSize) x.n_elem;
    mwSize Ny = (mwSize) y.n_elem;
    mwSize Nr = (mwSize) u0.n_rows;
    mwSize Nc = (mwSize) u0.n_cols;
    mwSize Ns = (mwSize) q.n_slices;
    mwSize n_bytes_x = Nx * sizeof(double);
    mwSize n_bytes_y = Ny * sizeof(double);
    mwSize n_bytes_2d = Nr * Nc * sizeof(double);
    mwSize n_bytes_3d = Nr * Nc * Ns * sizeof(double);
    mwSize dim3[3] = {Nr, Nc, Ns};
    mwSize dim2[2] = {Nr, Nc};
    mwSize dimx[1] = {Nx};
    mwSize dimy[1] = {Ny};

    CMatFile mat;
    mat.matInit("test_etdrk4_2d.mat", "u");
    if(!mat.queryStatus()){
        mat.matPut("x", x.memptr(), n_bytes_x, 1, dimx,
                mxDOUBLE_CLASS, mxREAL);
        mat.matPut("y", y.memptr(), n_bytes_y, 1, dimy,
                mxDOUBLE_CLASS, mxREAL);
        mat.matPut("u0", u0.memptr(), n_bytes_2d, 2, dim2,
                mxDOUBLE_CLASS, mxREAL);
        mat.matPut("w", w.memptr(), n_bytes_2d, 2, dim2,
                mxDOUBLE_CLASS, mxREAL);
        mat.matPut("q", q.memptr(), n_bytes_3d, 3, dim3,
                mxDOUBLE_CLASS, mxREAL);
    }
}

void test(Boundary &lb, Boundary &rb){
    uword N_test = 100;

    uword dim = 2;
    double lx = 2 * PI;
    double ly = 10.0;
    CrystalSystemType cstype = CrystalSystemType::RECTANGULAR;
    CellParam cp;
    cp.a = lx;
    cp.b = ly;
    UnitCell uc(dim, cstype, cp);
    uword Lx = 64;
    uword Ly = 128;
    uword Lz = 1;
    uword Ns = 100 + 1;
    double ds = 1.0 / (Ns - 1);
    ConfineType ctype = ConfineType::CUBE;
    //Following lines are default values
    //ETDRK4SCHEME stype = ETDRK4SCHEME::KROGSTAD
    //uword M = 16;
    Etdrk4 et(uc, dim, Lx, Ly, Lz, ds, ctype, lb, rb);

    colvec x = linspace(0, lx, Lx);
    Cheb cheb(Ly);
    colvec y = cheb.x();
    y = 0.5 * (y + 1) * ly;
    colvec sech = 1. / cosh(0.75 * (2.0*y - ly));
    colvec w1 = 1.0 - 2 * sech % sech;
    mat w = zeros<mat>(Ly, Lx);
    w.each_col() = w1;
    mat u0 = ones<mat>(Ly, Lx);
    cube q;

    // q's shape is Ly x Lx x Ns
    // slice q with q.slice()
    // For N_test = 100
    //      runtime 272.971 seconds (Lx=64, Ly=128, Ns=101, RBC-DBC)
    //      runtime 229.408 seconds (Lx=64, Ly=128, Ns=101, RBC-RBC)
    //      runtime 227.302 seconds (Lx=64, Ly=128, Ns=101, NBC-NBC)
    // For Python chebpy, N_test = 100
    //      runtime 1117.6 seconds (Lx=64, Ly=128, Ns=101, NBC-NBC)
    // There is 1117.6/227.302 = 5 x speed up replace Python by C++.
    wall_clock timer;
    timer.tic();
    for(uword i=0; i<N_test; i++)
        et.solve(u0, -w, Ns, q);
    double n_secs = timer.toc();
    cout<<N_test<<" runs took "<<n_secs<<" seconds."<<endl;

    // Cross check with 1D ETDRK4
    dim = 1;
    Lx = 128;
    Ly = 1;
    Lz = 1;
    cp.a = ly;
    UnitCell uc1(dim, cstype, cp);
    Etdrk4 et1(uc1, dim, Lx, Ly, Lz, ds, ctype, lb, rb);
    colvec u01 = ones<colvec>(Lx);
    mat q1;
    et1.solve(u01, -w1, Ns, q1);
    colvec u1 = q1.col(Ns-1);
    //u1.print("u1 =");
    mat q2 = q.slice(Ns-1);
    colvec u2 = q2.col(0);
    //u2.print("u2 =");
    colvec diff = abs(u1 - u2);
    cout<<"Max difference between 1D and 2D: "<<diff.max()<<endl;

    /*x.print("x =");
    u0.print("u0 =");
    w.print("w =");
    q.print("q =");*/

    save_mat(x, y, u0, w, q);
}

int main(){
    Boundary dbc = Boundary(0, 1, 0);
    Boundary nbc = Boundary(1, 0, 0);
    Boundary rbc = Boundary(1, 1, 0);
    test(nbc, nbc);

    return 0;
}
