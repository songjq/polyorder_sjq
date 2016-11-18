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

void save_mat(colvec &x, colvec &y, colvec &z, colvec &u1,
              cube &u0, cube &w, cube &u3){
    mwSize Nx = (mwSize) x.n_elem;
    mwSize Ny = (mwSize) y.n_elem;
    mwSize Nz = (mwSize) z.n_elem;
    mwSize Nr = (mwSize) u0.n_rows;
    mwSize Nc = (mwSize) u0.n_cols;
    mwSize Ns = (mwSize) u0.n_slices;
    mwSize n_bytes_x = Nx * sizeof(double);
    mwSize n_bytes_y = Ny * sizeof(double);
    mwSize n_bytes_z = Nz * sizeof(double);
    mwSize n_bytes_3d = Nr * Nc * Ns * sizeof(double);
    mwSize dim3[3] = {Nr, Nc, Ns};
    mwSize dimx[1] = {Nx};
    mwSize dimy[1] = {Ny};
    mwSize dimz[1] = {Nz};

    CMatFile mat;
    mat.matInit("test_etdrk4_3d.mat", "u");
    if(!mat.queryStatus()){
        mat.matPut("x", x.memptr(), n_bytes_x, 1, dimx,
                mxDOUBLE_CLASS, mxREAL);
        mat.matPut("y", y.memptr(), n_bytes_y, 1, dimy,
                mxDOUBLE_CLASS, mxREAL);
        mat.matPut("z", z.memptr(), n_bytes_z, 1, dimz,
                mxDOUBLE_CLASS, mxREAL);
        mat.matPut("u1", u1.memptr(), n_bytes_z, 1, dimz,
                mxDOUBLE_CLASS, mxREAL);
        mat.matPut("u0", u0.memptr(), n_bytes_3d, 3, dim3,
                mxDOUBLE_CLASS, mxREAL);
        mat.matPut("w", w.memptr(), n_bytes_3d, 3, dim3,
                mxDOUBLE_CLASS, mxREAL);
        mat.matPut("u3", u3.memptr(), n_bytes_3d, 3, dim3,
                mxDOUBLE_CLASS, mxREAL);
    }
}

void test(Boundary &lb, Boundary &rb){
    uword N_test = 10;

    uword dim = 3;
    double lx = 2 * PI;
    double ly = 2 * PI;
    double lz = 10.0;
    CrystalSystemType cstype = CrystalSystemType::ORTHORHOMBIC;
    CellParam cp;
    cp.a = lx;
    cp.b = ly;
    cp.c = lz;
    UnitCell uc(dim, cstype, cp);
    uword Lx = 32;
    uword Ly = 32;
    uword Lz = 33;
    uword Ns = 100 + 1;
    double ds = 1.0 / (Ns - 1);
    ConfineType ctype = ConfineType::CUBE;
    //Following lines are default values
    //ETDRK4SCHEME stype = ETDRK4SCHEME::KROGSTAD
    //uword M = 16;
    wall_clock timer;
    timer.tic();
    Etdrk4 et(uc, dim, Lx, Ly, Lz, ds, ctype, lb, rb);
    double n_secs = timer.toc();
    cout<<"Initialization took "<<n_secs<<" seconds."<<endl;

    colvec x = linspace(0, lx, Lx);
    colvec y = linspace(0, ly, Ly);
    Cheb cheb(Lz);
    colvec z = cheb.x();
    z = 0.5 * (z + 1) * lz;
    colvec sech = 1. / cosh(0.75 * (2.0*z - lz));
    colvec w1 = 1.0 - 2 * sech % sech;
    cube w = zeros<cube>(Lx, Ly, Lz);
    for(uword k=0; k<w1.n_elem; k++)
        w.slice(k).fill(w1(k));
    //w.print("w =");
    cube u0 = ones<cube>(Lx, Ly, Lz);
    field<cube> q;

    // q's shape is Lx x Ly x Lz x Ns
    // slice q with q().slice()
    // For N_test = 10, using scheme_krogstad_armafft
    //      runtime 117.795 seconds (Lx=32, Ly=32, Lz=64, Ns=101, RBC-RBC)
    //      runtime 113.947 seconds (Lx=64, Ly=16, Lz=64, Ns=101, RBC-RBC)
    //      runtime 191.875 seconds (Lx=64, Ly=64, Lz=32, Ns=101, RBC-RBC)
    //      runtime 53.9656 seconds (Lx=32, Ly=32, Lz=33, Ns=101, RBC-RBC)
    // For N_test = 10, using scheme_krogstad (fftw optimal version)
    //      runtime 98.7955 seconds (Lx=32, Ly=32, Lz=64, Ns=101, RBC-RBC)
    //      runtime 97.8068 seconds (Lx=64, Ly=16, Lz=64, Ns=101, RBC-RBC)
    //      runtime 155.889 seconds (Lx=64, Ly=64, Lz=32, Ns=101, RBC-RBC)
    //      runtime 45.3166 seconds (Lx=32, Ly=32, Lz=33, Ns=101, RBC-RBC)
    // For Python chebpy, N_test = 100
    //      runtime 131.901 seconds (Lx=32, Ly=32, Ly=33, Ns=101, RBC-RBC)
    // There is 131.901/53.9656 = 2.5x speed up replace Python by C++.
    //      runtime 1047.82 seconds (Lx=32, Ly=32, Ly=63, Ns=101, RBC-RBC)
    // There is 1047.82/117.795 = 9x speed up replace Python by C++.
    timer.tic();
    for(uword i=0; i<N_test; i++)
        et.solve(u0, -w, Ns, q);
    n_secs = timer.toc();
    cout<<N_test<<" runs took "<<n_secs<<" seconds."<<endl;

    // Cross check with 1D ETDRK4
    dim = 1;
    Lx = Lz;
    Ly = 1;
    Lz = 1;
    cp.a = lz;
    UnitCell uc1(dim, cstype, cp);
    Etdrk4 et1(uc1, dim, Lx, Ly, Lz, ds, ctype, lb, rb);
    colvec u01 = ones<colvec>(Lx);
    mat q1;
    et1.solve(u01, -w1, Ns, q1);
    colvec u1 = q1.col(Ns-1);
    cube q3 = q(Ns-1);
    colvec u3 = q3.tube(0, 0);
    colvec diff = abs(u1 - u3);
    cout<<"Max difference between 1D and 3D: "<<diff.max()<<endl;

    /*x.print("x =");
    u0.print("u0 =");
    w.print("w =");
    q.print("q =");*/

    save_mat(x, y, z, u1, u0, w, q(Ns-1));
}

int main(){
    Boundary dbc = Boundary(0, 1, 0);
    Boundary nbc = Boundary(1, 0, 0);
    Boundary rbc = Boundary(1, 1, 0);
    test(nbc, nbc);

    return 0;
}
