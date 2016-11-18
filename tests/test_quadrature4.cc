/**
 * Test Quadrature4: Quad4_Closed
 *
 * Copyright@ 2014, Yi-Xin Liu
 *
 */

#include <iostream>
#include <string>
#include <ctime>
#include <cmath>

#include "blitz/array.h"
#include "armadillo"

#include "Propagator.h"
#include "Simpson.h"
#include "Quadrature4.h"

using namespace std;
using namespace arma;

void test_quad4_closed(){
    Config cfg("test_param.ini");
    cout << "Enviroment configured successfully." << endl;
    uword Lx = cfg.Lx();
    uword Ly = cfg.Ly();
    uword Lz = cfg.Lz();
    //uvec Ms = cfg.Ms();
    vec ds = cfg.ds();
    vec f = cfg.f();
    double dsA = ds(0);
    double dsB = ds(1);
    double fA = f(0);
    double fB = f(1);
    uword LA = fA / dsA;
    uword sA = LA + 1;
    uword LB = fB / dsB;
    uword sB = LB + 1;
    fA = 1.0 * LA / (LA + LB);
    fB = 1.0 * LB / (LA + LB);

    cout<<"Lx, Ly, Lz ="<<Lx<<","<<Ly<<","<<Lz<<endl;
    cout<<"sA, sB = "<<sA<<", "<<sB<<endl;
    cout<<"dsA, dsB ="<<dsA<<", "<<dsB<<endl;
    cout<<endl;

    blitz::Array<double,3> data_simp(Lx, Ly, Lz);
    blitz::Array<double,3> data_quad4c(Lx, Ly, Lz);
    blitz::Array<double,3> data_quad4o(Lx, Ly, Lz);
    blitz::Array<double,3> data_quad4sol(Lx, Ly, Lz);
    blitz::Array<double,3> data_quad4sor(Lx, Ly, Lz);
    blitz::Array<double,3> one(Lx, Ly, Lz);
    one = 1.0;
    blitz::Array<double,3> g(Lx, Ly, Lz);
    Propagator qA("qA", cfg, sA, dsA);
    Propagator qAc("qAc", cfg, sA, dsA);
    Propagator qB("qB", cfg, sB, dsB);
    Propagator qBc("qBc", cfg, sB, dsB);

    blitz::Range all = blitz::Range::all();
    blitz::Array<double,4> tqA(qA.qs());
    blitz::Array<double,4> tqAc(qAc.qs());
    for(int s=0; s<sA; s++){
        tqA(s, all, all, all) = one;
        // g = (s*dsA) * (s*dsA);
        g = exp(s*dsA);
        tqAc(s, all, all, all) = g;
    }

    // For g = exp(s*dsA);
    double exact = exp(fA) - 1.0;
    // For g = (s*dsA) * (s*dsA);
    //double exact = fA*fA*fA/3.0;
    cout<<"Exact: "<<exact<<endl;

    Simpson simp;
    simp.solve(data_simp, qA, qAc, 1.0);
    cout<<"max(|Simpson - Exact|) = ";
    cout<<blitz::max(blitz::abs(data_simp - exact))<<endl;

    Quad4_Closed quad4c;
    quad4c.solve(data_quad4c, qA, qAc, 1.0);
    cout<<"max(|Quad4_Closed - Exact|) = ";
    cout<<blitz::max(blitz::abs(data_quad4c - exact))<<endl;

    Quad4_Open quad4o;
    quad4o.solve(data_quad4o, qA, qAc, 1.0);
    cout<<"max(|Quad4_Open - Exact|) = ";
    cout<<blitz::max(blitz::abs(data_quad4o - exact))<<endl;

    Quad4_Semiopen_Left quad4sol;
    quad4sol.solve(data_quad4sol, qA, qAc, 1.0);
    cout<<"max(|Quad4_Semiopen_Left - Exact|) = ";
    cout<<blitz::max(blitz::abs(data_quad4sol - exact))<<endl;

    Quad4_Semiopen_Right quad4sor;
    quad4sor.solve(data_quad4sor, qA, qAc, 1.0);
    cout<<"max(|Quad4_Semiopen_Right - Exact|) = ";
    cout<<blitz::max(blitz::abs(data_quad4sor - exact))<<endl;
}

int main(){
    test_quad4_closed();

    return 0;
}
