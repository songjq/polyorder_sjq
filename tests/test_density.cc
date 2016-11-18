/**
 * Test Grid, Field, FieldE, Yita, Density, DensitySmall, Propagator
 *
 * Copyright@ Yi-Xin Liu, 2012
 *
 */

#include <iostream>
#include <string>
#include <ctime>
#include <cmath>

#include "armadillo"

#include "UnitCell.h"
#include "Field.h"
#include "DensitySmall.h"
#include "Density.h"
#include "Simpson.h"

using namespace std;
using namespace arma;

void test_density(const string config){
    Config cfg(config.c_str());

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

    cout<<"sA, sB = "<<sA<<", "<<sB<<endl;
    cout<<"dsA, dsB ="<<dsA<<", "<<dsB<<endl;
    cout<<endl;

    Density phiA("phiA", cfg, fA);
    Density phiB("phiB", cfg, fB);
    cout<<"phiA.name = (phiA) "<<phiA.name()<<endl;
    cout<<"fA = (0.36) "<<phiA.mean()<<endl;
    cout<<"phiA.cc = (1.0) "<<phiA.cc()<<endl;
    cout<<"phiA.dim = (3) "<<phiA.dim()<<endl;
    cout<<"phiA.(Lx,Ly,Lz) = (32, 32, 64) ";
    cout<<"("<<phiA.Lx()<<","<<phiA.Ly()<<","<<phiA.Lz()<<")"<<endl;
    cout<<endl;
    cout<<"phiB.name = (phiB) "<<phiB.name()<<endl;
    cout<<"fB = (0.64) "<<phiB.mean()<<endl;
    cout<<"phiB.cc = (1.0) "<<phiB.cc()<<endl;
    cout<<"phiB.dim = (3) "<<phiB.dim()<<endl;
    cout<<"phiB.(Lx,Ly,Lz) = (32, 32, 64) ";
    cout<<"("<<phiB.Lx()<<","<<phiB.Ly()<<","<<phiB.Lz()<<")"<<endl;
    cout<<endl;

    UnitCell uc(cfg);
    Grid zero1(uc, Lx, Ly, Lz, 0.1);
    Grid zero2(uc, Lx, Ly, Lz, 0.3);
    Grid one(uc, Lx, Ly, Lz, 1.0);
    Propagator qA("qA", cfg, sA, dsA, one);
    Propagator qAc("qAc", cfg, sA, dsA);
    Propagator qB("qB", cfg, sB, dsB);
    Propagator qBc("qBc", cfg, sB, dsB, one);
    qA.update(zero1);
    qB.set_head(qA.get_tail());
    qB.update(zero2);
    qBc.update(zero2);
    qAc.set_head(qBc.get_tail());
    qAc.update(zero1);

    cout<<"Internal integrating algorithm (trapezoidal rule): "<<endl;
    phiA.update(qA, qAc);
    phiB.update(qB, qBc);
    cout<<"phiA(0,1,1) = () "<<phiA(0, 1, 1)<<endl;
    cout<<"phiA.mean = () "<<phiA.mean()<<endl;
    cout<<"phiB(0,1,1) = () "<<phiB(0, 1, 1)<<endl;
    cout<<"phiB.mean = () "<<phiB.mean()<<endl;
    cout<<endl;

    cout<<"Simpson's rule: "<<endl;
    phiA.update(qA, qAc, new Simpson);
    phiB.update(qB, qBc, new Simpson);
    cout<<"phiA(0,1,1) = () "<<phiA(0, 1, 1)<<endl;
    cout<<"phiA.mean = () "<<phiA.mean()<<endl;
    cout<<"phiB(0,1,1) = () "<<phiB(0, 1, 1)<<endl;
    cout<<"phiB.mean = () "<<phiB.mean()<<endl;
    cout<<endl;
}

void test_density_small(const string config){
    Config cfg(config.c_str());

    DensitySmall phiS("phiS", cfg, 0.2);
    DensitySmall phiP("phiP", cfg, 0.02);
    cout<<"phiS.name = (phiS) "<<phiS.name()<<endl;
    cout<<"phiS.cc = (0.2) "<<phiS.cc()<<endl;
    cout<<"phiS.dim = (3) "<<phiS.dim()<<endl;
    cout<<"phiS.(Lx,Ly,Lz) = (32, 32, 64) ";
    cout<<"("<<phiS.Lx()<<","<<phiS.Ly()<<","<<phiS.Lz()<<")"<<endl;
    cout<<endl;

    Field wS("wS", cfg, 1.0, 0.1);
    Field wP("wP", cfg, 0.1, -0.05, 0.05, 1234);
    phiS.update(wS, 0.1);
    phiP.update(wP, 0.02);
    cout<<"phiS(0,1,1) = () "<<phiS(0, 1, 1)<<endl;
    cout<<"phiS.mean = () "<<phiS.mean()<<endl;
    cout<<"phiP(0,1,1) = () "<<phiP(0, 1, 1)<<endl;
    cout<<"phiP.mean = () "<<phiP.mean()<<endl;
    cout<<endl;
}

int main(){
    string config("param.ini.AB");
    test_density_small(config);
    test_density(config);

    return 0;
}
