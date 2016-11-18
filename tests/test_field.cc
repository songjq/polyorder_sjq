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
#include "Yita.h"

using namespace std;
using namespace arma;

void test_yita(const string config){
    Config cfg(config.c_str());

    vec lam = cfg.lam();
    double lamYita = lam(2);
    cout<<"lamYita = (5) "<<lamYita<<endl;
    Yita yita("yita", cfg, 0.12, lamYita);
    Field wA("wA", cfg, 0.33, 0.01);
    Field wB("wB", cfg, 0.66, 0.01);
    cout<<"yita.lambda = (5) "<<yita.lambda()<<endl;
    cout<<"yita.name = (yita) "<<yita.name()<<endl;
    cout<<"yita.dim = (3) "<<yita.dim()<<endl;
    cout<<"yita.Lx = (32) "<<yita.Lx()<<endl;
    cout<<"yita.Ly = (32) "<<yita.Ly()<<endl;
    cout<<"yita.Lz = (64) "<<yita.Lz()<<endl;
    cout<<"yita(0,1,1) = (0.12) "<<yita(0, 1, 1)<<endl;
    cout<<"yita.mean = (0.12) "<<yita.mean()<<endl;
    yita.update(wA + wB - 1);
    cout<<"yita.mean = (0.07) "<<yita.mean()<<endl;
    cout<<endl;

    // copy constructor
    Yita y(yita);
    cout<<"y.lambda = (yita.lambda) "<<y.lambda()<<endl;
    cout<<"y.name = () "<<y.name()<<endl;
    cout<<"y.dim = (3) "<<y.dim()<<endl;
    cout<<"y.Lx = (32) "<<y.Lx()<<endl;
    cout<<"y.Ly = (32) "<<y.Ly()<<endl;
    cout<<"y.Lz = (64) "<<y.Lz()<<endl;
    cout<<"y(0,1,1) = (yita.mean) "<<y(0, 1, 1)<<endl;
    cout<<"y.mean = (yita.mean) "<<y.mean()<<endl;
    cout<<endl;
}

void test_field(const string config){
    Config cfg(config.c_str());

    vec lam = cfg.lam();
    double lamA = lam(0);
    double lamB = lam(1);
    cout<<"lamA = (0.01) "<<lamA<<endl;
    cout<<"lamB = (0.01) "<<lamB<<endl;
    Field wA("wA", cfg, 3, lamA);
    Field wB("wB", cfg, 5, lamB);
    Field wC("wC", cfg, 10, 0.01);
    cout<<"wA.lambda = (0.01) "<<wA.lambda()<<endl;
    cout<<"wA.name = (wA) "<<wA.name()<<endl;
    cout<<"wA.dim = (3) "<<wA.dim()<<endl;
    cout<<"wA.Lx = (32) "<<wA.Lx()<<endl;
    cout<<"wA.Ly = (32) "<<wA.Ly()<<endl;
    cout<<"wA.Lz = (64) "<<wA.Lz()<<endl;
    cout<<"wA(0,1,1) = (3) "<<wA(0, 1, 1)<<endl;
    cout<<"wA.mean = (3) "<<wA.mean()<<endl;
    wA.update(wB);
    cout<<"wA.mean = (3.02) "<<wA.mean()<<endl;
    wA.update(2 * wB * wC);
    cout<<"wA.mean = (3.9898) "<<wA.mean()<<endl;
    cout<<endl;

    // copy constructor
    // test passed
    Field wD(wA);
    cout<<"wD.lambda = (lamA) "<<wD.lambda()<<endl;
    cout<<"wD.name = () "<<wD.name()<<endl;
    cout<<"wD.dim = (3) "<<wD.dim()<<endl;
    cout<<"wD.Lx = (32) "<<wD.Lx()<<endl;
    cout<<"wD.Ly = (32) "<<wD.Ly()<<endl;
    cout<<"wD.Lz = (64) "<<wD.Lz()<<endl;
    cout<<"wD(0,1,1) = (3.9898) "<<wD(0, 1, 1)<<endl;
    cout<<"wD.mean = (3.9898) "<<wD.mean()<<endl;
    cout<<endl;

    // Assignment operator
    // test passed
    Field wE;
    wE=wD;
    cout<<"wE.lambda = (wD.lambda) "<<wE.lambda()<<endl;
    cout<<"wE.name = () "<<wE.name()<<endl;
    cout<<"wE.dim = (3) "<<wE.dim()<<endl;
    cout<<"wE(Lx,Ly,Lz) = (32, 32, 64) ";
    cout<<"("<<wE.Lx()<<","<<wE.Ly()<<","<<wE.Lz()<<")"<<endl;
    cout<<"wE(0,1,1) = (3.9898) "<<wE(0, 1, 1)<<endl;
    cout<<"wE.mean = (3.9898) "<<wE.mean()<<endl;
    cout<<endl;

    // Arithmetic operators
    // test passed
    Field wF;
    Yita yita("yita", cfg, 0.12, 5);
    wF = wB*wB + 2*wC - wC*wC + yita;
    cout<<"wF.lambda = () "<<wF.lambda()<<endl;
    cout<<"wF.name = () "<<wF.name()<<endl;
    cout<<"wF.dim = (3) "<<wF.dim()<<endl;
    cout<<"wF.(Lx,Ly,Lz) = (32, 32, 64) ";
    cout<<"("<<wF.Lx()<<","<<wF.Ly()<<","<<wF.Lz()<<")"<<endl;
    cout<<"wF(0,1,1) = (-54.88) "<<wF(0, 1, 1)<<endl;
    cout<<"wF.mean = (-54.88) "<<wF.mean()<<endl;
    cout<<endl;
}

int main(){
    string config("param.ini.AB");
    test_field(config);
    test_yita(config);

    return 0;
}
