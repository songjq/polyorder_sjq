/**
 * Test Propagator
 *
 * Copyright@ Yi-Xin Liu, 2012-2014.
 *
 */

#include "Grid.h"

#include <iostream>
#include <string>
#include <ctime>
#include <cmath>

#include "UnitCell.h"
#include "Field.h"
#include "Propagator.h"

#include "armadillo"

using namespace std;
using namespace arma;

void test_propagator(const string config){
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
    cout<<"Lx = "<<Lx<<endl;
    cout<<"Ly = "<<Ly<<endl;
    cout<<"Lz = "<<Lz<<endl;
    cout<<"fA = "<<fA<<endl;
    cout<<"dsA = "<<dsA<<endl;
    cout<<"dsB = "<<dsB<<endl;
    cout<<"sA = "<<sA<<endl;
    cout<<"sB = "<<sB<<endl;

    vec lam = cfg.lam();
    double lamA = lam(0);
    double lamB = lam(1);
    double low = -0.5;
    double high = 0.5;
    int seed = 0;
    cout<<"lamA = "<<lamA<<endl;
    cout<<"lamB = "<<lamB<<endl;
    cout<<"low = "<<low<<endl;
    cout<<"high = "<<high<<endl;
    cout<<"seed = "<<seed<<endl;
    Field wA("wA", cfg, lamA, low, high, seed);
    Field wB("wB", cfg, lamB, low, high, seed);
    cout<<"wA(0,1,1) = "<<wA(0, 1, 1)<<endl;
    cout<<"wA(0,31,48) = "<<wA(0, 31, 48)<<endl;
    cout<<"wA.mean = "<<wA.mean()<<endl;
    cout<<"wB(0,1,1) = "<<wB(0, 1, 1)<<endl;
    cout<<"wB(0,31,48) = "<<wB(0, 31, 48)<<endl;
    cout<<"wB.mean = "<<wB.mean()<<endl;
    cout<<endl;

    UnitCell uc(cfg);
    Grid one(uc, Lx, Ly, Lz, 1.0);
    cout<<"one(0,1,1) = "<<one(0, 1, 1)<<endl;
    cout<<"one(0,31,48) = "<<one(0, 31, 48)<<endl;
    cout<<"one.mean = "<<one.mean()<<endl;
    Propagator qA("qA", cfg, sA, dsA, one);
    Propagator qB("qB", cfg, sB, dsB);
    Propagator qAc("qAc", cfg, sA, dsA);
    Propagator qBc("qBc", cfg, sB, dsB, one);
    cout<<qA.qs().shape()<<endl;

    qA.update(wA);
    cout<<"qA.dim = (3) "<<qA.dim()<<endl;
    cout<<"qA.ds = (dsA) "<<qA.ds()<<endl;
    cout<<"qA.len = (sA) "<<qA.len()<<endl;
    cout<<"qA[0](0,1,1) = (1) "<<qA(0, 0, 1, 1)<<endl;
    cout<<"qA[0].mean = (1) "<<blitz::mean(qA[0])<<endl;
    cout<<"qA[1](0,0,0),(0,31,63),(0,16,63),(0,31,32)"<<endl;
    cout<<qA(1, 0, 0, 0)<<", "<<qA(1, 0, 31, 63)<<", ";
    cout<<qA(1, 0, 16, 63)<<", "<<qA(1, 0, 31, 32)<<endl;
    cout<<"qA[1].mean = () "<<blitz::mean(qA[1])<<endl;
    cout<<"qA[25](0,0,0),(0,31,63),(0,16,63),(0,31,32)"<<endl;
    cout<<qA(25, 0, 0, 0)<<", "<<qA(25, 0, 31, 63)<<", ";
    cout<<qA(25, 0, 16, 63)<<", "<<qA(25, 0, 31, 32)<<endl;
    cout<<"qA[25].mean = () "<<blitz::mean(qA[25])<<endl;
    cout<<"qA[36](0,0,0),(0,31,63),(0,16,63),(0,31,32)"<<endl;
    cout<<qA(36, 0, 0, 0)<<", "<<qA(36, 0, 31, 63)<<", ";
    cout<<qA(36, 0, 16, 63)<<", "<<qA(36, 0, 31, 32)<<endl;
    cout<<"qA[36].mean = () "<<blitz::mean(qA[50])<<endl;
    cout<<endl;
/*
    qB.set_head(qA.get_tail());
    qB.update(wB);
    cout<<"qB.dim = () "<<qB.dim()<<endl;
    cout<<"qB.ds = (ds) "<<qB.ds()<<endl;
    cout<<"qB.len = (sB) "<<qB.len()<<endl;
    cout<<"qB[0](0,1,1) = () "<<qB[0](0,1,1)<<endl;
    cout<<"qB[0].mean = () "<<blitz::mean(qB[0])<<endl;
    cout<<"qB[1](0,1,1) = () "<<qB[1](0,1,1)<<endl;
    cout<<"qB[1].mean = () "<<blitz::mean(qB[1])<<endl;
    cout<<"qB[25](0,1,1) = () "<<qB[25](0,1,1)<<endl;
    cout<<"qB[25].mean = () "<<blitz::mean(qB[25])<<endl;
    cout<<"qB[50](0,1,1) = () "<<qB[50](0,1,1)<<endl;
    cout<<"qB[50].mean = () "<<blitz::mean(qB[50])<<endl;
    cout<<endl;

    qBc.update(wB);
    cout<<"qBc.dim = () "<<qBc.dim()<<endl;
    cout<<"qBc.ds = (ds) "<<qBc.ds()<<endl;
    cout<<"qBc.len = (sB) "<<qBc.len()<<endl;
    cout<<"qBc[0](0,1,1) = (1) "<<qBc[0](0,1,1)<<endl;
    cout<<"qBc[0].mean = (1) "<<blitz::mean(qBc[0])<<endl;
    cout<<"qBc[1](0,1,1) = () "<<qBc[1](0,1,1)<<endl;
    cout<<"qBc[1].mean = () "<<blitz::mean(qBc[1])<<endl;
    cout<<"qBc[25](0,1,1) = () "<<qBc[25](0,1,1)<<endl;
    cout<<"qBc[25].mean = () "<<blitz::mean(qBc[25])<<endl;
    cout<<"qBc[50](0,1,1) = () "<<qBc[50](0,1,1)<<endl;
    cout<<"qBc[50].mean = () "<<blitz::mean(qBc[50])<<endl;
    cout<<endl;

    qAc.set_head(qBc.get_tail());
    qAc.update(wA);
    cout<<"qAc.dim = () "<<qAc.dim()<<endl;
    cout<<"qAc.ds = (ds) "<<qAc.ds()<<endl;
    cout<<"qAc.len = (sA) "<<qAc.len()<<endl;
    cout<<"qAc[0](0,1,1) = () "<<qAc[0](0,1,1)<<endl;
    cout<<"qA[0].mean = () "<<blitz::mean(qAc[0])<<endl;
    cout<<"qAc[1](0,1,1) = () "<<qAc[1](0,1,1)<<endl;
    cout<<"qAc[1].mean = () "<<blitz::mean(qAc[1])<<endl;
    cout<<"qAc[25](0,1,1) = () "<<qAc[25](0,1,1)<<endl;
    cout<<"qAc[25].mean = () "<<blitz::mean(qAc[25])<<endl;
    cout<<"qAc[50](0,1,1) = () "<<qAc[50](0,1,1)<<endl;
    cout<<"qAc[50].mean = () "<<blitz::mean(qAc[50])<<endl;
    cout<<endl;
*/
/*
    UnitCell uc(cfg);
    Grid one(uc,Lx,Ly,Lz,1.0);
    Propagator qA("qA",cfg,sA,ds,one);
    Propagator qB("qB",cfg,sB,ds);
    cout<<"qA.name = (qA) "<<qA.name()<<endl;
    cout<<"qA.len = (sA) "<<qA.len()<<endl;
    cout<<"qA.ds = (ds) "<<qA.ds()<<endl;
    cout<<"qA.dim = () "<<qA.dim()<<endl;
    cout<<"qA.(Lx,Ly,Lz) = () "<<"("<<qA.Lx()<<","<<qA.Ly()<<","<<qA.Lz()<<")"<<endl;
    cout<<endl;

    cout<<"qA[0](0,1,1) = (1) "<<qA[0](0,1,1)<<endl;
    cout<<"qA[0].mean = (1) "<<blitz::mean(qA[0])<<endl;
    cout<<"qA[20](0,1,1) = (0) "<<qA[20](0,1,1)<<endl;
    cout<<"qA[20].mean = (0) "<<blitz::mean(qA[20])<<endl;
    cout<<"qA[39](0,1,1) = (0) "<<qA[39](0,1,1)<<endl;
    cout<<"qA[39].mean = (0) "<<blitz::mean(qA[39])<<endl;
    blitz::Array<double,3> qq=qA.get_tail();
    cout<<"qA_tail(0,1,1) = (0) "<<qq(0,1,1)<<endl;
    cout<<"qA_tail.mean = (0) "<<blitz::mean(qq)<<endl;
    cout<<endl;

    Grid nine(uc,Lx,Ly,Lz,9.0);
    Grid five(uc,Lx,Ly,Lz,5.0);
    qA.set_head(nine);
    cout<<"qA[0](0,1,1) = (9) "<<qA[0](0,1,1)<<endl;
    cout<<"qA[0].mean = (9) "<<blitz::mean(qA[0])<<endl;
    qA[20]=five.data();
    cout<<"qA[20](0,1,1) = (5) "<<qA[20](0,1,1)<<endl;
    cout<<"qA[20].mean = (5) "<<blitz::mean(qA[20])<<endl;
    cout<<endl;

    cout<<"qA.Q0 = (9) "<<qA.Q(0)<<endl;
    cout<<"qA.Qt = (0) "<<qA.Qt()<<endl;
    cout<<endl;

    Grid zero(uc,Lx,Ly,Lz,0.1);
    Grid one_g(uc,Lx,Ly,Lz,1.0);
    qA.set_head(one_g);
    qA.update(zero);
    cout<<"qA.dim = () "<<qA.dim()<<endl;
    cout<<"qA.ds = (ds) "<<qA.ds()<<endl;
    cout<<"qA[0](0,1,1) = (1) "<<qA[0](0,1,1)<<endl;
    cout<<"qA[0].mean = (1) "<<blitz::mean(qA[0])<<endl;
    cout<<"qA[1](0,1,1) = () "<<qA[1](0,1,1)<<endl;
    cout<<"qA[1].mean = () "<<blitz::mean(qA[1])<<endl;
    cout<<"qA[20](0,1,1) = () "<<qA[20](0,1,1)<<endl;
    cout<<"qA[20].mean = () "<<blitz::mean(qA[20])<<endl;
    cout<<"qA[39](0,1,1) = () "<<qA[39](0,1,1)<<endl;
    cout<<"qA[39].mean = () "<<blitz::mean(qA[39])<<endl;
    cout<<endl;

    qB.set_head(qA.get_tail());
    qB.update(zero);
    cout<<"qB[0](0,1,1) = () "<<qB[0](0,1,1)<<endl;
    cout<<"qB[0].mean = () "<<blitz::mean(qB[0])<<endl;
    cout<<"qB[20](0,1,1) = () "<<qB[20](0,1,1)<<endl;
    cout<<"qB[20].mean = () "<<blitz::mean(qB[20])<<endl;
    cout<<"qB[39](0,1,1) = () "<<qB[39](0,1,1)<<endl;
    cout<<"qB[39].mean = () "<<blitz::mean(qB[39])<<endl;
    cout<<endl;
    */
}

int main(){
    test_propagator("param.ini.AB");

    return 0;
}
