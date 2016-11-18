/**
 * Test Grid.
 *
 * Copyright@ Yi-Xin Liu, 2012-2014
 *
 */

#include "Grid.h"
#include "UnitCell.h"
#include "Cheb.h"
#include "armadillo"

#include <iostream>
#include <string>
#include <ctime>
#include <cmath>

using namespace std;
using namespace arma;

void test_grid(const string config){
    Config cfg(config.c_str());
    UnitCell uc(cfg);

    Grid w0(uc, 1, 128, 128);
    Grid w1(uc, 1, 128, 128, 1.0);
    blitz::Array<double,3> data2(1, 128, 128);
    data2 = 9.0;
    Grid w2(uc, 1, 128, 128, data2);
    cout<<"w0.(Lx,Ly,Lz) = (1,128,128) "<<"("<<w0.Lx()<<","<<w0.Ly()<<","<<w0.Lz()<<")"<<endl;
    cout<<"w0.mean = (0) "<<w0.mean()<<endl;
    cout<<"w0(0,1,1) = (0) "<<w0(0, 1, 1)<<endl;
    cout<<"w1.(Lx,Ly,Lz) = (1,128,128) "<<"("<<w1.Lx()<<","<<w1.Ly()<<","<<w1.Lz()<<")"<<endl;
    cout<<"w1.mean = (1) "<<w1.mean()<<endl;
    cout<<"w1(0,1,1) = (1) "<<w1(0, 1, 1)<<endl;
    cout<<"w2.(Lx,Ly,Lz) = (1,128,128) "<<"("<<w2.Lx()<<","<<w2.Ly()<<","<<w2.Lz()<<")"<<endl;
    cout<<"w2.mean = (9) "<<w2.mean()<<endl;
    cout<<"w2(0,1,1) = (9) "<<w2(0, 1, 1)<<endl;
    cout<<endl;

    Grid wA("wA", cfg, -0.05, 0.05, 1234);
    cout<<"Begin Grid test..."<<endl;
    cout<<"Grid Name: "<<wA.name()<<endl;
    cout<<"(Lx,Ly,Lz) = "<<"("<<wA.Lx()<<","<<wA.Ly()<<","<<wA.Lz()<<")"<<endl;
    cout<<"(lx,ly,lz) = "<<"("<<wA.lx()<<","<<wA.ly()<<","<<wA.lz()<<")"<<endl;
    cout<<"(dx,dy,dz) = "<<"("<<wA.dx()<<","<<wA.dy()<<","<<wA.dz()<<")"<<endl;
    cout<<"(mean,min,max) = "<<"("<<wA.mean()<<",";
    cout<<wA.min()<<","<<wA.max()<<")"<<endl;
    cout<<"number of grids = "<<wA.ngrids()<<endl;
    cout<<"grid volume = "<<wA.volume()<<endl;
    cout<<"random seed = "<<wA.seed()<<endl;
    wA.save("data.mat");

    Grid wB("wB", cfg, -0.05, 0.05, 5678);
    cout<<"wB[0][1][1] = "<<wB(0, 1, 1)<<endl;
    wB(0, 1, 1) = 0.0; // should be 0
    cout<<"wB[0][1][1] = (0) "<<wB(0, 1, 1)<<endl;
    wB(0, 1, 1) = wB(0,1,1) + 1; // should be 1
    cout<<"wB[0][1][1] = (1) "<<wB(0, 1, 1)<<endl;
    wB(0, 1, 1) += 999.0; // should be 1000
    cout<<"wB[0][1][1] = (1000) "<<wB(0, 1, 1)<<endl;
    wB(0, 1, 1)++; // should be 1001;
    cout<<"wB[0][1][1] = (1001) "<<wB(0, 1, 1)<<endl;
    wB.save("data.mat");
    cout<<endl;

    // convert an pointer to a reference
    // succeed
    Grid *pwC;
    pwC=new Grid("wC",cfg,999.0);
    Grid &wC=(*pwC);
    cout<<"pwC = "<<pwC<<endl;
    cout<<"&wC = "<<&wC<<endl;
    wC(1,1)=1.0;
    wC.save("data.mat");
    cout<<endl;

    // overloaded arithmetic operator
    // + - * for Grid Grid passed, / untest
    // + - * / for double Grid  passed
    // + - * for Grid double passed, / untest
    Grid wD("wD",cfg,3.0);
    Grid wE("wE",cfg,2.0);
    Grid wF,wG,wH,wX;
    wF=7+wD;
    cout<<"wF.mean = (10) "<<wF.mean()<<endl;
    wF.set_name("wF");
    cout<<"wF.name = "<<wF.name()<<endl;
    wF.save("data.mat");
    wG=7-wD;
    cout<<"wD.mean = (3) "<<wD.mean()<<endl;
    cout<<"wG.mean = (4) "<<wG.mean()<<endl;
    wG.set_name("wG");
    cout<<"wG.name = "<<wG.name()<<endl;
    wG.save("data.mat");
    wH=7*wD;
    cout<<"wH.mean = (21) "<<wH.mean()<<endl;
    wH.set_name("wH");
    cout<<"wH.name = "<<wH.name()<<endl;
    wH.save("data.mat");
    wX=7/wD;
    cout<<"wX.mean = (2.3333...) "<<wX.mean()<<endl;
    wX.set_name("wX");
    cout<<"wX.name = "<<wX.name()<<endl;
    wX.save("data.mat");
    cout<<endl;

    // +=, -=, *=, /= for Grid passed
    // +=, -=, *= for double passed, /= untest
    Grid wI("wI",cfg,9.0);
    Grid wJ("wJ",cfg,9.0);
    Grid wK("wK",cfg,9.0);
    Grid wY("wY",cfg,9.0);
    Grid wL("wL",cfg,6.0);
    wI+=wL;
    cout<<"wI.mean = (15) "<<wI.mean()<<endl;
    cout<<"wI.name = "<<wI.name()<<endl;
    wI.save("data.mat");
    wJ-=wL;
    cout<<"wJ.mean = (3) "<<wJ.mean()<<endl;
    cout<<"wJ.name = "<<wJ.name()<<endl;
    wJ.save("data.mat");
    wK*=wL;
    cout<<"wK.mean = (54) "<<wK.mean()<<endl;
    cout<<"wK.name = "<<wK.name()<<endl;
    wK.save("data.mat");
    wY/=wL;
    cout<<"wY.mean = (1.5) "<<wY.mean()<<endl;
    cout<<"wY.name = "<<wY.name()<<endl;
    wY.save("data.mat");
    cout<<endl;

    // arithmetic combinations
    Grid wM("wM",cfg,4.0);
    Grid wN("wN",cfg,5.0);
    Grid wO("wO",cfg,6.0);
    Grid wP("wP",cfg,7.0);
    Grid wQ;
    wQ=log(1+exp(3*wM/wO)-2*wN*wN+4*(wO+wP));
    cout<<"wQ.mean = (2.34075) "<<wQ.mean()<<endl;
    wQ.set_name("wQ");
    cout<<"wQ.name = "<<wQ.name()<<endl;
    wQ.save("data.mat");
    cout<<endl;

    // assignment operator
    // test passed
    Grid wR("wR",cfg,8.0);
    Grid wS("wS",cfg,9.0);
    Grid wT;
    wS=wR;
    cout<<"wS.mean = (8) "<<wS.mean()<<endl;
    cout<<"wS.name = (wS) "<<wS.name()<<endl;
    wT=wR;
    cout<<"wT.mean = (8) "<<wT.mean()<<endl;
    cout<<"wT.name = () "<<wT.name()<<endl;
    cout<<endl;

    blitz::Range all = blitz::Range::all();
    double ly = cfg.b();
    int Lx = cfg.Lx();
    int Ly = cfg.Ly();
    int Lz = cfg.Lz();
    Cheb cheb(Ly);
    colvec x = cheb.x();
    x = 0.5 * (1-x) * ly;
    blitz::Array<double, 3> data(Lx, Ly, Lz);
    for (int i=0; i<Lx; i++)
        for(int j=0; j<Ly; j++)
            for(int k=0; k<Lz; k++)
                data(i, j, k) = exp(x(j));
    Grid ww(uc, Lx, Ly, Lz, data);
    cout << ww.quadrature() << endl;
    cout << "The dim is " << cfg.dim() << endl;
    cout << cfg.get_gtypex_string() << endl;
   
}

int main(){
    test_grid("test_param.ini");

    return 0;
}
