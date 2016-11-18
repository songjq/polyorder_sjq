/**
 * Test Model, scft, UnitCell, Helper, Config
 *
 * Copyright@ Yi-Xin Liu, 2012
 *
 */

#include "FieldE.h"

#include <iostream>
#include <string>
#include <ctime>
#include <cmath>

#include "UnitCell.h"
#include "Field.h"
#include "Yita.h"
#include "DensitySmall.h"
#include "Density.h"
#include "Propagator.h"
#include "Simpson.h"
#include "multigrid.h"
#include "MUD.h"
#include "MUD2D.h"
#include "MUD3D.h"
#include "Helper.h"
#include "Model_AB.h"
#include "Model_ABSe.h"
#include "scft.h"

using namespace std;

void test_scft(){
    clock_t t_b,t_e;
    double t;

    t_b=clock();
    Config *cfg;
    cfg=new Config("param.ini");
    Model_ABSe model(*cfg);
//    Model_AB model(*cfg);
    delete cfg;
    scft sim("param.ini",&model);
    sim.run();
    t_e=clock();
    t=static_cast<double>(t_e-t_b)/CLOCKS_PER_SEC;
    cout<<"Total time: "<<t<<" s"<<endl;
}

void test_helper(){
    CMatFile mat;
    double lx,ly,lz;
    Config cfg("param.ini");

    cfg.set_grid_type(SQUARE_GRID);
    cout<<"w123"<<endl;
    cout<<GridTypes[cfg.get_grid_type()]<<endl;
    cout<<"(lx,ly,lz) = "<<cfg.get_double("Grid","lx")<<",";
    cout<<cfg.get_double("Grid","ly")<<",";
    cout<<cfg.get_double("Grid","lz")<<endl;;
    Field w1("w1",cfg,0.1);
    Field w2("w2",cfg,0.1);
    Field w3("w3",cfg,0.1);
    cout<<"(lx,ly,lz) = "<<w1.lx()<<","<<w1.ly()<<","<<w1.lz()<<endl;;
    Helper::init_pattern(w1,LAM1_PATTERN,0.4,0.1);
    Helper::init_pattern(w2,LAM2_PATTERN,0.4,0.1);
    Helper::init_pattern(w3,LAM3_PATTERN,0.4,0.1);
    mat.matInit("w123.mat","w");
    mat.matPutScalar("lx",w1.lx());
    mat.matPutScalar("ly",w1.ly());
    mat.matPutScalar("lz",w1.lz());
    mat.matRelease();
    w1.save("w123.mat");
    w2.save("w123.mat");
    w3.save("w123.mat");

    cfg.set_grid_type(RECT_GRID);
    cout<<"w456"<<endl;
    cout<<GridTypes[cfg.get_grid_type()]<<endl;
    cout<<"(lx,ly,lz) = "<<cfg.get_double("Grid","lx")<<",";
    cout<<cfg.get_double("Grid","ly")<<",";
    cout<<cfg.get_double("Grid","lz")<<endl;;
    Field w4("w4",cfg,0.1);
    Field w5("w5",cfg,0.1);
    Field w6("w6",cfg,0.1);
    Helper::init_pattern(w4,LAM4_PATTERN,0.4,0.1);
    Helper::init_pattern(w5,LAM5_PATTERN,0.4,0.1);
    Helper::init_pattern(w6,LAM6_PATTERN,0.4,0.1);
    lx=w4.lx();ly=w4.ly();lz=w4.lz();
    mat.matInit("w456.mat","w");
    mat.matPutScalar("lx",lx);
    mat.matPutScalar("ly",ly);
    mat.matPutScalar("lz",lz);
    mat.matRelease();
    w4.save("w456.mat");
    w5.save("w456.mat");
    w6.save("w456.mat");

    cfg.set_grid_type(HEX_RECT_1_GRID);
    cout<<"h12"<<endl;
    cout<<GridTypes[cfg.get_grid_type()]<<endl;
    cout<<"(lx,ly,lz) = "<<cfg.get_double("Grid","lx")<<",";
    cout<<cfg.get_double("Grid","ly")<<",";
    cout<<cfg.get_double("Grid","lz")<<endl;;
    Field h1("h1",cfg,0.1);
    Field h2("h2",cfg,0.1);
    Helper::init_pattern(h1,HEX1_PATTERN,0.32,0.1);
    Helper::init_pattern(h2,HEX2_PATTERN,0.32,0.1);
    lx=h1.lx();ly=h1.ly();lz=h1.lz();
    mat.matInit("h12.mat","w");
    mat.matPutScalar("lx",lx);
    mat.matPutScalar("ly",ly);
    mat.matPutScalar("lz",lz);
    mat.matRelease();
    h1.save("h12.mat");
    h2.save("h12.mat");

    cfg.set_grid_type(HEX_RECT_2_GRID);
    cout<<"h34"<<endl;
    cout<<GridTypes[cfg.get_grid_type()]<<endl;
    cout<<"(lx,ly,lz) = "<<cfg.get_double("Grid","lx")<<",";
    cout<<cfg.get_double("Grid","ly")<<",";
    cout<<cfg.get_double("Grid","lz")<<endl;;
    Field h3("h3",cfg,0.1);
    Field h4("h4",cfg,0.1);
    Helper::init_pattern(h3,HEX3_PATTERN,0.32,0.1);
    Helper::init_pattern(h4,HEX4_PATTERN,0.32,0.1);
    lx=h3.lx();ly=h3.ly();lz=h3.lz();
    mat.matInit("h34.mat","w");
    mat.matPutScalar("lx",lx);
    mat.matPutScalar("ly",ly);
    mat.matPutScalar("lz",lz);
    mat.matRelease();
    h3.save("h34.mat");
    h4.save("h34.mat");
}

void test_model(){
    Config cfg("param.ini");
    clock_t t_b,t_e,t_begin,t_end;
    double t;
    int step = 10;

    t_b=clock();
    Model_ABSe model(cfg);
//    Model_AB model(cfg);
    model.save_model("model.mat");

    t_begin=clock();
    for(int i=1;i<=3000;i++){
        model.update();
        if(i%step==0){
            t_end=clock();
            t=static_cast<double>(t_end-t_begin)/CLOCKS_PER_SEC;
            t=t/step;
            cout<<i<<endl;
            cout<<"t = "<<t<<endl;
            model.display();
            t_begin=clock();
        }
        if(model.residual_error()<3e-7)
          break;
    }
    model.save("data.mat");
    t_e=clock();
    cout<<endl;
    cout<<"Total time: "<<(t_e-t_b)*1.0/CLOCKS_PER_SEC<<" s"<<endl;
}

int main(){
    //test_helper();
    test_model();
    //test_scft();

    return 0;
}
