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

#include "FieldAX.h"

using namespace std;
using namespace arma;

/**
 * To test FieldAX, please keep the number of grids equal for comparing
 * 1D, 2D and 3D calculations.
 */
void test_field(const string config){
    Config cfg(config.c_str());
    bool is_show = true;

    vec lam = cfg.lam();
    double lamA = lam(0);
    double lamB = lam(1);
    cout<<"lamA = (0.9) "<<lamA<<endl;
    cout<<"lamB = (0.9) "<<lamB<<endl;
    FieldAX wA("wA", cfg, lamA, 3, 0, 1);
    FieldAX wB("wB", cfg, lamB, 0, 0, 1);
    FieldAX w1("w1", cfg, lamB, 0, 0, 1);
    FieldAX w2("w2", cfg, lamB, 0, 0, 1);
    FieldAX w3("w3", cfg, lamB, 0, 0, 1);
    FieldAX w4("w4", cfg, lamB, 0, 0, 1);
    FieldAX w5("w5", cfg, lamB, 0, 0, 1);
    FieldAX w6("w6", cfg, lamB, 0, 0, 1);
    FieldAX w7("w7", cfg, lamB, 0, 0, 1);
    FieldAX w8("w8", cfg, lamB, 0, 0, 1);
    cout<<"wA.lambda = (0.9) "<<wA.lambda()<<endl;
    cout<<"wA.name = (wA) "<<wA.name()<<endl;
    cout<<"wA.dim = (1) "<<wA.dim()<<endl;
    cout<<"wA.Lx = (4) "<<wA.Lx()<<endl;
    cout<<"wA.Ly = (1) "<<wA.Ly()<<endl;
    cout<<"wA.Lz = (1) "<<wA.Lz()<<endl;
    cout<<"n = "<<wA.n_history()<<endl;
    cout<<endl;

    cout<<"k = "<<wA.n_iteration()<<endl;  // k = 0
    cout<<"cur_pos = "<<wA.current_index()<<endl;
    if(is_show) cout<<"wA ="<<wA.data();
    cout<<"wA.mean = (0) "<<wA.mean()<<endl;
    cout<<endl;

    wA.update(w1);  // k = 1
    cout<<"k = "<<wA.n_iteration()<<endl;
    cout<<"cur_pos = "<<wA.current_index()<<endl;
    if(is_show) cout<<"wA ="<<wA.data();
    cout<<"wA.mean = () "<<wA.mean()<<endl;
    cout<<endl;

    wA.update(w2);  // k = 2
    cout<<"k = "<<wA.n_iteration()<<endl;
    cout<<"cur_pos = "<<wA.current_index()<<endl;
    if(is_show) cout<<"wA ="<<wA.data();
    cout<<"wA.mean = () "<<wA.mean()<<endl;
    cout<<endl;

    wA.update(w3);  // k = 3
    cout<<"k = "<<wA.n_iteration()<<endl;
    cout<<"cur_pos = "<<wA.current_index()<<endl;
    if(is_show) cout<<"wA ="<<wA.data();
    cout<<"wA.mean = () "<<wA.mean()<<endl;
    cout<<endl;

    wA.update(w4);  // k = 4
    cout<<"k = "<<wA.n_iteration()<<endl;
    cout<<"cur_pos = "<<wA.current_index()<<endl;
    if(is_show) cout<<"wA ="<<wA.data();
    cout<<"wA.mean = () "<<wA.mean()<<endl;
    cout<<endl;

    wA.update(w5);  // k = 5
    cout<<"k = "<<wA.n_iteration()<<endl;
    cout<<"cur_pos = "<<wA.current_index()<<endl;
    if(is_show) cout<<"wA ="<<wA.data();
    cout<<"wA.mean = () "<<wA.mean()<<endl;
    cout<<endl;

    wA.update(w6);  // k = 6
    cout<<"k = "<<wA.n_iteration()<<endl;
    cout<<"cur_pos = "<<wA.current_index()<<endl;
    if(is_show) cout<<"wA ="<<wA.data();
    cout<<"wA.mean = () "<<wA.mean()<<endl;
    cout<<endl;

    wA.update(w7);  // k = 7
    cout<<"k = "<<wA.n_iteration()<<endl;
    cout<<"cur_pos = "<<wA.current_index()<<endl;
    if(is_show) cout<<"wA ="<<wA.data();
    cout<<"wA.mean = () "<<wA.mean()<<endl;
    cout<<endl;

    wA.update(w8);  // k = 8
    cout<<"k = "<<wA.n_iteration()<<endl;
    cout<<"cur_pos = "<<wA.current_index()<<endl;
    if(is_show) cout<<"wA ="<<wA.data();
    cout<<"wA.mean = () "<<wA.mean()<<endl;
    cout<<endl;
}

int main(){
    string config("param.ini.AB");
    test_field(config);

    return 0;
}
