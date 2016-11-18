/**
 * Test UnitCell
 *
 * Copyright@ Yi-Xin Liu, 2014
 *
 */
#include <iostream>
#include <string>
#include <ctime>
#include <cmath>

#include "armadillo"

#include "UnitCell.h"

using namespace std;
using namespace arma;

void test_unitcell(const string config){
    clock_t t_begin,t_end;
    double t;
    Config cfg(config.c_str());
    UnitCell uc(cfg);
    cout<<"dimension: "<<uc.dim()<<endl;
    cout<<"crystal system type: "<<uc.type()<<endl;
    cout<<"l: "<<uc.l()<<endl;
    cout<<"h: "<<uc.h()<<endl;
    cout<<"g: "<<uc.g()<<endl;

    uword Lx = cfg.Lx();
    uword Ly = cfg.Ly();
    uword Lz = cfg.Lz();
    uvec Ms = cfg.Ms();
    vec vds = cfg.ds();
    double ds = vds(0);

    blitz::Array<double,3> L1(Lx, Ly, Lz);
    t_begin = clock();
    for(int i=0; i<100; i++)
        L1 = uc.calc_kLaplacian(Lx, Ly, Lz, ds);
    t_end = clock();
    t = static_cast<double>(t_end - t_begin) / CLOCKS_PER_SEC;
    cout<<"t_kLaplacian = "<<t<<endl;

    blitz::Array<double,3> L2(Lx, Ly, Lz);
    t_begin = clock();
    for(int i=0; i<100; i++)
        L2 = uc.calc_kLaplacian_orthogonal(Lx, Ly, Lz, ds);
    t_end = clock();
    t = static_cast<double>(t_end - t_begin) / CLOCKS_PER_SEC;
    cout<<"t_laplace = "<<t<<endl;

    cout<<"L1(0,0,0), (1,1,1), (0,1,1), (1,0,1), (1,1,0), (31,31,15)"<<endl;
    cout<<L1(0,0,0)<<", "<<L1(1,1,1)<<", "<<L1(0,1,1)<<", ";
    cout<<L1(1,0,1)<<", "<<L1(1,1,0)<<", "<<L1(31,31,15)<<endl;
    for(int i=0; i<Lx; i++)
        for(int j=0; j<Ly; j++)
            for(int k=0; k<Lz; k++)
              if(abs(L1(i,j,k) - L2(i,j,k)) > SMALL){
                  cout<<"Unequal at: "<<endl;
                  cout<<"L1("<<i<<",";
                  cout<<j<<",";
                  cout<<k<<") = ";
                  cout<<L1(i,j,k)<<endl;
                  cout<<"L2("<<i<<",";
                  cout<<j<<",";
                  cout<<k<<") = ";
                  cout<<L2(i,j,k)<<endl;
                  return;
              }
    cout<<"Passed."<<endl;
}

int main(){
    test_unitcell("param.ini.AB");

    return 0;
}
