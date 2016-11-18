#include <iostream>

#include "Etdrk4_PBC.h"
#include "Config.h"
#include "UnitCell.h"
#include "Propagator.h"

#include "armadillo"
#include "blitz/Array.h"

using namespace std;
using namespace arma;

void test_propagator_solve(){
    uword N_test = 1;

    Config cfg("test_etdrk4_pbc_1d.ini");
    UnitCell uc(cfg);
    uword Lx = cfg.Lx();  // = 32
    uword Ly = cfg.Ly();  // = 1
    uword Lz = cfg.Lz();  // = 1
    double lx = uc.a();   // = 10.0
    cout<<"lx = "<<lx<<endl;

    uword Ns = 100 + 1;
    double ds = 1.0 / (Ns - 1);
    //Following lines are default values
    //ETDRK4SCHEME stype = ETDRK4SCHEME::KROGSTAD, not implemented yet.
    //uword M = 32;
    Etdrk4_PBC et(uc, Lx, Ly, Lz, ds, 16);

    colvec x = linspace(lx/Lx, lx, Lx);
    colvec sech = 1. / cosh(0.75 * (2.0*x - lx));
    colvec w = 1.0 - 2 * sech % sech;
    w.print("w =");

    Grid gw("wA", cfg);
    cout<<"wA.dim = "<<gw.dim()<<endl;
    cout<<"wA.Lx = "<<gw.Lx()<<endl;
    cout<<"wA.Ly = "<<gw.Ly()<<endl;
    cout<<"wA.Lz = "<<gw.Lz()<<endl;
    for(uword i=0; i<Lx; i++)  // ONLY for 1D
        gw(i) = w(i);
    cout<<"gw = "<<gw.data()<<endl;

    Grid one(uc, Lx, Ly, Lz, 1.0);
    Propagator pq("qA", cfg, Ns, ds, one, &et);
    cout<<"pq.len() = "<<pq.len()<<endl;
    cout<<"pq[0] = "<<pq[0]<<endl;

    pq.update(gw);
    gw.save("test_etdrk4_pbc_1d.mat");
    pq.save("test_etdrk4_pbc_1d.mat");
}

int main(){
    test_propagator_solve();

    return 0;
}
