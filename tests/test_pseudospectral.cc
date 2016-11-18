/**
 * Test 2nd order Pseudospectral algorithm by comparing to Etdrk4_PBC algorithm
 * which has been tested.
 *
 */

#include <iostream>

#include "PseudoSpectral.h"
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

    Config cfg("test_pseudospectral.ini");
    UnitCell uc(cfg);
    uword Lx = cfg.Lx();  // = 32
    uword Ly = cfg.Ly();  // = 1
    uword Lz = cfg.Lz();  // = 1
    double lx = uc.a();   // = 10.0
    cout<<"lx = "<<lx<<endl;

    uword Ns = 100 + 1;
    double ds = 1.0 / (Ns - 1);

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
    //Following lines are default values
    //ETDRK4SCHEME stype = ETDRK4SCHEME::KROGSTAD, not implemented yet.
    //uword M = 32;
    Etdrk4_PBC et(uc, Lx, Ly, Lz, ds, 16);
    Propagator pq_et("q_et", cfg, Ns, ds, one, &et);
    PseudoSpectral os(uc, Lx, Ly, Lz, ds);
    //PseudoSpectral os(os0);  // For testing copy constructor
    Propagator pq_os("q_os", cfg, Ns, ds, one, &os);

    pq_et.update(gw);
    pq_os.update(gw);

    cout<<"Difference between ETDRK4 and OS2: ";
    cout<<blitz::max(pq_os.get_tail() - pq_et.get_tail())<<endl;

    gw.save("test_pseudospectral.mat");
    pq_et.save("test_pseudospectral.mat");
    pq_os.save("test_pseudospectral.mat");
}

int main(){
    test_propagator_solve();

    return 0;
}
