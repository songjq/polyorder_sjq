#include <iostream>
#include "armadillo"

#include "Etdrk4.h"
#include "Config.h"
#include "UnitCell.h"

using namespace std;
using namespace arma;

void test_phi1(){
    uword dim = 1;
    double lx = 10.0;
    CrystalSystemType cstype = LAM;
    CellParam cp;
    cp.a = lx;
    UnitCell uc(dim, cstype, cp);
    uword Lx = 64;
    uword Ly = 1;
    uword Lz = 1;
    uword Ns = 100 + 1;
    double ds = 1.0 / (Ns - 1);
    ConfineType ctype = ConfineType::CUBE;
    Boundary lb = Boundary(1, 1, 0);
    Boundary rb = Boundary(1, 1, 0);
    //ETDRK4SCHEME stype = ETDRK4SCHEME::KROGSTAD;
    //uword M = 16;

    Etdrk4 et(uc, dim, Lx, Ly, Lz, ds, ctype, lb, rb);
    //Etdrk4 et(uc, dim, Lx, Ly, Lz, ds, ctype, lb, rb, stype, M);

    et.save();
}

void test_phi3(){
    uword dim = 3;
    double lx = 2 * PI;
    double ly = 2 * PI;
    double lz = 10.0;
    CrystalSystemType cstype = TRICLINIC;
    CellParam cp;
    cp.a = lx;
    cp.b = ly;
    cp.c = lz;
    UnitCell uc(dim, cstype, cp);
    uword Lx = 1;
    uword Ly = 1;
    uword Lz = 64;
    uword Ns = 100 + 1;
    double ds = 1.0 / (Ns - 1);
    ConfineType ctype = ConfineType::CUBE;
    //Following lines are default values
    //ETDRK4SCHEME stype = ETDRK4SCHEME::KROGSTAD
    //uword M = 16;
    Boundary lb = Boundary(1, 1, 0);
    Boundary rb = Boundary(1, 1, 0);
    Etdrk4 et(uc, dim, Lx, Ly, Lz, ds, ctype, lb, rb);

    et.save();
}

int main(){
    //test_phi1();
    test_phi3();

    return 0;
}
