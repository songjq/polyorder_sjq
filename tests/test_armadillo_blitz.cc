#include <iostream>
#include "armadillo"
#include "blitz/array.h"
#include "Grid.h"

using namespace std;
using namespace arma;

void test_1d(){
    uword dim = 1;
    double lx = 2 * PI;
    double ly = 0;
    double lz = 0;
    CrystalSystemType cstype = LAMELLAR;
    CellParam cp;
    cp.a = lx;
    cp.b = ly;
    cp.c = lz;
    UnitCell uc(dim, cstype, cp);

    uword Lx = 8;
    uword Ly = 1;
    uword Lz = 1;

    Grid w(uc, Lx, Ly, Lz);
    for(uword i=0; i<Lx; i++)
        w(i) = i;
    // w data [0, 1, 2, 3, 4, 5, 6, 7]

    colvec wb(w.data().data(), Lx, false);
    wb.print("wb =");  // Using Armadillo output
    // expected: [0, 1, 2, 3, 4, 5, 6, 7].T

    wb(Lx/2) = 99;
    cout<<"w ="<<w.data()<<endl;  // Using blitz++ output
    // expected: [0, 1, 2, 3, 99, 5, 6, 7].T
}

void test_2d(){
    uword Lx = 3;
    uword Ly = 4;
    uword Lz = 1;

    blitz::Array<double, 2> w(Lx, Ly);
    blitz::firstIndex i;
    blitz::secondIndex j;
    w = Ly * i + j;
    cout<<"w ="<<w<<endl;
    // expected: [0, 1, 2, 3
    //            4, 5, 6, 7
    //            8, 9, 10, 11]

    /** VERY IMPORTANT **/
    // From blitz to Armadillo
    // Array storage is from Row-Major to Column-Major
    // We should treat blitz's row as Armadillo's column, and vice versa.
    // Nothing else needs to be done.
    // Just keep in mind that the row in blitz is the column in Armadillo,
    // and the column in blitz is the row in Armadillo.
    mat wb(w.data(), Ly, Lx, false);
    wb.print("wb =");  // Using Armadillo output
    // expected: [0, 4, 8,
    //            1, 5, 9,
    //            2, 6, 10,
    //            3, 7, 11]

    wb(Ly/2, Lx/2) = 99;
    wb.print("wb =");  // Using Armadillo output
    // expected: [0, 4, 8,
    //            1, 5, 9,
    //            2, 99, 10,
    //            3, 7, 11]
    cout<<"w ="<<w<<endl;  // Using blitz++ output
    // expected: [0, 1, 2, 3
    //            4, 5, 99, 7
    //            8, 9, 10, 11]
}

void test_3d(){
    uword dim = 3;
    double lx = 2 * PI;
    double ly = 2 * PI;
    double lz = 10.0;
    CrystalSystemType cstype = LAMELLAR;
    CellParam cp;
    cp.a = lx;
    cp.b = ly;
    cp.c = lz;
    UnitCell uc(dim, cstype, cp);

    uword Lx = 3;
    uword Ly = 4;
    uword Lz = 2;

    Grid wg(uc, Lx, Ly, Lz);
    blitz::Array<double, 3> w = wg.data();

    cout<<"number of rows: "<<w.rows()<<endl;  // number of rows = Lx
    cout<<"number of cols: "<<w.cols()<<endl;  // number of cols = Ly
    cout<<"depth: "<<w.depth()<<endl;  // number of slices (depth) = Lz

    blitz::firstIndex i;
    blitz::secondIndex j;
    blitz::thirdIndex k;
    w = Ly * Lz * i + Lz * j + k;
    blitz::Range all = blitz::Range::all();
    cout<<"w(:, :, 0) ="<<w(all, all, 0)<<endl;  // Using blitz++ output
    cout<<"w(:, :, 1) ="<<w(all, all, 1)<<endl;
    // expected:

    cube wb(w.data(), Lz, Ly, Lx, false);
    cout<<"&w = "<<w.data()<<endl;
    cout<<"&wb = "<<wb.memptr()<<endl;
    wb.print("wb =");  // Using Armadillo output
    mat wb0 = wb(span(0,0), span::all, span::all);
    mat wb1 = wb(span(1,1), span::all, span::all);
    wb0.print("wb(0, :, :) =");
    wb1.print("wb(0, :, :) =");
    // expected:

    wb(Lz/2, Ly/2, Lx/2) = 99;
    wb.print("wb =");  // Using Armadillo output
    wb0 = wb(span(0,0), span::all, span::all);
    wb1 = wb(span(1,1), span::all, span::all);
    wb0.print("wb(0, :, :) =");
    wb1.print("wb(0, :, :) =");
    cout<<"w(:, :, 0) ="<<w(all, all, 0)<<endl;  // Using blitz++ output
    cout<<"w(:, :, 1) ="<<w(all, all, 1)<<endl;
    // expected:
}

void test_4d(){
    uword dim = 3;
    double lx = 2 * PI;
    double ly = 2 * PI;
    double lz = 10.0;
    CrystalSystemType cstype = LAMELLAR;
    CellParam cp;
    cp.a = lx;
    cp.b = ly;
    cp.c = lz;
    UnitCell uc(dim, cstype, cp);

    uword Ns = 2;
    uword Lx = 3;
    uword Ly = 4;
    uword Lz = 2;

    blitz::Array<double, 4> q(Ns, Lx, Ly, Lz);

    cout<<"number of rows: "<<q.rows()<<endl;  // number of rows = Ns
    cout<<"number of cols: "<<q.cols()<<endl;  // number of cols = Lx
    cout<<"depth: "<<q.depth()<<endl;  // number of slices (depth) = Ly

    cout<<"Initialize blitz++ array:"<<endl;
    blitz::firstIndex i;
    blitz::secondIndex j;
    blitz::thirdIndex k;
    blitz::fourthIndex l;
    q = Lx * Ly * Lz * i+ Ly * Lz * j + Lz * k + l;
    blitz::Range all = blitz::Range::all();
    cout<<"q(0, :, :, 0) ="<<q(0, all, all, 0)<<endl;  // Using blitz++ output
    cout<<"q(0, :, :, 1) ="<<q(0, all, all, 1)<<endl;
    cout<<"q(1, :, :, 0) ="<<q(1, all, all, 0)<<endl;
    cout<<"q(1, :, :, 1) ="<<q(1, all, all, 1)<<endl;

    cout<<"Referece blitz++ array in Armadillo: "<<endl;
    field<cube> qb(Ns);  // Here, all qb(s) are initialized to empty cube
    for(uword s=0; s<Ns; s++){
        blitz::Array<double, 3> u = q(s, all, all, all);

        // The data in u is copied to qb(s)
        //qb(s) = cube(u.data(), Lz, Ly, Lx, false);

        // Although u and ub share the same data storage,
        // This data is also copied to qb(s)
        // So we cannot modify q by modifying qb.
        cube ub(u.data(), Lz, Ly, Lx, false);
        qb(s) = ub;
        cout<<"&q ="<<q.data()<<endl;
        cout<<"&u ="<<u.data()<<endl;
        cout<<"&ub ="<<ub.memptr()<<endl;
        cout<<"&qb ="<<qb(s).memptr()<<endl;
    }
    mat ub0 = qb(0)(span(0,0), span::all, span::all);
    mat ub1 = qb(0)(span(1,1), span::all, span::all);
    mat ub2 = qb(1)(span(0,0), span::all, span::all);
    mat ub3 = qb(1)(span(1,1), span::all, span::all);
    ub0.print("qb(0, 0, :, :) =");
    ub1.print("qb(0, 1, :, :) =");
    ub2.print("qb(1, 0, :, :) =");
    ub3.print("qb(1, 1, :, :) =");

    cout<<"Modify array in Armadillo: "<<endl;
    qb(0)(Lz/2, Ly/2, Lx/2) = 99;
    ub0 = qb(0)(span(0,0), span::all, span::all);
    ub1 = qb(0)(span(1,1), span::all, span::all);
    ub2 = qb(1)(span(0,0), span::all, span::all);
    ub3 = qb(1)(span(1,1), span::all, span::all);
    ub0.print("qb(0, 0, :, :) =");
    ub1.print("qb(0, 1, :, :) =");
    ub2.print("qb(1, 0, :, :) =");
    ub3.print("qb(1, 1, :, :) =");

    cout<<"Check if array has been changed in blitz by Armadillo:"<<endl;
    cout<<"q(0, :, :, 0) ="<<q(0, all, all, 0)<<endl;  // Using blitz++ output
    cout<<"q(0, :, :, 1) ="<<q(0, all, all, 1)<<endl;
    cout<<"q(1, :, :, 0) ="<<q(1, all, all, 0)<<endl;
    cout<<"q(1, :, :, 1) ="<<q(1, all, all, 1)<<endl;
    //Conclusion: Failed, because data has been copied from blitz to Armadillo.

    cout<<"Check if array has been changed in blitz by copying:"<<endl;
    for(uword s=0; s<Ns; s++){
        blitz::Array<double, 3> u(qb(s).memptr(), blitz::shape(Lx, Ly, Lz),
                                  blitz::neverDeleteData);
        q(s, all, all, all) = u;
    }
    cout<<"q(0, :, :, 0) ="<<q(0, all, all, 0)<<endl;  // Using blitz++ output
    cout<<"q(0, :, :, 1) ="<<q(0, all, all, 1)<<endl;
    cout<<"q(1, :, :, 0) ="<<q(1, all, all, 0)<<endl;
    cout<<"q(1, :, :, 1) ="<<q(1, all, all, 1)<<endl;
    //Conclusion: Successful!
}

void test_arma2d_to_blitz4d(){
    uword dim = 1;
    uword Ns = 3;
    uword Lx = 4;
    uword Ly = 1;
    uword Lz = 1;

    blitz::Range all = blitz::Range::all();

    blitz::Array<double, 4> qb(Ns, Lx, Ly, Lz);
    qb = 0;
    cout<<"qb(0, :, 0, 0) ="<<qb(0, all, 0, 0)<<endl;
    cout<<"qb(1, :, 0, 0) ="<<qb(1, all, 0, 0)<<endl;
    cout<<"qb(2, :, 0, 0) ="<<qb(2, all, 0, 0)<<endl;

    mat qa;
    qa << 1 << 5 << 9  << endr
       << 2 << 6 << 10 << endr
       << 3 << 7 << 11 << endr
       << 4 << 8 << 12 << endr;
    qa.print("qa =");

    blitz::Array<double, 4> qt(qa.memptr(), blitz::shape(Ns, Lx, Ly, Lz),
                               blitz::neverDeleteData);
    cout<<"qt(0, :, 0, 0) ="<<qt(0, all, 0, 0)<<endl;
    cout<<"qt(1, :, 0, 0) ="<<qt(1, all, 0, 0)<<endl;
    cout<<"qt(2, :, 0, 0) ="<<qt(2, all, 0, 0)<<endl;
    qb = qt;
    cout<<"qb(0, :, 0, 0) ="<<qb(0, all, 0, 0)<<endl;
    cout<<"qb(1, :, 0, 0) ="<<qb(1, all, 0, 0)<<endl;
    cout<<"qb(2, :, 0, 0) ="<<qb(2, all, 0, 0)<<endl;
}

void test_arma3d_to_blitz4d(){
    uword dim = 2;
    uword Ns = 3;
    uword Lx = 4;
    uword Ly = 2;
    uword Lz = 1;

    blitz::Range all = blitz::Range::all();

    blitz::Array<double, 4> qb(Ns, Lx, Ly, Lz);
    qb = 0;
    cout<<"qb(0, :, :, 0) ="<<qb(0, all, all, 0)<<endl;
    cout<<"qb(1, :, :, 0) ="<<qb(1, all, all, 0)<<endl;
    cout<<"qb(2, :, :, 0) ="<<qb(2, all, all, 0)<<endl;

    cube qa(Ly, Lx, Ns);
    mat u;
    u << 1 << 3 << 5 << 7 << endr
      << 2 << 4 << 6 << 8 << endr;
    for(uword s=0; s<Ns; s++)
        qa.slice(s) = u + Ly * Lx * s;
    qa.print("qa =");

    blitz::Array<double, 4> qt(qa.memptr(), blitz::shape(Ns, Lx, Ly, Lz),
                               blitz::neverDeleteData);
    cout<<"qt(0, :, :, 0) ="<<qt(0, all, all, 0)<<endl;
    cout<<"qt(1, :, :, 0) ="<<qt(1, all, all, 0)<<endl;
    cout<<"qt(2, :, :, 0) ="<<qt(2, all, all, 0)<<endl;
    qb = qt;
    cout<<"qb(0, :, :, 0) ="<<qb(0, all, all, 0)<<endl;
    cout<<"qb(1, :, :, 0) ="<<qb(1, all, all, 0)<<endl;
    cout<<"qb(2, :, :, 0) ="<<qb(2, all, all, 0)<<endl;
}

void test_arma4d_to_blitz4d(){
    uword dim = 3;
    uword Ns = 2;
    uword Lx = 4;
    uword Ly = 3;
    uword Lz = 2;

    blitz::Range all = blitz::Range::all();

    blitz::Array<double, 4> qb(Ns, Lx, Ly, Lz);
    qb = 0;
    cout<<"qb(0, :, :, 0) ="<<qb(0, all, all, 0)<<endl;
    cout<<"qb(0, :, :, 1) ="<<qb(0, all, all, 1)<<endl;
    cout<<"qb(1, :, :, 0) ="<<qb(1, all, all, 0)<<endl;
    cout<<"qb(1, :, :, 1) ="<<qb(1, all, all, 1)<<endl;

    cube cu(Lz, Ly, Lx);
    mat u;
    u << 1 << 3 << 5 << endr
      << 2 << 4 << 6 << endr;
    for(uword i=0; i<Lx; i++)
        cu.slice(i) = u + Lz * Ly * i;
    field<cube> qa(Ns);
    for(uword s=0; s<Ns; s++){
        qa(s) = cu + Lz * Ly * Lx * s;
        mat ua0 = qa(s)(span(0,0), span::all, span::all);
        mat ua1 = qa(s)(span(1,1), span::all, span::all);
        ua0.print("qa(0, 0, :, :) =");
        ua1.print("qa(0, 1, :, :) =");
    }

    for(uword s=0; s<Ns; s++){
        blitz::Array<double, 3> ut(qa(s).memptr(), blitz::shape(Lx, Ly, Lz),
                                   blitz::neverDeleteData);
        qb(s, all, all, all) = ut;
    }
    cout<<"qb(0, :, :, 0) ="<<qb(0, all, all, 0)<<endl;
    cout<<"qb(0, :, :, 1) ="<<qb(0, all, all, 1)<<endl;
    cout<<"qb(1, :, :, 0) ="<<qb(1, all, all, 0)<<endl;
    cout<<"qb(1, :, :, 1) ="<<qb(1, all, all, 1)<<endl;
}

void test_blitz3d_colmajor_to_arma3d(){
    uword dim = 3;
    uword Ns = 2;
    uword Lx = 4;
    uword Ly = 3;
    uword Lz = 2;

    blitz::Range all = blitz::Range::all();
    blitz::firstIndex i;
    blitz::secondIndex j;
    blitz::thirdIndex k;

    blitz::Array<double, 3> wb(Lx, Ly, Lz);
    wb = i + Lx * j + Lx * Ly * k;
    cout<<"wb(:, :, 0) ="<<wb(all, all, 0)<<endl;
    cout<<"wb(:, :, 1) ="<<wb(all, all, 1)<<endl;
    blitz::Array<double, 3> wb_col(Lx, Ly, Lz, blitz::ColumnMajorArray<3>());
    // data logical order preserved by blitz internal mechanics.
    wb_col = wb;
    cout<<"wb_col(:, :, 0) ="<<wb_col(all, all, 0)<<endl;
    cout<<"wb_col(:, :, 1) ="<<wb_col(all, all, 1)<<endl;

    // data logical order lost because of row-major to col-major transfer
    cube wa(wb.data(), Lx, Ly, Lz);
    wa.print("wa =");
    // data logical order preserved by rearrange storage order in blitz using
    // blitz::ColumnMajorArray.
    cube wa_col(wb_col.data(), Lx, Ly, Lz);
    wa_col.print("wa_col =");
}

void test_arma3d_to_blitz3d_colmajor(){
    uword dim = 3;
    uword Ns = 2;
    uword Lx = 4;
    uword Ly = 3;
    uword Lz = 2;

    blitz::Range all = blitz::Range::all();
    blitz::firstIndex i;
    blitz::secondIndex j;
    blitz::thirdIndex k;

    blitz::Array<double, 3> wb(Lx, Ly, Lz);
    wb = i + Lx * j + Lx * Ly * k;
    cout<<"wb(:, :, 0) ="<<wb(all, all, 0)<<endl;
    cout<<"wb(:, :, 1) ="<<wb(all, all, 1)<<endl;
    blitz::Array<double, 3> wb_col(Lx, Ly, Lz, blitz::ColumnMajorArray<3>());
    // data logical order preserved by blitz internal mechanics.
    wb_col = wb;
    cout<<"wb_col(:, :, 0) ="<<wb_col(all, all, 0)<<endl;
    cout<<"wb_col(:, :, 1) ="<<wb_col(all, all, 1)<<endl;

    // data logical order preserved by rearrange storage order in blitz using
    // blitz::ColumnMajorArray.
    cube wa_col(wb_col.data(), Lx, Ly, Lz);
    wa_col.print("wa_col =");

    wa_col += 1;    // Do something in arma
    wa_col.print("wa_col + 1 =");
    // Transfer back to blitz
    blitz::Array<double, 3> wwb_col(wa_col.memptr(), blitz::shape(Lx, Ly, Lz),
                                    blitz::neverDeleteData,
                                    blitz::ColumnMajorArray<3>());
    cout<<"wwb_col(:, :, 0) ="<<wwb_col(all, all, 0)<<endl;
    cout<<"wwb_col(:, :, 1) ="<<wwb_col(all, all, 1)<<endl;
    // data logical order preserved by blitz internal mechanics.
    blitz::Array<double, 3> wwb(Lx, Ly, Lz);
    wwb = wwb_col;
    cout<<"wwb(:, :, 0) ="<<wwb(all, all, 0)<<endl;
    cout<<"wwb(:, :, 1) ="<<wwb(all, all, 1)<<endl;
}

void test_arma4d_to_blitz4d_colmajor(){
}

int main(){
    //test_1d();
    //test_2d();
    //test_3d();
    //test_4d();

    //test_arma2d_to_blitz4d();
    //test_arma3d_to_blitz4d();
    //test_arma4d_to_blitz4d();

    //test_blitz3d_colmajor_to_arma3d();
    test_arma3d_to_blitz3d_colmajor();
    //test_arma4d_to_blitz4d_colmajor();

    return 0;
}
