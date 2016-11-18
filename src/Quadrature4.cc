#include "Quadrature4.h"
#include "Propagator.h"

#include <blitz/array.h>

#include <iostream>

using namespace std;

Quad4_Closed* Quad4_Closed::clone() const{
    return new Quad4_Closed;
}

/** For [0, n]
 * See http://ngpy.org/post/simpson for the formula.
 */
void Quad4_Closed::solve(blitz::Array<double,3> data,
                         const Propagator &q,
                         const Propagator &qc,
                         const double cc) const{
    int n = q.len() - 1;
    if(n < 7){
        cerr<<"Quad4_Closed requires at least 7 grid points!"<<endl;
        return;
    }
    double c0 = 0.375;   // =  9/24
    double c1 = 7/6.0;   // = 28/24
    double c2 = 23/24.0; // = 23/24
    blitz::Array<double,4> tq(q.qs());
    blitz::Array<double,4> tqc(qc.qs());
    // q[s](i,j,k) * qc[L-s](i,j,k)
    blitz::Array<double,4> qq(tq * tqc.reverse(blitz::firstDim));

    blitz::firstIndex i;
    blitz::secondIndex j;
    blitz::thirdIndex k;
    blitz::fourthIndex l;
    blitz::Range all = blitz::Range::all();

    blitz::Array<double,3> qq0(qq(0, all, all, all));
    blitz::Array<double,3> qq1(qq(1, all, all, all));
    blitz::Array<double,3> qq2(qq(2, all, all, all));
    blitz::Array<double,3> qqn2(qq(n-2, all, all, all));
    blitz::Array<double,3> qqn1(qq(n-1, all, all, all));
    blitz::Array<double,3> qqn0(qq(n, all, all, all));
    blitz::Array<double,4> qqi(qq(blitz::Range(3, n-3), all, all, all));

    data = 0.0; // clear the data
    data += c0*qq0 + c1*qq1 + c2*qq2;
    data += blitz::sum(qqi(l,i,j,k),l);
    data += c2*qqn2 + c1*qqn1 + c0*qqn0;
    data *= cc * q.ds();
}

Quad4_Open* Quad4_Open::clone() const{
    return new Quad4_Open;
}

/** For (0, n)
 * See http://ngpy.org/post/simpson for the formula.
 */
void Quad4_Open::solve(blitz::Array<double,3> data,
                       const Propagator &q,
                       const Propagator &qc,
                       const double cc) const{
    int n = q.len() - 1;
    if(n < 9){
        cerr<<"Quad4_Open requires at least 9 grid points!"<<endl;
        return;
    }
    double c1 = 55/24.0;   // = 55/24
    double c2 = -1/6.0;    // = -4/24
    double c3 = 33/24.0;   // = 33/24

    blitz::Array<double,4> tq(q.qs());
    blitz::Array<double,4> tqc(qc.qs());
    // q[s](i,j,k) * qc[L-s](i,j,k)
    blitz::Array<double,4> qq(tq * tqc.reverse(blitz::firstDim));

    blitz::firstIndex i;
    blitz::secondIndex j;
    blitz::thirdIndex k;
    blitz::fourthIndex l;
    blitz::Range all = blitz::Range::all();

    blitz::Array<double,3> qq1(qq(1, all, all, all));
    blitz::Array<double,3> qq2(qq(2, all, all, all));
    blitz::Array<double,3> qq3(qq(3, all, all, all));
    blitz::Array<double,3> qqn3(qq(n-3, all, all, all));
    blitz::Array<double,3> qqn2(qq(n-2, all, all, all));
    blitz::Array<double,3> qqn1(qq(n-1, all, all, all));
    blitz::Array<double,4> qqi(qq(blitz::Range(4, n-4), all, all, all));

    data = 0.0; // clear the data
    data += c1*qq1 + c2*qq2 + c3*qq3;
    data += blitz::sum(qqi(l,i,j,k),l);
    data += c3*qqn3 + c2*qqn2 + c1*qqn1;
    data *= cc * q.ds();
}

Quad4_Semiopen_Left* Quad4_Semiopen_Left::clone() const{
    return new Quad4_Semiopen_Left;
}

/** For (0, n]
 * See http://ngpy.org/post/simpson for the formula.
 */
void Quad4_Semiopen_Left::solve(blitz::Array<double,3> data,
                                const Propagator &q,
                                const Propagator &qc,
                                const double cc) const{
    int n = q.len() - 1;
    if(n < 8){
        cerr<<"Quad4_Semiopen_Left requires at least 8 grid points!"<<endl;
        return;
    }
    double c1 = 55/24.0;   // = 55/24
    double c2 = -1/6.0;    // = -4/24
    double c3 = 33/24.0;   // = 33/24
    double cn0 = 0.375;    // =  9/24
    double cn1 = 7/6.0;    // = 28/24
    double cn2 = 23/24.0;  // = 23/24

    blitz::Array<double,4> tq(q.qs());
    blitz::Array<double,4> tqc(qc.qs());
    // q[s](i,j,k) * qc[L-s](i,j,k)
    blitz::Array<double,4> qq(tq * tqc.reverse(blitz::firstDim));

    blitz::firstIndex i;
    blitz::secondIndex j;
    blitz::thirdIndex k;
    blitz::fourthIndex l;
    blitz::Range all = blitz::Range::all();

    blitz::Array<double,3> qq1(qq(1, all, all, all));
    blitz::Array<double,3> qq2(qq(2, all, all, all));
    blitz::Array<double,3> qq3(qq(3, all, all, all));
    blitz::Array<double,3> qqn2(qq(n-2, all, all, all));
    blitz::Array<double,3> qqn1(qq(n-1, all, all, all));
    blitz::Array<double,3> qqn0(qq(n, all, all, all));
    blitz::Array<double,4> qqi(qq(blitz::Range(4, n-3), all, all, all));

    data = 0.0; // clear the data
    data += c1*qq1 + c2*qq2 + c3*qq3;
    data += blitz::sum(qqi(l,i,j,k),l);
    data += cn2*qqn2 + cn1*qqn1 + cn0*qqn0;
    data *= cc * q.ds();
}

Quad4_Semiopen_Right* Quad4_Semiopen_Right::clone() const{
    return new Quad4_Semiopen_Right;
}

/** For (0, n]
 * See http://ngpy.org/post/simpson for the formula.
 */
void Quad4_Semiopen_Right::solve(blitz::Array<double,3> data,
                                 const Propagator &q,
                                 const Propagator &qc,
                                 const double cc) const{
    int n = q.len() - 1;
    if(n < 8){
        cerr<<"Quad4_Semiopen_Right requires at least 8 grid points!"<<endl;
        return;
    }
    double cn1 = 55/24.0;   // = 55/24
    double cn2 = -1/6.0;    // = -4/24
    double cn3 = 33/24.0;   // = 33/24
    double c0 = 0.375;      // =  9/24
    double c1 = 7/6.0;      // = 28/24
    double c2 = 23/24.0;    // = 23/24

    blitz::Array<double,4> tq(q.qs());
    blitz::Array<double,4> tqc(qc.qs());
    // q[s](i,j,k) * qc[L-s](i,j,k)
    blitz::Array<double,4> qq(tq * tqc.reverse(blitz::firstDim));

    blitz::firstIndex i;
    blitz::secondIndex j;
    blitz::thirdIndex k;
    blitz::fourthIndex l;
    blitz::Range all = blitz::Range::all();

    blitz::Array<double,3> qq0(qq(0, all, all, all));
    blitz::Array<double,3> qq1(qq(1, all, all, all));
    blitz::Array<double,3> qq2(qq(2, all, all, all));
    blitz::Array<double,3> qqn3(qq(n-3, all, all, all));
    blitz::Array<double,3> qqn2(qq(n-2, all, all, all));
    blitz::Array<double,3> qqn1(qq(n-1, all, all, all));
    blitz::Array<double,4> qqi(qq(blitz::Range(3, n-4), all, all, all));

    data = 0.0; // clear the data
    data += c0*qq0 + c1*qq1 + c2*qq2;
    data += blitz::sum(qqi(l,i,j,k),l);
    data += cn3*qqn3 + cn2*qqn2 + cn1*qqn1;
    data *= cc * q.ds();
}
