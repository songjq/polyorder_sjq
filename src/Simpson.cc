#include "Simpson.h"
#include "Propagator.h"

#include <blitz/array.h>

#include <iostream>

using namespace std;

Simpson* Simpson::clone() const{
    return new Simpson;
}

void Simpson::solve(blitz::Array<double,3> data,
                    const Propagator &q,
                    const Propagator &qc,
                    const double cc) const{
    int n = q.len() - 1;
    if(n%2 != 0){
        cerr<<"Simpson requires even subintervals!"<<endl;
        return;
    }
    if(n < 4){
        cerr<<"Simpson requires at least 4 subintervals!"<<endl;
        return;
    }

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
    blitz::Array<double,3> qqn(qq(n, all, all, all));
    blitz::Array<double,4> qq1(4.0*qq(blitz::Range(1, n-1, 2), all, all, all));
    blitz::Array<double,4> qq2(2.0*qq(blitz::Range(2, n-2, 2), all, all, all));
    data = 0.0; // clear the data
    data += qq0 + qqn + blitz::sum(qq1(l,i,j,k),l) + blitz::sum(qq2(l,i,j,k),l);
    data *= (cc * q.ds() / 3.0);
}

