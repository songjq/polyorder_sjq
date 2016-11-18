#include <blitz/array.h>

#include "Grid.h"
#include "Propagator.h"
#include "Updater.h"

#include "Density.h"

using std::string;

Density::Density():_cc(1.0), _updater(0){

}

Density::Density(const Density &rhs):Grid(rhs), _cc(rhs._cc), _updater(0){set_updater(rhs._updater);
}

Density::Density(const string name, const Config &cfg,
                 const Updater *up /* up = 0 */):Grid(name, cfg), _cc(1.0),
                                                 _updater(0){
    set_updater(up);
}

Density::Density(const string name, const Config &cfg, const double val,
                 const Updater *up /* up = 0 */):Grid(name, cfg, val),
                                                 _cc(1.0), _updater(0){
    set_updater(up);
}

Density::Density(const string name, const Config &cfg,
                 const blitz::Array<double,3> data,
                 const Updater *up /* up = 0 */):Grid(name, cfg, data),
                                                 _cc(1.0), _updater(0){
    set_updater(up);
}

Density::Density(const string name, const Config &cfg, const string file,
                 const Updater *up /* up = 0 */):Grid(name, cfg, file),
                                                 _cc(1.0), _updater(0){
    set_updater(up);
}

Density::Density(const string name, const Config &cfg,
                 const double low, const double high,
                 const size_t seed, /* seed = 0 */
                 const Updater *up /* up = 0 */):
                            Grid(name, cfg, low, high, seed),
                            _cc(1.0), _updater(0){
    set_updater(up);
}

Density& Density::operator= (const Density &rhs){
    if(&rhs != this){
        Grid::operator=(rhs);
        set_updater(rhs._updater);
    }
    return *this;
}

Density& Density::operator= (const Grid &rhs){
    Grid::operator=(rhs);
    return *this;
}

void Density::set_cc(const double cc) {
    _cc = cc;
}

double Density::cc() const{
    return _cc;
}

Density::~Density(){
    delete _updater;
}

void Density::set_updater(const Updater *up){
    if(up) _updater = up->clone();
}



void Density::update(const Propagator &q, const Propagator &qc){
    if(_updater)
        _updater->solve(_data, q, qc, _cc);
    else
        _update(q, qc);
}

void Density::update(const Propagator &q, const Propagator &qc,
                     const double cc){
    _cc = cc;
    update(q, qc);
}

void Density::update(const Propagator &q, const Propagator &qc,
                     const Updater *up){
    up->solve(_data, q, qc, _cc);
}

void Density::update(const Propagator &q, const Propagator &qc,
                     const double cc,const Updater *up){
    _cc = cc;
    up->solve(_data, q, qc, _cc);
}

void Density::_update(const Propagator &q, const Propagator &qc){
    blitz::Array<double,4> tq(q.qs());
    blitz::Array<double,4> tqc(qc.qs());
    // q[s](i,j,k) * qc[L-s](i,j,k)
    blitz::Array<double,4> qq(tq * tqc.reverse(blitz::firstDim));
    blitz::Array<double,3> q0(0.5 * q[0] * qc.get_tail());
    blitz::Array<double,3> qt(0.5 * qc[0] * q.get_tail());
    blitz::firstIndex i;
    blitz::secondIndex j;
    blitz::thirdIndex k;
    blitz::fourthIndex l;
    _data = blitz::sum(qq(l, i, j, k), l);
    _data -= (q0 + qt);
    _data *= (_cc * q.ds());
}
