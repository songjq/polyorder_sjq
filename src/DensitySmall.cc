#include "DensitySmall.h"

#include "Grid.h"
//#include "blitz/Array.h"

using std::string;

DensitySmall::DensitySmall(const string name, const Config &cfg,
                           const double cc):Grid(name, cfg), _cc(cc){

}

DensitySmall::DensitySmall(const string name,
                           const Config &cfg,
                           const double val,
                           const double cc):Grid(name, cfg, val), _cc(cc){

}

DensitySmall::DensitySmall(const string name,
                           const Config &cfg,
                           const blitz::Array<double,3> data,
                           const double cc):Grid(name, cfg, data), _cc(cc){

}

DensitySmall::DensitySmall(const string name,
                           const Config &cfg,
                           const string file,
                           const double cc):Grid(name, cfg, file), _cc(cc){

}

DensitySmall::DensitySmall(const string name,
                           const Config &cfg,
                           const double cc,
                           const double low,
                           const double high,
                           const size_t seed /* seed = 0 */):
                                Grid(name, cfg, low, high, seed), _cc(cc){

}

DensitySmall& DensitySmall::operator= (const Grid &rhs){
    Grid::operator=(rhs);
    return *this;
}

void DensitySmall::update(const Grid &g){
    _update(g);
}

void DensitySmall::update(const Grid &g, const double cc){
    _cc = cc;
    _update(g);
}

void DensitySmall::set_cc(const double cc) {
    _cc = cc;
}

double DensitySmall::cc() const{
    return _cc;
}

void DensitySmall::_update(const Grid &g){
    double c=_cc/Q(g);
    _data = c*blitz::exp(-g.data());
}

const double DensitySmall::Q(const Grid &g) const{
    return blitz::mean(blitz::exp(-g.data()));
}

