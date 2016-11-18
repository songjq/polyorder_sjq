#include <blitz/array.h>

#include "Grid.h"
#include "Field.h"

using std::string;

Field::Field(const string name, const Config &cfg, const double lam):
                                            Grid(name, cfg), _lambda(lam){

}

Field::Field(const string name, const Config &cfg, const double val,
             const double lam):Grid(name, cfg, val), _lambda(lam){

}

Field::Field(const string name, const Config &cfg,
             const blitz::Array<double,3> data,
             const double lam):Grid(name, cfg, data), _lambda(lam){

}

Field::Field(const string name, const Config &cfg, const string file,
             const double lam):Grid(name, cfg, file), _lambda(lam){

}

Field::Field(const string name, const Config &cfg, const double lam,
             const double low, const double high,
             const size_t seed /* seed = 0 */):
                                Grid(name, cfg, low, high, seed), _lambda(lam){

}

Field& Field::operator= (const Grid &rhs){
    Grid::operator=(rhs);
    return *this;
}

void Field::set_lambda(const double lam) {
    _lambda=lam;
}

double Field::lambda() const{
    return _lambda;
}

void Field::update(const Grid &u){
    _data *= (1.0 - _lambda);
    _data += _lambda * u.data();
}

void Field::update(const blitz::Array<double,3> bu){
    _data *= (1.0 - _lambda);
    _data += _lambda * bu;
}

double Field::Q(const double N) const{
    return blitz::mean(blitz::exp(-_data / N));
}
