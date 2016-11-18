#include <blitz/array.h>

#include "Grid.h"
#include "Yita.h"

using std::string;

Yita::Yita(const string name, const Config &cfg, const double lam):
                                                Grid(name,cfg),_lambda(lam){

}

Yita::Yita(const string name, const Config &cfg, const double val,
           const double lam):Grid(name, cfg, val), _lambda(lam){

}

Yita::Yita(const string name, const Config &cfg,
           const blitz::Array<double,3> data,
           const double lam):Grid(name, cfg, data), _lambda(lam){

}

Yita::Yita(const string name, const Config &cfg, const string file,
           const double lam):Grid(name, cfg, file), _lambda(lam){

}

Yita::Yita(const string name, const Config &cfg, const double lam,
           const double low, const double high,
           const size_t seed /* seed = 0 */):
                                Grid(name, cfg, low, high, seed), _lambda(lam){

}

Yita& Yita::operator= (const Grid &rhs){
    Grid::operator=(rhs);
    return *this;
}

void Yita::set_lambda(const double lam){
    _lambda = lam;
}

double Yita::lambda() const{
    return _lambda;
}

void Yita::update(const Grid &u){
    _data += _lambda * u.data();
}

void Yita::update(const blitz::Array<double,3> bu){
    _data += _lambda * bu;
}
