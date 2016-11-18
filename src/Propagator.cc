#include "Propagator.h"
#include "Etdrk4_PBC.h"
#include "Grid.h"

#include "CMatFile.h"
//#include "blitz/Array.h"

Propagator::Propagator():_length(0), _ds(0), _updater(0){

}

Propagator::Propagator(const string name, const Config &cfg,
                       const int len, const double ds,
                       Updater *up /* up = 0 */):
                                Grid(name, cfg), _length(len), _ds(ds),
                                _qs(len, _Lx, _Ly, _Lz){
    set_updater(up);
}

Propagator::Propagator(const string name, const Config &cfg,
                       const int len, const double ds,
                       const Grid &data,
                       Updater *up /* up = 0 */):
                                Grid(name, cfg), _length(len), _ds(ds),
                                _qs(len, _Lx, _Ly, _Lz){
    set_head(data);
    set_updater(up);
}

Propagator::Propagator(const string name, const Config &cfg,
                       const int len, const double ds,
                       const blitz::Array<double,3> data,
                       Updater *up /* up = 0 */):
                                Grid(name, cfg), _length(len), _ds(ds),
                                _qs(len, _Lx, _Ly, _Lz){
    set_head(data);
    set_updater(up);
}

double Propagator::operator() (int s, int ix, int iy, int iz){
    return _qs(s, ix, iy, iz);
}

const double Propagator::operator() (int s, int ix, int iy, int iz) const{
    return _qs(s, ix, iy, iz);
}

const blitz::Array<double,3> Propagator::operator[] (const int i) const{
    blitz::Range all = blitz::Range::all();
    return _qs(i, all, all, all);
}

blitz::Array<double,3> Propagator::operator[] (const int i){
    blitz::Range all = blitz::Range::all();
    return _qs(i, all, all, all);
}

const blitz::Array<double,3> Propagator::get_tail() const {
    return (*this)[_length-1];
}

void Propagator::set_head(const Grid &u){
    (*this)[0] = u.data();
}
void Propagator::set_head(const blitz::Array<double,3> bu){
    (*this)[0] = bu;
}

blitz::Array<double,4> Propagator::qs(){
    return _qs;
}

blitz::Array<double,4> Propagator::qs() const{
    return _qs;
}

void Propagator::update(const Grid &w){
    _updater->solve(*this, w);
}

void Propagator::update(const Grid &w, const Updater *up){
    up->solve(*this, w);
}

double Propagator::Q(const int s){
    blitz::Range all = blitz::Range::all();
    _data = _qs(s, all, all, all);  // Propagator is a Grid
    return quadrature();  // Call Grid::quadrature()
}

double Propagator::Qt(){
    return Q(_length-1);
}

int Propagator::len() const{
    return _length;
}

double Propagator::ds() const{
    return _ds;
}

Propagator::~Propagator(){
    //delete _updater;
}

void Propagator::set_updater(Updater *up){
    // Use pointer here to avoid creating the Updater again.
    // But use must make sure that the Updater object shall not be destroyed
    // before this propagator is destroyed.
    // Otherwise, this pointer will point to nothing or dangerous memory.
    if(up)
        _updater = up;
    else
        _updater = new Etdrk4_PBC(_uc, _Lx, _Ly, _Lz, _ds);
}

void Propagator::save(const string file){
    CMatFile mat;
    mwSize dims4[4]={(mwSize)_length, (mwSize)_Lx, (mwSize)_Ly, (mwSize)_Lz};

    blitz::Array<double,4> data(_length, _Lx, _Ly, _Lz, blitz::fortranArray);
    data = _qs;

    mat.matInit(file.c_str(), "u");
    if(!mat.queryStatus()){
        mat.matPut(_name, data.data(), data.size()*sizeof(double), 4, dims4,mxDOUBLE_CLASS,mxREAL);
    }
    else{
        cerr<<"error: cannot open MAT-file: "<<file<<endl;
    }
    mat.matRelease();
}

