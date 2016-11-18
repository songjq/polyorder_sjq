/**
 * Grid.cc
 * Created at 2011.6.6
 *
 * Implementation of Grid.h
 *
 * Copyright (C) 2012 Yi-Xin Liu <lyx@fudan.edu.cn>
 *
 * This file is part of Polyorder
 *
 * Polyorder is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License
 * as published by the Free Software Foundation, either version 3
 * of the License, or (at your option) any later version.
 *
 * Polyorder is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Polyorder.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#include "common.h"
#include "Config.h"
#include "UnitCell.h"
#include "Grid.h"

//#include "blitz/Array.h"
#include "armadillo"
#include "Cheb.h"

Grid::Grid(const Grid &rhs):_uc(rhs._uc),
                            _Lx(rhs._Lx), _Ly(rhs._Ly), _Lz(rhs._Lz),
                            _ctype(rhs._ctype),
                            _gtypex(rhs._gtypex),
                            _gtypey(rhs._gtypey),
                            _gtypez(rhs._gtypez),
                            _grid_init_type(rhs._grid_init_type),
                            _seed(0),
                            _data(_Lx, _Ly, _Lz){
    _data = rhs._data;
}

Grid::Grid(const UnitCell &uc, const int Lx, const int Ly, const int Lz):
                                _uc(uc), _Lx(Lx), _Ly(Ly), _Lz(Lz),
                                _ctype(ConfineType::NONE),
                                _gtypex(GridType::REGULAR),
                                _gtypey(GridType::REGULAR),
                                _gtypez(GridType::REGULAR),
                                _grid_init_type(GridInitType::RANDOM_INIT),
                                _seed(0),
                                _data(Lx, Ly, Lz){
    _data = 0.0;
}

Grid::Grid(const UnitCell &uc, const int Lx, const int Ly, const int Lz,
           const double val):_uc(uc), _Lx(Lx), _Ly(Ly), _Lz(Lz),
                             _ctype(ConfineType::NONE),
                             _gtypex(GridType::REGULAR),
                             _gtypey(GridType::REGULAR),
                             _gtypez(GridType::REGULAR),
                             _grid_init_type(GridInitType::RANDOM_INIT),
                             _seed(0),
                             _data(Lx, Ly, Lz){
    _data = val;
}

Grid::Grid(const UnitCell &uc, const int Lx, const int Ly, const int Lz,
           const blitz::Array<double,3> data)
                                    :_uc(uc),
                                    _Lx(Lx), _Ly(Ly), _Lz(Lz),
                                    _ctype(ConfineType::NONE),
                                    _gtypex(GridType::REGULAR),
                                    _gtypey(GridType::REGULAR),
                                    _gtypez(GridType::REGULAR),
                                    _grid_init_type(GridInitType::RANDOM_INIT),
                                    _seed(0),
                                    _data(Lx, Ly, Lz){
    _data = data;
}

Grid::Grid(const string name):_name(name), _seed(0){

}

Grid::Grid(const string name, const Config &cfg):_name(name),
                                                 _uc(cfg),
                                                 _seed(0){
    init(cfg);
    _data.resize(_Lx, _Ly, _Lz);
    _data = 0.0;
}

Grid::Grid(const string name, const Config &cfg, const double val):
                                            _name(name), _uc(cfg), _seed(0){
    init(cfg);
    _data.resize(_Lx, _Ly, _Lz);
    _data = val;
}

Grid::Grid(const string name, const Config &cfg,
           const blitz::Array<double,3> data):_name(name), _uc(cfg), _seed(0){
    init(cfg);
    _data.resize(_Lx, _Ly, _Lz);
    _data = data;
}

Grid::Grid(const string name, const Config &cfg, const string file):
                                            _name(name),_uc(cfg),_seed(0){
    init(cfg);
    set_value(file);
}

Grid::Grid(const string name, const Config &cfg,
           const double low, const double high,
           const size_t seed /* seed = 0 */): _name(name), _uc(cfg),
                                              _seed(seed){
    init(cfg);
    set_value(seed, low, high);
}

/**
 * Assignment operator must be explicitly defined because the assignment
 * operator of `blitz::Array` does not work properly.
 *
 */
Grid& Grid::operator= (const Grid &rhs){
    if (&rhs != this){
        _uc = rhs._uc;
        _Lx = rhs._Lx;
        _Ly = rhs._Ly;
        _Lz = rhs._Lz;
        _ctype = rhs._ctype;
        _gtypex = rhs._gtypex;
        _gtypey = rhs._gtypey;
        _gtypez = rhs._gtypez;
        _grid_init_type = rhs._grid_init_type;
        _seed = rhs._seed;
        _data.resize(_Lx, _Ly, _Lz);
        _data=rhs.data();
    }
    return *this;
}

blitz::Array<double,3> Grid::data(){
    return _data;
}

const blitz::Array<double,3> Grid::data() const{
    return _data;
}

double& Grid::operator() (int ix, int iy, int iz){
    return _data(ix, iy, iz);
}

const double& Grid::operator() (int ix, int iy, int iz) const{
    return _data(ix, iy, iz);
}

// this hides the complicated internal structure of array for 2D space.
double& Grid::operator() (int ix, int iy){
    return _data(ix, iy, 0);
}

const double& Grid::operator() (int ix, int iy) const{
    return _data(ix, iy, 0);
}

// this hides the complicated internal structure of array for 1D space.
double& Grid::operator() (int ix){
    return _data(ix, 0, 0);
}

const double& Grid::operator() (int ix) const{
    return _data(ix, 0, 0);
}

Grid& Grid::operator+= (const Grid &rhs){
    _data += rhs.data();
    return *this;
}

Grid& Grid::operator+= (const double rhs){
    _data+=rhs;
    return *this;
}

Grid& Grid::operator-= (const Grid &rhs){
    _data -= rhs.data();
    return *this;
}

Grid& Grid::operator-= (const double rhs){
    _data -= rhs;
    return *this;
}

Grid& Grid::operator*= (const Grid &rhs){
    _data *= rhs.data();
    return *this;
}

Grid& Grid::operator*= (const double rhs){
    _data *= rhs;
    return *this;
}

Grid& Grid::operator/= (const Grid &rhs){
    _data /= rhs.data();
    return *this;
}

Grid& Grid::operator/= (const double rhs){
    _data /= rhs;
    return *this;
}

const string Grid::name() const{
    return _name;
}

void Grid::set_name(const string name){
    _name = name;
}

const UnitCell Grid::uc() const{
    return _uc;
}

const int Grid::dim() const{
    return _uc.dim();
}

const int Grid::Lx() const{
    return _Lx;
}

const int Grid::Ly() const{
    return _Ly;
}

const int Grid::Lz() const{
    return _Lz;
}

const double Grid::lx() const{
    return _uc.lx();
}

const double Grid::ly() const{
    return _uc.ly();
}

const double Grid::lz() const{
    return _uc.lz();
}

const double Grid::dx() const{
    return lx() / Lx();
}

const double Grid::dy() const{
    return ly() / Ly();
}

const double Grid::dz() const{
    return lz() / Lz();
}

const double Grid::sum() const{
    return blitz::sum(_data);
}

const double Grid::abs_sum() const{
    return blitz::sum(blitz::abs(_data));
}

const double Grid::mean() const{
    return blitz::mean(_data);
}

const double Grid::mean(const blitz::Array<double, 3> data) const{
    return blitz::mean(data);
}

const double Grid::abs_mean() const{
    return blitz::mean(blitz::abs(_data));
}

const double Grid::min() const{
    return blitz::min(_data);
}

const double Grid::max() const{
    return blitz::max(_data);
}

const double Grid::abs_quadrature() const{
    blitz::Array<double, 3> data(_Lx, _Ly, _Lz);
    data = blitz::abs(_data);
    switch(dim()){
        case 1:
            return quadrature_1d(data);
        case 2:
            return quadrature_2d(data);
        case 3:
            return quadrature_3d(data);
    }
}

const double Grid::abs_quadrature(const blitz::Array<double, 3> data) const{
    blitz::Array<double, 3> data_abs(_Lx, _Ly, _Lz);
    data_abs = blitz::abs(data);
    switch(dim()){
        case 1:
            return quadrature_1d(data_abs);
        case 2:
            return quadrature_2d(data_abs);
        case 3:
            return quadrature_3d(data_abs);
    }
}

const double Grid::quadrature() const{
    switch(dim()){
        case 1:
            return quadrature_1d(_data);
        case 2:
            return quadrature_2d(_data);
        case 3:
            return quadrature_3d(_data);
    }
}

const double Grid::quadrature(const blitz::Array<double, 3> data) const{
    switch(dim()){
        case 1:
            return quadrature_1d(data);
        case 2:
            return quadrature_2d(data);
        case 3:
            return quadrature_3d(data);
    }
}

const double Grid::quadrature_1d(const blitz::Array<double, 3> data) const{
    if(_gtypex == GridType::REGULAR){
        return mean(data);
    }
    else if(_gtypex == GridType::CHEBYSHEV_GAUSS_LOBATTO){
        arma::vec v(data.data(), _Lx);
        Cheb cheb(_Lx);
        return 0.5 * cheb.quadrature_clencurt(v);
    }
}

/**
 * Currently, only y dimension are allowed to be Chebyshev-Gauss-Lobatto grid.
 */
const double Grid::quadrature_2d(const blitz::Array<double, 3> data) const{
    if(_gtypex != GridType::REGULAR){
        cerr<<"x dimension must be Regular."<<endl;
        exit(1);
    }

    if(_gtypey == GridType::REGULAR){        
        //cout << "Regular Grid" << endl;
        return mean(data);
    }
    else if(_gtypey == GridType::CHEBYSHEV_GAUSS_LOBATTO){
        blitz::firstIndex i;
        blitz::secondIndex j;
        blitz::thirdIndex k;
        blitz::Array<double, 2> dataz(_Lx, _Ly);
        dataz = blitz::mean(data, k);
        //cout<<"quadrature dataz ="<<dataz<<endl;
        blitz::Array<double, 1> dataxz(_Ly);
        dataxz = blitz::mean(dataz(j, i), j);
        arma::vec v(dataxz.data(), _Ly);
        //v.print("quadrature v =");
        Cheb cheb(_Ly);
        return 0.5 * cheb.quadrature_clencurt(v);
    }
}

/**
 * Currently, only z dimension are allowed to be Chebyshev-Gauss-Lobatto grid.
 */
const double Grid::quadrature_3d(const blitz::Array<double, 3> data) const{
    if(_gtypex != GridType::REGULAR || _gtypey != GridType::REGULAR){
        cerr<<"x and y dimensions must be Regular."<<endl;
        exit(1);
    }

    if(_gtypez == GridType::REGULAR){
        return mean(data);
    }
    else if(_gtypez == GridType::CHEBYSHEV_GAUSS_LOBATTO){
        blitz::firstIndex i;
        blitz::secondIndex j;
        blitz::thirdIndex k;
        blitz::Array<double, 2> datay(_Lx, _Lz);
        datay = blitz::mean(data(i, k, j), k);
        blitz::Array<double, 1> dataxy(_Lz);
        dataxy = blitz::mean(datay(j, i), j);
        arma::vec v(dataxy.data(), _Lz);
        Cheb cheb(_Lz);
        return 0.5 * cheb.quadrature_clencurt(v);
    }
}

const GridInitType Grid::grid_init_type() const{
    return _grid_init_type;
}

void Grid::save(const string file){
    CMatFile mat;
    mwSize dims3[3]={(mwSize)_Lx, (mwSize)_Ly, (mwSize)_Lz};

    blitz::Array<double,3> data(_Lx, _Ly, _Lz, blitz::fortranArray);
    data = _data;

    mat.matInit(file.c_str(), "u");
    if(!mat.queryStatus()){
        mat.matPut(_name, data.data(), data.size()*sizeof(double), 3, dims3,mxDOUBLE_CLASS,mxREAL);
    }
    else{
        cerr<<"error: cannot open MAT-file: "<<file<<endl;
        exit(1);
    }
    mat.matRelease();
}

const Grid Grid::diffx() const{
    switch(dim()){
        case 1:
            return diffx_1d();
        case 2:
            return diffx_2d();
        case 3:
            return diffx_3d();
    }
}

const Grid Grid::diffx_1d() const{
    Grid g(*this);
    double dx = this->dx();
    int ip, is;
    for(int i=0; i<_Lx; i++){
        ip = (i+1) % _Lx;
        is = i - 1;
        if(is < 0) is = _Lx - 1;
        // Central difference scheme
        g(i, 0, 0) = 0.5 * (_data(ip, 0, 0) - _data(is, 0, 0)) / dx;
    }
    return g;
}

const Grid Grid::diffx_2d() const{
    Grid g(*this);
    double dx = this->dx();
    int ip, is;
    for(int i=0; i<_Lx; i++)
      for(int j=0; j<_Ly; j++){
        ip = (i+1) % _Lx;
        is = i - 1;
        if(is < 0) is = _Lx - 1;
        // Central difference scheme
        g(i, j, 0) = 0.5 * (_data(ip, j, 0) - _data(is, j, 0)) / dx;
      }
    return g;
}

const Grid Grid::diffx_3d() const{
    Grid g(*this);
    double dx = this->dx();
    int ip, is;
    for(int i=0; i<_Lx; i++)
      for(int j=0; j<_Ly; j++)
        for(int k=0; k<_Lz; k++){
            ip = (i+1) % _Lx;
            is = i - 1;
            if(is < 0) is = _Lx - 1;
            // Central difference scheme
            g(i, j, k) = 0.5 * (_data(ip, j, k) - _data(is, j, k)) / dx;
        }
    return g;
}

const Grid Grid::diffy() const{
    switch(dim()){
        case 1:
            throw("1D Grid has no df/dy!");
        case 2:
            return diffy_2d();
        case 3:
            return diffy_3d();
        default:
            return diffy_3d();
    }
}

const Grid Grid::diffy_2d() const{
    Grid g(*this);
    double dy = this->dy();
    int jp, js;
    for(int i=0; i<_Lx; i++)
      for(int j=0; j<_Ly; j++){
        jp = (j+1) % _Ly;
        js = j - 1;
        if(js < 0) js = _Ly - 1;
        // Central difference scheme
        g(i, j, 0) = 0.5 * (_data(i, jp, 0) - _data(i, js, 0)) / dy;
        }
    return g;
}

const Grid Grid::diffy_3d() const{
    Grid g(*this);
    double dy = this->dy();
    int jp, js;
    for(int i=0; i<_Lx; i++)
      for(int j=0; j<_Ly; j++)
        for(int k=0; k<_Lz; k++){
            jp = (j+1) % _Ly;
            js = j - 1;
            if(js < 0) js = _Ly - 1;
            // Central difference scheme
            g(i, j, k) = 0.5 * (_data(i, jp, k) - _data(i, js, k)) / dy;
        }
    return g;
}

const Grid Grid::diffz() const{
    Grid g(*this);
    double dz = this->dz();
    int kp, ks;
    for(int i=0; i<_Lx; i++)
      for(int j=0; j<_Ly; j++)
        for(int k=0; k<_Lz; k++){
            kp = (k+1) % _Lz;
            ks = k - 1;
            if(ks < 0) ks = _Lz - 1;
            // Central difference scheme
            g(i, j, k) = 0.5 * (_data(i, j, kp) - _data(i, j, ks)) / dz;
        }
    return g;
}

const Grid Grid::diff2() const{
    if(dim() == 1){
        return diff2_1d();
    }

    if(dim() == 2){
        if(_uc.type_key() == CrystalSystemType::HEXAGONAL)
            return diff2_hex_2d();
        else
            return diff2_2d();
    }

    if(dim() == 3){
        if(_uc.type_key() == CrystalSystemType::HEXAGONAL)
            return diff2_hex_3d();
        else if(_uc.type_key() == CrystalSystemType::MONOCLINIC)
            return diff2_mono_3d();
        else
            return diff2_3d();
    }
}

const Grid Grid::diff2_1d() const{
    Grid g(*this);
    double dx = this->dx();
    int ip, is;
    for(int i=0; i<_Lx; i++){
        ip = (i+1) % _Lx;
        is = i - 1;
        if(is < 0) is = _Lx - 1;
        // Central difference scheme
        g(i, 0, 0) = 0.25 * (_data(ip, 0, 0) - _data(is, 0, 0))* \
                            (_data(ip, 0, 0) - _data(is, 0, 0))/dx/dx;
    }
    return g;
}

const Grid Grid::diff2_2d() const{
    Grid g(*this);
    double dx = this->dx();
    double dy = this->dy();
    int ip, jp, is, js;
    for(int i=0; i<_Lx; i++)
      for(int j=0; j<_Ly; j++){
        ip = (i+1) % _Lx;
        is = i - 1;
        if(is < 0) is = _Lx - 1;
        jp = (j+1) % _Ly;
        js = j - 1;
        if(js < 0) js = _Ly - 1;
        // Central difference scheme
        g(i, j, 0) = 0.25 * (_data(ip, j, 0) - _data(is, j, 0))* \
                     (_data(ip, j, 0) - _data(is, j, 0))/dx/dx + \
                     0.25 * (_data(i, jp, 0) - _data(i, js, 0))* \
                     (_data(i, jp, 0) - _data(i, js, 0))/dy/dy;
        }
    return g;
}

const Grid Grid::diff2_hex_2d() const{
    Grid px = this->diffx();
    Grid py = this->diffy();
    return (4.0/3.0) * (px*px + py*py + px*py);
}

const Grid Grid::diff2_3d() const{
    Grid g(*this);
    double dx = this->dx();
    double dy = this->dy();
    double dz = this->dz();
    int ip, jp, kp, is, js, ks;
    for(int i=0; i<_Lx; i++)
      for(int j=0; j<_Ly; j++)
        for(int k=0; k<_Lz; k++){
            ip = (i+1) % _Lx;
            is = i-1;
            if(is < 0) is = _Lx - 1;
            jp = (j+1) % _Ly;
            js = j - 1;
            if(js < 0) js = _Ly - 1;
            kp = (k+1) % _Lz;
            ks = k - 1;
            if(ks < 0) ks = _Lz - 1;
            // Central difference scheme
            g(i, j, k) = 0.25 * (_data(ip, j, k) - _data(is, j, k))* \
                         (_data(ip, j, k) - _data(is, j, k))/dx/dx + \
                       0.25 * (_data(i, jp, k) - _data(i, js, k))* \
                       (_data(i, jp, k) - _data(i, js, k))/dy/dy + \
                       0.25 * (_data(i, j, kp) - _data(i, j, ks))* \
                       (_data(i, j, kp) - _data(i, j, ks))/dz/dz;
        }
    return g;
}

const Grid Grid::diff2_hex_3d() const{
    Grid px = this->diffx();
    Grid py = this->diffy();
    Grid pz = this->diffz();
    return (4.0/3.0) * (px*px + py*py + px*py) + pz*pz;
}

const Grid Grid::diff2_mono_3d() const{
    Grid px = this->diffx();
    Grid py = this->diffy();
    Grid pz = this->diffz();
    double beta = _uc.beta();
    double sin2 = 1.0 / (sin(beta) * sin(beta));
    double cos2 = 2.0 * cos(beta);
    return sin2 * (px*px + pz*pz - cos2*px*pz) + pz*pz;
}

const int Grid::ngrids() const{
    return Lx() * Ly() * Lz();
}

const int Grid::size() const{
    return Lx() * Ly() * Lz();
}

const double Grid::volume() const{
    return lx() * ly() * lz();
}

const size_t Grid::seed() const{
    return _seed;
}

void Grid::init(const Config &param){
    _Lx = param.Lx();
    _Ly = param.Ly();
    _Lz = param.Lz();
    _ctype = param.ctype();
    _gtypex = param.gtypex();
    _gtypey = param.gtypey();
    _gtypez = param.gtypez();
    _grid_init_type = param.get_grid_init_type();
    //_phase_pattern=param.get_phase_pattern();
}

void Grid::set_value(const string file){
    CMatFile mat;

    _data.resize(_Lx, _Ly, _Lz);
    blitz::Array<double,3> data(_Lx, _Ly, _Lz, blitz::fortranArray);

    mat.matInit(file, "r");
    if(!mat.queryStatus()){
        mat.matGetArray(_name, data.data(), data.size() * sizeof(double));
        mat.matRelease();
        _data = data;
    }
    else{
        cerr<<"WARNING: cannot read file: "<<file<<endl;
        cerr<<"Use random initialization instead."<<endl;
        set_value(0, 0.0, 1.0);
    }
}

void Grid::set_value(const size_t seed, const double low, const double high){
    _data.resize(_Lx, _Ly, _Lz);
    if(seed == 0)
        _seed = static_cast<size_t>(time(NULL) + clock());
    else
        _seed = seed;
    Ran *pran = new Ran(_seed);
    double range = high - low;
    for(int i=0; i<_Lx; i++)
        for(int j=0; j<_Ly; j++)
            for(int k=0; k<_Lz; k++)
                _data(i, j, k) = low + range * pran->doub();
}

Grid operator+ (const Grid &lhs, const Grid &rhs){
    Grid tmp(lhs);
    tmp += rhs;
    return tmp;
}

Grid operator+ (const double lhs, const Grid &rhs){
    Grid tmp(rhs);
    tmp += lhs;
    return tmp;
}

Grid operator+ (const Grid &lhs, const double rhs){
    Grid tmp(lhs);
    tmp += rhs;
    return tmp;
}

Grid operator- (const Grid &lhs, const Grid &rhs){
    Grid tmp(lhs);
    tmp -= rhs;
    return tmp;
}

Grid operator- (const double lhs, const Grid &rhs){
    Grid tmp(rhs);
    tmp *= -1.0;
    tmp += lhs;
    return tmp;
}

Grid operator- (const Grid &lhs, const double rhs){
    Grid tmp(lhs);
    tmp -= rhs;
    return tmp;
}

Grid operator* (const Grid &lhs, const Grid &rhs){
    Grid tmp(lhs);
    tmp *= rhs;
    return tmp;
}

Grid operator* (const double lhs, const Grid &rhs){
    Grid tmp(rhs);
    tmp *= lhs;
    return tmp;
}

Grid operator* (const Grid &lhs, const double rhs){
    Grid tmp(lhs);
    tmp *= rhs;
    return tmp;
}

Grid operator/ (const Grid &lhs, const Grid &rhs){
    Grid tmp(lhs);
    tmp /= rhs;
    return tmp;
}

Grid operator/ (const double lhs, const Grid &rhs){
    Grid tmp(rhs);
    tmp.data() = lhs / rhs.data();
    return tmp;
}

Grid operator/ (const Grid &lhs, const double rhs){
    Grid tmp(lhs);
    tmp /= rhs;
    return tmp;
}

Grid exp(const Grid& g){
    Grid tmp(g);
    tmp.data() = exp(g.data());
    return tmp;
}

Grid log(const Grid& g){
    Grid tmp(g);
    tmp.data() = log(g.data());
    return tmp;
}
