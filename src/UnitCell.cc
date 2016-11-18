#include "UnitCell.h"
#include "Config.h"
#include "blitz/array.h"

UnitCell::UnitCell(const UnitCell &rhs):
                            _dim(rhs._dim),
                            _crystal_system_type(rhs._crystal_system_type),
                            _cell_param(rhs._cell_param){
    init();
}

UnitCell::UnitCell(const Config &cfg){
    init(cfg);
}

UnitCell::UnitCell(const int dim,
                   const CrystalSystemType cs,
                   const CellParam &cp):
                        _dim(dim),
                        _crystal_system_type(cs),
                        _cell_param(cp){
    init();
}

UnitCell& UnitCell::operator= (const UnitCell &rhs){
    _dim = rhs._dim;
    _crystal_system_type = rhs._crystal_system_type;
    _cell_param = rhs._cell_param;
    init();
    return *this;
}

CrystalSystemType UnitCell::type_key() const{
    return _crystal_system_type;
}

string UnitCell::type() const{
    return Config::get_crystal_system_type_string(_crystal_system_type);
}

const int UnitCell::dim() const{
    return _dim;
}

const double UnitCell::a() const{
    return _cell_param.a;
}

const double UnitCell::b() const{
    return _cell_param.b;
}

const double UnitCell::c() const{
    return _cell_param.c;
}

const double UnitCell::alpha() const{
    return _cell_param.alpha;
}

const double UnitCell::beta() const{
    return _cell_param.beta;
}

const double UnitCell::gamma() const{
    return _cell_param.gamma;
}

const blitz::TinyVector<double,3> UnitCell::l() const{
    return _l;
}

const double UnitCell::lx() const{
    return _l(0);
}

const double UnitCell::ly() const{
    return _l(1);
}

const double UnitCell::lz() const{
    return _l(2);
}

const blitz::TinyMatrix<double,3,3> UnitCell::h() const{
    return _h;
}

const blitz::TinyMatrix<double,3,3> UnitCell::g() const{
    return _g;
}

void UnitCell::init(const Config &cfg){
    _dim = cfg.dim();
    _crystal_system_type = cfg.get_crystal_system_type();
    _cell_param.a = cfg.a();
    _cell_param.b = cfg.b();
    _cell_param.c = cfg.c();
    _cell_param.alpha = cfg.alpha();
    _cell_param.beta = cfg.beta();
    _cell_param.gamma = cfg.gamma();

    init();
}

void UnitCell::init(){
    if(_dim == 1)
      calc_shape_matrix1();
    else if(_dim == 2)
      calc_shape_matrix2();
    else if(_dim == 3)
      calc_shape_matrix3();
}

void UnitCell::update(const CellParam &cp){
    _cell_param.a = cp.a;
    _cell_param.b = cp.b;
    _cell_param.c = cp.c;
    _cell_param.alpha = cp.alpha;
    _cell_param.beta = cp.beta;
    _cell_param.gamma = cp.gamma;

    init();
}

void UnitCell::save(const string file, int Lx, int Ly, int Lz) const{
    CMatFile mat;
    mwSize dims3[3]={(mwSize)Lx,(mwSize)Ly,(mwSize)Lz};

    blitz::Array<double,3> rx(Lx,Ly,Lz,blitz::fortranArray);
    blitz::Array<double,3> ry(Lx,Ly,Lz,blitz::fortranArray);
    blitz::Array<double,3> rz(Lx,Ly,Lz,blitz::fortranArray);
    rx = x(Lx,Ly,Lz);
    ry = y(Lx,Ly,Lz);
    rz = z(Lx,Ly,Lz);

    mat.matInit(file.c_str(),"u");
    if(!mat.queryStatus()){
        mat.matPut("x", rx.data(), rx.size()*sizeof(double), 3, dims3,mxDOUBLE_CLASS,mxREAL);
        mat.matPut("y", ry.data(), ry.size()*sizeof(double), 3, dims3,mxDOUBLE_CLASS,mxREAL);
        mat.matPut("z", rz.data(), rz.size()*sizeof(double), 3, dims3,mxDOUBLE_CLASS,mxREAL);
    }
    else{
        cerr<<"error: cannot open MAT-file: "<<file<<endl;
    }
    mat.matRelease();
}

void UnitCell::calc_shape_matrix1(){
    _a = _cell_param.a, .0, .0;

    _l = _cell_param.a, .0, .0;

    _h = _a(0), .0, .0,
            .0, 1., .0,
            .0, .0, 1.;

    double twopi = 2.0 * PI;
    _g = twopi/_a(0),    .0,    .0,
                  .0, twopi,    .0,
                  .0,    .0, twopi;
}

void UnitCell::calc_shape_matrix2(){
    double pi2 = PI / 2.0;
    double pi3 = 2.0 * PI / 3.0;
    if(_cell_param.a <= 0){
        cout<<"Bad cell parameters for genearal case!"<<endl;
        exit(1);
    }
    switch(_crystal_system_type){
        case CrystalSystemType::SQUARE:
            _cell_param.b = _cell_param.a;
            _cell_param.gamma = pi2;           
            break;
        case CrystalSystemType::RECTANGULAR:
            if(_cell_param.b <= 0){
                cout<<"a = "<<_cell_param.a;
                cout<<" b = "<<_cell_param.b;
                cout<<" c = "<<_cell_param.c<<endl;
                cout<<"Bad cell parameters for RECTANGULAR!"<<endl;
                exit(1);
            }
            _cell_param.gamma = pi2;           
            break;
        case CrystalSystemType::HEXAGONAL:
            _cell_param.b = _cell_param.a;
            _cell_param.gamma = pi3;            
            break;
        default:
            if(_cell_param.b <= 0 || _cell_param.gamma <= 0){
                cout<<" a = "<<_cell_param.a;
                cout<<" b = "<<_cell_param.b;
                cout<<" c = "<<_cell_param.c<<endl;
                cout<<"Bad cell parameters for default case!"<<endl;
                exit(1);
            }
            break;
    }

    _a = _cell_param.a, .0, .0;
    _b = _cell_param.b*cos(_cell_param.gamma), \
         _cell_param.b*sin(_cell_param.gamma), \
         .0;

    _l = _cell_param.a, _cell_param.b, .0;

    _h = _a(0), _b(0), .0,
            .0, _b(1), .0,
            .0,    .0, 1.;

    blitz::TinyVector<double,3> acb; // cross product of a and b
    acb = blitz::cross(_a,_b);
    double denom = blitz::dot(acb,acb);
    blitz::TinyVector<double,3> as, bs; // reciprocal a and b
    as = blitz::cross(_b,acb) / denom;
    bs = blitz::cross(acb,_a) / denom;
    double twopi = 2.0 * PI;
    _g = twopi*as(0), twopi*bs(0),    .0,
         twopi*as(1), twopi*bs(1),    .0,
                  .0,          .0, twopi;
}

void UnitCell::calc_shape_matrix3(){
    double pi2 = PI / 2.0;
    double pi3 = 2.0 * PI / 3.0;
    switch(_crystal_system_type){
        case CrystalSystemType::CUBIC:
            _cell_param.b = _cell_param.a;
            _cell_param.c = _cell_param.a;
            _cell_param.alpha = pi2;
            _cell_param.beta = pi2;
            _cell_param.gamma = pi2;
            break;
        case CrystalSystemType::TETRAGONAL:
            _cell_param.b = _cell_param.a;
            _cell_param.alpha = pi2;
            _cell_param.beta = pi2;
            _cell_param.gamma = pi2;
            break;
        case CrystalSystemType::ORTHORHOMBIC:
            _cell_param.alpha = pi2;
            _cell_param.beta = pi2;
            _cell_param.gamma = pi2;
            break;
        case CrystalSystemType::HEXAGONAL:
            _cell_param.b = _cell_param.a;
            _cell_param.alpha = pi2;
            _cell_param.beta = pi2;
            _cell_param.gamma = pi3;
            break;
        case CrystalSystemType::TRIGONAL:
            _cell_param.b = _cell_param.a;
            _cell_param.c = _cell_param.a;
            _cell_param.alpha = _cell_param.gamma;
            _cell_param.beta = _cell_param.gamma;
            break;
        case CrystalSystemType::MONOCLINIC:
            _cell_param.alpha = pi2;
            _cell_param.gamma = pi2;
            break;
        default:
            cerr<<"Unknown CrystalSystemType. Abort!"<<endl;
            exit(1);
    }

    _a = _cell_param.a, .0, .0;
    _b = _cell_param.b * cos(_cell_param.gamma), \
         _cell_param.b * sin(_cell_param.gamma), \
         .0;
    double cx = _cell_param.c * cos(_cell_param.beta);
    double cy = _cell_param.c * (cos(_cell_param.alpha) - \
                cos(_cell_param.beta) * cos(_cell_param.gamma));
    double cz = sqrt(_cell_param.c * _cell_param.c - cx * cx - cy * cy);
    _c = cx, cy, cz;

    _l = _cell_param.a, _cell_param.b, _cell_param.c;

    _h = _a(0), _b(0), _c(0),
            .0, _b(1), _c(1),
            .0,    .0, _c(2);

    blitz::TinyVector<double,3> bcc,cca,acb; // cross products
    bcc = blitz::cross(_b, _c);
    cca = blitz::cross(_c, _a);
    acb = blitz::cross(_a, _b);
    double denom = blitz::dot(_a, bcc);
    blitz::TinyVector<double,3> as, bs, cs; // reciprocal a and b
    as = bcc / denom;
    bs = cca / denom;
    cs = acb / denom;
    double twopi = 2.0 * PI;
    _g = twopi*as(0), twopi*bs(0), twopi*cs(0),
         twopi*as(1), twopi*bs(1), twopi*cs(1),
         twopi*as(2), twopi*bs(2), twopi*cs(2);
}

const blitz::Array<double,3> UnitCell::calc_kLaplacian(
                                                blitz::TinyVector<int,3> vN,
                                                double ds) const{
    return calc_kLaplacian(vN(0), vN(1), vN(2), ds);
}

/**
 * Calculate the Laplacian operator in reciprocal space for
 * general unit cell.
 *
 * To be compatible with previous code, we map Nc -> Lx, Nb -> Ly, Na -> Lz.
 * However, the shape matrix is in the order of a, b, and c,
 * so we must construct wave vector with (k,j,i) while k -> Na -> Lz,
 * j -> Nb -> Ly, i -> Na -> Lx.
 *
 * bare_q: the index for each dimension in reciprocal space.
 * q: the cartesian coordinates of any postion in the reciprocal space.
 *    It can be constructed from the shape matrix of the reciprocal space
 *    `g` and the position index `bare_q`.
 *
 * Return: the exp(-ds*|q|^2), where q is in the unit of Rg,
 * and ds in the unit of N (polymerization degree).
 */
const blitz::Array<double,3> UnitCell::calc_kLaplacian(int Na, int Nb, int Nc,
                                                       double ds) const{
    blitz::Array<double,3> q2(Na, Nb, Nc);
    blitz::TinyVector<double,3> bare_q;
    for(int i=0; i<Na; i++)
        for(int j=0; j<Nb; j++)
            for(int k=0; k<Nc; k++){
                bare_q = 1.0*i, 1.0*j, 1.0*k;
                q2(i, j, k) = calc_k2(bare_q, Na, Nb, Nc);
            }
    // q is in the unit of Rg, ds is in the unit of N.
    blitz::Array<double,3> ret(blitz::exp(-ds * q2));
    return ret;
}

/**
 * To resevere the symmetry, the wave vector with longest wavelength
 * should be choose. These can be achived by
 * shifting the wave vector to the first Brillouin Zone.
 *
 */
double UnitCell::calc_k2(const blitz::TinyVector<double,3> q,
                         int Na, int Nb, int Nc) const{
    blitz::TinyVector<double,3> bare_qt,qt;
    double q2;
    double q2_min = LARGE;

    for(int i=1; i>-2; i--)
      for(int j=1; j>-2; j--)
        for(int k=1; k>-2; k--){
            bare_qt = i*Na, j*Nb, k*Nc;
            bare_qt += q;
            qt = blitz::product(_g, bare_qt);
            q2 = blitz::dot(qt, qt);
            if(q2 < q2_min)
              q2_min = q2;
        }
    return q2_min;
}

/**
 * Calculate the Laplacian operator in reciprocal space for
 * orthogonal unit cell (square, rectangular, cubic, tetragonal,
 * orthorohmbic)
 *
 * Return: -|q|^2, where q is in the unit of Rg.
 */
const blitz::Array<double,3> UnitCell::calc_k2_orthogonal(
                                                    const int Lx,
                                                    const int Ly,
                                                    const int Lz) const{
    blitz::Array<double,3> k2(Lx, Ly, Lz);
    double lx = _l(0);
    double ly = _l(1);
    double lz = _l(2);
    int ri,rj,rk;
    double ccx = 0.0;
    double ccy = 0.0;
    double ccz = 0.0;
    switch(_dim){
        case 1:
            ccx = -4.0*PI*PI / (lx*lx);
            break;
        case 2:
            ccx = -4.0*PI*PI / (lx*lx);
            ccy = -4.0*PI*PI / (ly*ly);
            break;
        case 3:
            ccx = -4.0*PI*PI / (lx*lx);
            ccy = -4.0*PI*PI / (ly*ly);
            ccz = -4.0*PI*PI / (lz*lz);
            break;
    }
    for(int i=0; i<Lx; i++)
        for(int j=0; j<Ly; j++)
            for(int k=0; k<Lz; k++)
            {
                // FT(Laplace operator) -> k=2*PI*r/(delta*LX)
                if (i<Lx/2+1)
                    ri=i;
                else
                    ri=Lx-i;

                if (j<Ly/2+1)
                    rj=j;
                else
                    rj=Ly-j;

                if (k<Lz/2+1)
                    rk=k;
                else
                    rk=Lz-k;

                // NOTE: the negative sign is in ccx, ccy, and ccz
                k2(i,j,k) = ccx*ri*ri + ccy*rj*rj + ccz*rk*rk;
            }
    return k2;
}

/**
 * Calculate the Laplacian operator in reciprocal space for
 * orthogonal unit cell (square, rectangular, cubic, tetragonal,
 * orthorohmbic)
 *
 * Return: the exp(-ds*|q|^2), where q is in the unit of Rg,
 * and ds in the unit of N (polymerization degree).
 */
const blitz::Array<double,3> UnitCell::calc_kLaplacian_orthogonal(
                                                    const int Lx,
                                                    const int Ly,
                                                    const int Lz,
                                                    const double ds) const{
    // calc_k2_orthogonal returns -|q|^2.
    blitz::Array<double,3> ret(blitz::exp(ds*calc_k2_orthogonal(Lx, Ly, Lz)));
    return ret;
}

const blitz::Array<double,3> UnitCell::x(int Lx, int Ly, int Lz) const{
    blitz::Array<double,3> rx(Lx,Ly,Lz);
    blitz::TinyVector<double,3> r;
    for(int i=0;i<Lx;i++)
      for(int j=0;j<Ly;j++)
        for(int k=0;k<Lz;k++){
            blitz::TinyVector<double,3> bare_r(1.0*i/Lx, 1.0*j/Ly, 1.0*k/Lz);
            r = blitz::product(_h, bare_r);
            rx(i,j,k) = r(0);
        }
    return rx;
}

const blitz::Array<double,3> UnitCell::y(int Lx, int Ly, int Lz) const{
    blitz::Array<double,3> ry(Lx,Ly,Lz);
    blitz::TinyVector<double,3> r;
    for(int i=0;i<Lx;i++)
      for(int j=0;j<Ly;j++)
        for(int k=0;k<Lz;k++){
            blitz::TinyVector<double,3> bare_r(1.0*i/Lx, 1.0*j/Ly, 1.0*k/Lz);
            r = blitz::product(_h, bare_r);
            ry(i,j,k) = r(1);
        }
    return ry;
}

const blitz::Array<double,3> UnitCell::z(int Lx, int Ly, int Lz) const{
    blitz::Array<double,3> rz(Lx,Ly,Lz);
    blitz::TinyVector<double,3> r;
    for(int i=0;i<Lx;i++)
      for(int j=0;j<Ly;j++)
        for(int k=0;k<Lz;k++){
            blitz::TinyVector<double,3> bare_r(1.0*i/Lx, 1.0*j/Ly, 1.0*k/Lz);
            r = blitz::product(_h, bare_r);
            rz(i,j,k) = r(2);
        }
    return rz;
}

