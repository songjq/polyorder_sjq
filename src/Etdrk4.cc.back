#include "Etdrk4.h"
#include "common.h"
#include "Propagator.h"
#include "Grid.h"
#include "UnitCell.h"

#include "armadillo"
#include "CMatFile.h"
#include "Cheb.h"
#include "Boundary.h"
#include "fftw3.h"
//#include "blitz/Array.h"

#include <iostream>
using namespace std;

using namespace arma;

Etdrk4::Etdrk4(const Etdrk4 &rhs): _uc(rhs._uc), _dim(rhs._dim),
                                   _Lx(rhs._Lx), _Ly(rhs._Ly), _Lz(rhs._Lz),
                                   _ds(rhs._ds), _ctype(rhs._ctype),
                                   _lb(rhs._lb), _rb(rhs._rb),
                                   _stype(rhs._stype), _M(rhs._M),
                                   E1(rhs.E1), E2(rhs.E2), Q(rhs.Q),
                                   f1(rhs.f1), f2(rhs.f2), f3(rhs.f3),
                                   f4(rhs.f4), f5(rhs.f5),
                                   mE1(rhs.mE1), mE2(rhs.mE2), mQ(rhs.mQ),
                                   mf1(rhs.mf1), mf2(rhs.mf2), mf3(rhs.mf3),
                                   mf4(rhs.mf4), mf5(rhs.mf5){
    init_fftw();
}

Etdrk4::Etdrk4(const Config &cfg, const double ds,
               const ETDRK4SCHEME stype, /* = ETDRK4SCHEME::KROGSTAD */
               const uword M /* = 16 */
               ):_ds(ds), _stype(stype), _M(M){  
    _uc = UnitCell(cfg);
    _dim = cfg.dim();
    _Lx = cfg.Lx();
    _Ly = cfg.Ly();
    _Lz = cfg.Lz();
    _ctype = cfg.ctype();
    vec lbc = cfg.BC_coefficients_left();
    vec rbc = cfg.BC_coefficients_right();
    _lb = Boundary(lbc(0), lbc(1), lbc(2));
    _rb = Boundary(rbc(0), rbc(1), rbc(2));

    init_coefficients();
    init_fftw();
}

Etdrk4::Etdrk4(const UnitCell &uc, const uword dim,
               const uword Lx, const uword Ly, const uword Lz,
               const double ds, const ConfineType ctype,
               const Boundary &lb, const Boundary &rb,
               const ETDRK4SCHEME stype, /* = ETDRK4SCHEME::KROGSTAD */
               const uword M /* = 16 */ 
               ): _uc(uc), _dim(dim), _Lx(Lx), _Ly(Ly), _Lz(Lz), _ds(ds),
                  _ctype(ctype), _lb(lb), _rb(rb),
                  _stype(stype), _M(M){
    init_coefficients();
    init_fftw();
    //save("etdrk4_coeff.mat");
}

Etdrk4* Etdrk4::clone() const{
    return new Etdrk4(*this);
}

void Etdrk4::init_fftw(){
    if(_dim == 3){
        arma_in.set_size(_Lx, _Ly);
        arma_out.set_size(_Lx, _Ly);
        _fftw_in = reinterpret_cast<fftw_complex*>(arma_in.memptr());
        _fftw_out = reinterpret_cast<fftw_complex*>(arma_out.memptr());
        _p_forward = fftw_plan_dft_2d(_Lx, _Ly, _fftw_in, _fftw_out,
                                      FFTW_FORWARD, FFTW_MEASURE);
        _p_backward = fftw_plan_dft_2d(_Lx, _Ly, _fftw_in, _fftw_out,
                                       FFTW_BACKWARD, FFTW_MEASURE);
    }
}

/**
 * Evaluate \phi_l(tA) using complex contour integral methods
 * with hyperbolic contour.
 *
 * phi_l(z) = [phi_{l-1}(z) - phi_{l-1}(0)] / z
 * phi_0(z) = exp(z)
 *
 * For example:
 *      phi_1(z) = [exp(z) - 1] / z
 *      phi_2(z) = [exp(z) - z - 1] / z^2
 *      phi_3(z) = [exp(z) - z^2/2 - z - 1] / z^3
 *
 * A must be a square matrix.
 *
 * TESTED, 2014.5.28.
 *
 * REF:
 * 1. Notebook page 2013.07.05
 * 2. Schmelzer, T.; Trefethen, L. N.
 *    Electronic Transaction on Numerical Analysis, 2007, 29, 1-18.
 * 3. Weideman, J. A. C.; Trefethen, L. N.
 *    Mathematics of Computation, 2007, 76, 1341-1356.
 */
mat Etdrk4::phi(const unsigned int l, const mat &A,
                const double t, const uword M){
    uword N = A.n_cols;
    mat I = eye<mat>(N, N);
    cx_mat cphi(N, N);
    cphi.fill(fill::zeros);

    double alpha = 1.1721;
    double h = 1.0818 / M;
    double mu = 4.4921 * M / t;
    cx_double ci = cx_double(0, 1);
    cx_colvec u = h * ci * (linspace(0, 2*M, 2*M+1) - M);

    cx_colvec z = mu * (sin(u - alpha) + 1.0);
    cx_colvec v = cos(u - alpha);

    cx_colvec c(z.n_elem);
    if(l == 0)
        c = exp(t * z) % v;
    else
        c = exp(t * z) % v / pow(t*z, l);

    for(int i = 0; i < z.n_elem; i++)
        cphi += c(i) * inv(z(i) * I - A);

    return 0.5 * h * mu / PI * real(cphi);
}

void Etdrk4::init_coefficients(){
    if(_dim == 1)
        init_coefficients_1d();
    else if(_dim == 2)
        init_coefficients_2d();
    else if(_dim == 3)
        init_coefficients_3d();
}

void Etdrk4::init_coefficients_size(const uword N){
    E1.set_size(N, N);
    E2.set_size(N, N);
    Q.set_size(N, N);
    f1.set_size(N, N);
    f2.set_size(N, N);
    f3.set_size(N, N);
    if(_stype == ETDRK4SCHEME::KROGSTAD){
        f4.set_size(N, N);
        f5.set_size(N, N);
    }
}

void Etdrk4::init_coefficients_size(const uword N, const uword Nc){
    mE1.set_size(N, N, Nc);
    mE2.set_size(N, N, Nc);
    mQ.set_size(N, N, Nc);
    mf1.set_size(N, N, Nc);
    mf2.set_size(N, N, Nc);
    mf3.set_size(N, N, Nc);
    if(_stype == ETDRK4SCHEME::KROGSTAD){
        mf4.set_size(N, N, Nc);
        mf5.set_size(N, N, Nc);
    }
}

void Etdrk4::init_coefficients_1d(){
    Cheb cheb(_Lx);
    mat L = cheb.D2(_lb, _rb);
    double lx = _uc.a();
    L *= 4.0 / lx / lx;  //  Mapping from [-1, 1] to [0, lx]
    // DBC-DBC: N = Lx -2
    // DBC-RBC, RBC-DBC, DBC-NBC, NBC-DBC: N = Lx -1
    // RBC-RBC, RBC-NBC, NBC-RBC, NBC-NBC: N = Lx
    uword N = L.n_cols;

    init_coefficients_size(N);

    calc_coefficients(L, _ds, _M);
}

void Etdrk4::init_coefficients_2d(){
    Cheb cheb(_Ly);
    mat L = cheb.D2(_lb, _rb);
    double lx = _uc.a();
    double ly = _uc.b();
    L *= 4.0 / ly / ly;  //  Mapping from [-1, 1] to [0, ly]
    // DBC-DBC: N = Ly -2
    // DBC-RBC, RBC-DBC, DBC-NBC, NBC-DBC: N = Ly -1
    // RBC-RBC, RBC-NBC, NBC-RBC, NBC-NBC: N = Ly
    uword N = L.n_cols;
    mat I = eye<mat>(N, N);

    init_coefficients_size(N);  // for E1 etc.
    init_coefficients_size(N, _Lx);  // for mE1 etc.

    double kx, k2;
    for(uword i=0; i<_Lx; i++){
        if(i < _Lx/2 + 1)
            kx = i * (2 * PI / lx);
        else
            kx = (_Lx - i) * (2 * PI / lx);
        k2 = kx * kx;
        calc_coefficients(L - k2*I, _ds, _M);
        mE1.slice(i) = E1;
        mE2.slice(i) = E2;
        mQ.slice(i) = Q;
        mf1.slice(i) = f1;
        mf2.slice(i) = f2;
        mf3.slice(i) = f3;
        if(_stype == ETDRK4SCHEME::KROGSTAD){
            mf4.slice(i) = f4;
            mf5.slice(i) = f5;
        }
    }
}

void Etdrk4::init_coefficients_3d(){
    Cheb cheb(_Lz);
    mat L = cheb.D2(_lb, _rb);
    double lx = _uc.a();
    double ly = _uc.b();
    double lz = _uc.c();
    L *= 4.0 / lz / lz;  //  Mapping from [-1, 1] to [0, lz]
    // DBC-DBC: N = Lz -2
    // DBC-RBC, RBC-DBC, DBC-NBC, NBC-DBC: N = Lz -1
    // RBC-RBC, RBC-NBC, NBC-RBC, NBC-NBC: N = Lz
    uword N = L.n_cols;
    mat I = eye<mat>(N, N);

    init_coefficients_size(N);
    init_coefficients_size(N, _Lx*_Ly);

    double kx, ky, k2;
    uword si; // cube slice index = i + _Lx * j
    for(uword i=0; i<_Lx; i++)
        for(uword j=0; j<_Ly; j++){
            if(i < _Lx/2 + 1)
                kx = i * (2 * PI / lx);
            else
                kx = (_Lx - i) * (2 * PI / lx);
            if(j < _Ly/2 + 1)
                ky = j * (2 * PI / ly);
            else
                ky = (_Ly - j) * (2 * PI / ly);
            k2 = kx * kx + ky * ky;
            calc_coefficients(L - k2*I, _ds, _M);
            si = i + _Lx * j;
            mE1.slice(si) = E1;
            mE2.slice(si) = E2;
            mQ.slice(si) = Q;
            mf1.slice(si) = f1;
            mf2.slice(si) = f2;
            mf3.slice(si) = f3;
            if(_stype == ETDRK4SCHEME::KROGSTAD){
                mf4.slice(si) = f4;
                mf5.slice(si) = f5;
            }
        }
}

/**
 * Evaluate etdrk4 coefficients by complex contour integral
 * using hyperbolic contour for both Cox-Matthews and Krogstad schemes.
 *
 * TESTED, 2014.5.28.
 *
 * REF:
 * 1. Schmelzer, T.; Trefethen, L. N.
 *    Electronic Transaction on Numerical Analysis, 2007, 29, 1-18.
 * 2. Weideman, J. A. C.; Trefethen, L. N.
 *    Mathematics of Computation, 2007, 76, 1341-1356.
 * 3. Trefethen, L. N.; Weideman, J. A. C.; Schmelzer, T.
 *    BIT Numer. Math. 2006, 46, 653.
 */
void Etdrk4::calc_coefficients(const mat &L, const double t, const uword M){
    double h = t;

    E1 = phi(0, h*L, 1.0, _M);
    E2 = phi(0, 0.5*h*L, 1.0, _M);
    Q = 0.5 * h * phi(1, 0.5*h*L, 1.0, _M);
    mat phi1 = phi(1, h*L, 1.0, _M);
    mat phi2 = phi(2, h*L, 1.0, _M);
    mat phi3 = phi(3, h*L, 1.0, _M);
    if(_stype == ETDRK4SCHEME::COX){
        f1 = h * (phi1 - 3*phi2 + 4*phi3);
        f2 = h * 2 * (phi2 - 2*phi2);
        f3 = h * (4*phi3 - phi2);
    }
    else if(_stype == ETDRK4SCHEME::KROGSTAD){
        f1 = h * phi(2, 0.5*h*L, 1.0, _M);
        f2 = h * phi1;
        f3 = h * 2 * phi2;
        f4 = h * (4*phi3 - phi2);
        f5 = h * (-4*phi3);
    }
}

void Etdrk4::save(const string file /* = "etdrk4_coeff.mat" */){
    CMatFile mat;
    mwSize N = (mwSize) E1.n_cols;
    mwSize n_bytes = N * N * sizeof(double);
    mwSize dims2[2] = {N, N};

    mat.matInit(file.c_str(), "u");
    if(!mat.queryStatus()){
        mat.matPut("E1", E1.memptr(), n_bytes, 2, dims2,
                   mxDOUBLE_CLASS, mxREAL);
        mat.matPut("E2", E2.memptr(), n_bytes, 2, dims2,
                   mxDOUBLE_CLASS, mxREAL);
        mat.matPut("Q", Q.memptr(), n_bytes, 2, dims2,
                   mxDOUBLE_CLASS, mxREAL);
        mat.matPut("f1", f1.memptr(), n_bytes, 2, dims2,
                   mxDOUBLE_CLASS, mxREAL);
        mat.matPut("f2", f2.memptr(), n_bytes, 2, dims2,
                   mxDOUBLE_CLASS, mxREAL);
        mat.matPut("f3", f3.memptr(), n_bytes, 2, dims2,
                   mxDOUBLE_CLASS, mxREAL);
        if(_stype == ETDRK4SCHEME::KROGSTAD){
            mat.matPut("f4", f4.memptr(), n_bytes, 2, dims2,
                       mxDOUBLE_CLASS, mxREAL);
            mat.matPut("f5", f5.memptr(), n_bytes, 2, dims2,
                       mxDOUBLE_CLASS, mxREAL);
        }
        mat.matRelease();
    }

}

Etdrk4::~Etdrk4(){
    if(_dim == 3){
        fftw_destroy_plan(_p_forward);
        fftw_destroy_plan(_p_backward);
    }
}

/**
 * The MDE is
 *          dq/ds = Lq - wq
 * Note that the input gw is just w, we must add negative sign here.
 */
void Etdrk4::solve(Propagator &pq, const Grid &gw){
    uword Ns = pq.len();
    if(_dim == 1){
        colvec w(gw.data().data(), _Lx);
        colvec u(pq[0].data(), _Lx);
        mat q;
        solve(u, -w, Ns, q);
        blitz::Array<double, 4> qb(q.memptr(), blitz::shape(Ns, _Lx, 1, 1),
                                   blitz::neverDeleteData);
        pq.qs() = qb;
    }
    else if(_dim == 2){
        // No special treatment for storage order.
        // ETDRK4 handle row major data internally.
        mat w(gw.data().data(), _Ly, _Lx);
        mat u(pq[0].data(), _Ly, _Lx);
        cube q;
        solve(u, -w, Ns, q);
        blitz::Array<double, 4> qb(q.memptr(), blitz::shape(Ns, _Lx, _Ly, 1),
                                   blitz::neverDeleteData);
        pq.qs() = qb;
    }
    else if(_dim == 3){
        // Convert to column-major storage order in blitz
        blitz::Array<double, 3> b_col(_Lx, _Ly, _Lz,
                                      blitz::ColumnMajorArray<3>());
        b_col = gw.data();
        // Transfer data from blitz to arma
        cube w(b_col.data(), _Lx, _Ly, _Lz);
        b_col = pq[0];
        cube u(b_col.data(), _Lx, _Ly, _Lz);
        // Solve
        field<cube> q;
        solve(u, -w, Ns, q);
        // Transfer data back from arma to blitz
        blitz::Range all = blitz::Range::all();
        for(uword s=0; s<Ns; s++){
            blitz::Array<double, 3> ut_col(q(s).memptr(),
                                           blitz::shape(_Lx, _Ly, _Lz),
                                           blitz::neverDeleteData,
                                           blitz::ColumnMajorArray<3>());
            pq[s] = ut_col;
        }
    }
}

/**
 * 1D Cox-Matthews scheme, col major implementation.
 *
 */
mat Etdrk4::scheme_cox(colvec &u, const colvec &w, const uword Ns){
    uword N = u.n_elem;
    mat q(N, Ns);
    q.col(0) = u;
    colvec Nu(N);
    colvec a(N);
    colvec Na(N);
    colvec b(N);
    colvec Nb(N);
    colvec c(N);
    colvec Nc(N);
    for(uword i=1; i<Ns; i++){
        Nu = w % u;
        a = E2 * u + Q * Nu;
        Na = w % a;
        b = E2 * u + Q * Na;
        Nb = w % b;
        c = E2 * a + Q * (2*Nb - Nu);
        Nc = w % c;
        u = E1 * u + f1 * Nu + f2 * (Na + Nb) + f3 * Nc;
        q.col(i) = u;
    }
    return q;
}

/**
 * 1D Cox-Matthews scheme, row major implementation.
 *
 */
mat Etdrk4::scheme_cox(rowvec &u, const rowvec &w, const uword Ns){
    uword N = u.n_elem;
    mat q(Ns, N);
    q.row(0) = u;
    colvec v = u.t();
    colvec cw = w.t();
    colvec Nu(N);
    colvec a(N);
    colvec Na(N);
    colvec b(N);
    colvec Nb(N);
    colvec c(N);
    colvec Nc(N);
    for(uword i=1; i<Ns; i++){
        Nu = cw % v;
        a = E2 * v + Q * Nu;
        Na = cw % a;
        b = E2 * v + Q * Na;
        Nb = cw % b;
        c = E2 * a + Q * (2*Nb - Nu);
        Nc = cw % c;
        u = E1 * v + f1 * Nu + f2 * (Na + Nb) + f3 * Nc;
        q.row(i) = v.t();
    }
    return q;
}

/**
 * 1D Krogstad scheme, col major implementation.
 *
 * runtime less than 1% best than row major implementation.
 *
 * TESTED, 2014.5.29.
 */
mat Etdrk4::scheme_krogstad(colvec &u, const colvec &w, const uword Ns){
    uword N = u.n_elem;
    mat q(N, Ns);
    q.col(0) = u;
    colvec Nu(N);
    colvec a(N);
    colvec Na(N);
    colvec b(N);
    colvec Nb(N);
    colvec c(N);
    colvec Nc(N);
    for(uword i=1; i<Ns; i++){
        Nu = w % u;
        a = E2 * u + Q * Nu;
        Na = w % a;
        b = a + f1 * (Na - Nu);
        Nb = w % b;
        c = E1 * u + f2 * Nu + f3 * (Nb - Nu);
        Nc = w % c;
        u = c + f3 * Na + f4 * (Nu + Nc) + f5 * (Na + Nb);
        q.col(i) = u;
    }
    return q;
}

/**
 * 1D Krogstad scheme, row major implementation.
 *
 * Runtime less than 1% worse than col major implementation.
 *
 * TESTED, 2014.5.29.
 */
mat Etdrk4::scheme_krogstad(rowvec &u, const rowvec &w, const uword Ns){
    uword N = u.n_elem;
    mat q(Ns, N);
    q.row(0) = u;
    colvec v = u.t();
    colvec cw = w.t();
    colvec Nu(N);
    colvec a(N);
    colvec Na(N);
    colvec b(N);
    colvec Nb(N);
    colvec c(N);
    colvec Nc(N);
    for(uword i=1; i<Ns; i++){
        Nu = cw % v;
        a = E2 * v + Q * Nu;
        Na = cw % a;
        b = a + f1 * (Na - Nu);
        Nb = cw % b;
        c = E1 * v + f2 * Nu + f3 * (Nb - Nu);
        Nc = cw % c;
        v = c + f3 * Na + f4 * (Nu + Nc) + f5 * (Na + Nb);
        q.row(i) = v.t();
    }
    return q;
}

/**
 * 2D Cox-Matthews scheme.
 * The size of u and w is Ly x Lx, where y is the confined dimension.
 */
cube Etdrk4::scheme_cox(mat &u, const mat &w, const uword Ns){

    /** NOT IMPLEMENTED **/

}

/**
 * 2D Krogstad scheme.
 * The size of u and w is Ly x Lx, where y is the confined dimension.
 *
 * TESTED, 2014.5.30
 */
cube Etdrk4::scheme_krogstad(mat &u, const mat &w, const uword Ns){
    uword Ny = u.n_rows;
    uword Nx = u.n_cols;
    cube q(Ny, Nx, Ns);
    q.slice(0) = u;
    mat Nu(Ny, Nx);
    mat a(Ny, Nx);
    mat Na(Ny, Nx);
    mat b(Ny, Nx);
    mat Nb(Ny, Nx);
    mat c(Ny, Nx);
    mat Nc(Ny, Nx);
    cx_mat uk = fft(u.st()).st();
    cx_mat Nuk(Ny, Nx);
    cx_mat ak(Ny, Nx);
    cx_mat Nak(Ny, Nx);
    cx_mat bk(Ny, Nx);
    cx_mat Nbk(Ny, Nx);
    cx_mat ck(Ny, Nx);
    cx_mat Nck(Ny, Nx);
    for(uword s=1; s<Ns; s++){
        Nu = w % u;
        Nuk = fft(Nu.st()).st();
        for(uword i=0; i<Nx; i++)
            ak.col(i) = mE2.slice(i) * uk.col(i) + mQ.slice(i) * Nuk.col(i);
        a = real(ifft(ak.st())).st();
        Na = w % a;
        Nak = fft(Na.st()).st();
        for(uword i=0; i<Nx; i++)
            bk.col(i) = ak.col(i) + mf1.slice(i) * (Nak.col(i) - Nuk.col(i));
        b = real(ifft(bk.st())).st();
        Nb = w % b;
        Nbk = fft(Nb.st()).st();
        for(uword i=0; i<Nx; i++)
            ck.col(i) = mE1.slice(i) * uk.col(i)
                        + mf2.slice(i) * Nuk.col(i)
                        + mf3.slice(i) * (Nbk.col(i) - Nuk.col(i));
        c = real(ifft(ck.st())).st();
        Nc = w % c;
        Nck = fft(Nc.st()).st();
        for(uword i=0; i<Nx; i++)
            uk.col(i) = ck.col(i) + mf3.slice(i) * Nak.col(i)
                        + mf4.slice(i) * (Nuk.col(i) + Nck.col(i))
                        + mf5.slice(i) * (Nak.col(i) + Nbk.col(i));
        u = real(ifft(uk.st())).st();
        q.slice(s) = u;
    }
    return q;
}

/**
 * 3D Cox-Matthews scheme.
 * The size of u and w is Lx x Ly x Lz, where z is the confined dimension.
 */
field<cube> Etdrk4::scheme_cox(cube &u, const cube &w, const uword Ns){
}

/**
 * 3D Krogstad scheme, using Armadillo internal FFT.
 * The size of u and w is Lx x Ly x Lz, where z is the confined dimension.
 *
 * TESTED, 2014-5-31.
 */
field<cube> Etdrk4::scheme_krogstad_armafft(cube &u, const cube &w,
                                            const uword Ns){
    uword Nx = u.n_rows;
    uword Ny = u.n_cols;
    uword Nz = u.n_slices;

    field<cube> q(Ns);
    q(0) = u;

    cube Nu(Nx, Ny, Nz);
    cube a(Nx, Ny, Nz);
    cube Na(Nx, Ny, Nz);
    cube b(Nx, Ny, Nz);
    cube Nb(Nx, Ny, Nz);
    cube c(Nx, Ny, Nz);
    cube Nc(Nx, Ny, Nz);

    cx_cube uk(Nx, Ny, Nz);
    for(uword k=0; k<Nz; k++)
        uk.slice(k) = fft2(u.slice(k));
    cx_cube Nuk(Nx, Ny, Nz);
    cx_cube ak(Nx, Ny, Nz);
    cx_cube Nak(Nx, Ny, Nz);
    cx_cube bk(Nx, Ny, Nz);
    cx_cube Nbk(Nx, Ny, Nz);
    cx_cube ck(Nx, Ny, Nz);
    cx_cube Nck(Nx, Ny, Nz);

    for(uword s=1; s<Ns; s++){
        Nu = w % u;
        for(uword k=0; k<Nz; k++)
            Nuk.slice(k) = fft2(Nu.slice(k));
        for(uword i=0; i<Nx; i++)
            for(uword j=0; j<Ny; j++){
                uword si = i + Nx * j;
                cx_colvec vuk = uk.tube(i, j);
                cx_colvec vNuk = Nuk.tube(i, j);
                cx_colvec vak = mE2.slice(si) * vuk + mQ.slice(si) * vNuk;
                for(uword k=0; k<vak.n_elem; k++)
                    ak(i, j, k) = vak(k);
            }

        for(uword k=0; k<Nz; k++)
            a.slice(k) = real(ifft2(ak.slice(k)));
        Na = w % a;
        for(uword k=0; k<Nz; k++)
            Nak.slice(k) = fft2(Na.slice(k));
        for(uword i=0; i<Nx; i++)
            for(uword j=0; j<Ny; j++){
                uword si = i + Nx * j;
                cx_colvec vak = ak.tube(i, j);
                cx_colvec vNak = Nak.tube(i, j);
                cx_colvec vNuk = Nuk.tube(i, j);
                cx_colvec vbk = vak + mf1.slice(si) * (vNak - vNuk);
                for(uword k=0; k<vbk.n_elem; k++)
                    bk(i, j, k) = vbk(k);
            }

        for(uword k=0; k<Nz; k++)
            b.slice(k) = real(ifft2(bk.slice(k)));
        Nb = w % b;
        for(uword k=0; k<Nz; k++)
            Nbk.slice(k) = fft2(Nb.slice(k));
        for(uword i=0; i<Nx; i++)
            for(uword j=0; j<Ny; j++){
                uword si = i + Nx * j;
                cx_colvec vuk = uk.tube(i, j);
                cx_colvec vNuk = Nuk.tube(i, j);
                cx_colvec vNbk = Nbk.tube(i, j);
                cx_colvec vck = mE1.slice(si) * vuk
                                + mf2.slice(si) * vNuk
                                + mf3.slice(si) * (vNbk - vNuk);
                for(uword k=0; k<vck.n_elem; k++)
                    ck(i, j, k) = vck(k);
            }

        for(uword k=0; k<Nz; k++)
            c.slice(k) = real(ifft2(ck.slice(k)));
        Nc = w % c;
        for(uword k=0; k<Nz; k++)
            Nck.slice(k) = fft2(Nc.slice(k));
        for(uword i=0; i<Nx; i++)
            for(uword j=0; j<Ny; j++){
                uword si = i + Nx * j;
                cx_colvec vck = ck.tube(i, j);
                cx_colvec vNuk = Nuk.tube(i, j);
                cx_colvec vNak = Nak.tube(i, j);
                cx_colvec vNbk = Nbk.tube(i, j);
                cx_colvec vNck = Nck.tube(i, j);
                cx_colvec vuk = vck + mf3.slice(si) * vNak
                                + mf4.slice(si) * (vNuk + vNck)
                                + mf5.slice(si) * (vNak + vNbk);
                for(uword k=0; k<vuk.n_elem; k++)
                    uk(i, j, k) = vuk(k);
            }

        for(uword k=0; k<Nz; k++)
            u.slice(k) = real(ifft2(uk.slice(k)));
        q(s) = u;
    }
    return q;
}

/**
 * 3D Krogstad scheme, using FFTW.
 * The size of u and w is Lx x Ly x Lz, where z is the confined dimension.
 *
 * This implementation is about 10% faster than scheme_krogstad_armafft.
 *
 * TESTED, 2014-5-31.
 */
field<cube> Etdrk4::scheme_krogstad_fftw_raw(cube &u, const cube &w,
                                             const uword Ns){
    uword Nx = u.n_rows;
    uword Ny = u.n_cols;
    uword Nz = u.n_slices;

    field<cube> q(Ns);
    q(0) = u;

    cube Nu(Nx, Ny, Nz);
    cx_cube ca(Nx, Ny, Nz);
    cube Na(Nx, Ny, Nz);
    cx_cube cb(Nx, Ny, Nz);
    cube Nb(Nx, Ny, Nz);
    cx_cube cc(Nx, Ny, Nz);
    cube Nc(Nx, Ny, Nz);

    cx_cube cu(Nx, Ny, Nz);
    cx_cube uk(Nx, Ny, Nz);
    uk.fill(0);
    uk.set_real(u);
    for(uword k=0; k<Nz; k++){
        fftw_complex *in = reinterpret_cast<fftw_complex*>
                           (uk.slice(k).memptr());
        fftw_plan plan = fftw_plan_dft_2d(Nx, Ny, in, in,
                                          FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(plan);
    }
    cx_cube Nuk(Nx, Ny, Nz);
    cx_cube ak(Nx, Ny, Nz);
    cx_cube Nak(Nx, Ny, Nz);
    cx_cube bk(Nx, Ny, Nz);
    cx_cube Nbk(Nx, Ny, Nz);
    cx_cube ck(Nx, Ny, Nz);
    cx_cube Nck(Nx, Ny, Nz);

    for(uword s=1; s<Ns; s++){
        Nu = w % u;
        Nuk.fill(0);
        Nuk.set_real(Nu);
        for(uword k=0; k<Nz; k++){
            fftw_complex *in = reinterpret_cast<fftw_complex*>
                               (Nuk.slice(k).memptr());
            fftw_plan plan = fftw_plan_dft_2d(Nx, Ny, in, in,
                                              FFTW_FORWARD, FFTW_ESTIMATE);
            fftw_execute(plan);
        }
        for(uword i=0; i<Nx; i++)
            for(uword j=0; j<Ny; j++){
                uword si = i + Nx * j;
                cx_colvec vuk = uk.tube(i, j);
                cx_colvec vNuk = Nuk.tube(i, j);
                cx_colvec vak = mE2.slice(si) * vuk + mQ.slice(si) * vNuk;
                for(uword k=0; k<vak.n_elem; k++)
                    ak(i, j, k) = vak(k);
            }

        for(uword k=0; k<Nz; k++){
            fftw_complex *in = reinterpret_cast<fftw_complex*>
                               (ak.slice(k).memptr());
            fftw_complex *out = reinterpret_cast<fftw_complex*>
                                (ca.slice(k).memptr());
            fftw_plan plan = fftw_plan_dft_2d(Nx, Ny, in, out,
                                              FFTW_BACKWARD, FFTW_ESTIMATE);
            fftw_execute(plan);
        }
        Na = w % real(ca)/(Nx*Ny);
        Nak.fill(0);
        Nak.set_real(Na);
        for(uword k=0; k<Nz; k++){
            fftw_complex *in = reinterpret_cast<fftw_complex*>
                               (Nak.slice(k).memptr());
            fftw_plan plan = fftw_plan_dft_2d(Nx, Ny, in, in,
                                              FFTW_FORWARD, FFTW_ESTIMATE);
            fftw_execute(plan);
        }
        for(uword i=0; i<Nx; i++)
            for(uword j=0; j<Ny; j++){
                uword si = i + Nx * j;
                cx_colvec vak = ak.tube(i, j);
                cx_colvec vNak = Nak.tube(i, j);
                cx_colvec vNuk = Nuk.tube(i, j);
                cx_colvec vbk = vak + mf1.slice(si) * (vNak - vNuk);
                for(uword k=0; k<vbk.n_elem; k++)
                    bk(i, j, k) = vbk(k);
            }

        for(uword k=0; k<Nz; k++){
            fftw_complex *in = reinterpret_cast<fftw_complex*>
                               (bk.slice(k).memptr());
            fftw_complex *out = reinterpret_cast<fftw_complex*>
                                (cb.slice(k).memptr());
            fftw_plan plan = fftw_plan_dft_2d(Nx, Ny, in, out,
                                              FFTW_BACKWARD, FFTW_ESTIMATE);
            fftw_execute(plan);
        }
        Nb = w % real(cb)/(Nx*Ny);
        Nbk.fill(0);
        Nbk.set_real(Nb);
        for(uword k=0; k<Nz; k++){
            fftw_complex *in = reinterpret_cast<fftw_complex*>
                               (Nbk.slice(k).memptr());
            fftw_plan plan = fftw_plan_dft_2d(Nx, Ny, in, in,
                                              FFTW_FORWARD, FFTW_ESTIMATE);
            fftw_execute(plan);
        }
        for(uword i=0; i<Nx; i++)
            for(uword j=0; j<Ny; j++){
                uword si = i + Nx * j;
                cx_colvec vuk = uk.tube(i, j);
                cx_colvec vNuk = Nuk.tube(i, j);
                cx_colvec vNbk = Nbk.tube(i, j);
                cx_colvec vck = mE1.slice(si) * vuk
                                + mf2.slice(si) * vNuk
                                + mf3.slice(si) * (vNbk - vNuk);
                for(uword k=0; k<vck.n_elem; k++)
                    ck(i, j, k) = vck(k);
            }

        for(uword k=0; k<Nz; k++){
            fftw_complex *in = reinterpret_cast<fftw_complex*>
                               (ck.slice(k).memptr());
            fftw_complex *out = reinterpret_cast<fftw_complex*>
                                (cc.slice(k).memptr());
            fftw_plan plan = fftw_plan_dft_2d(Nx, Ny, in, out,
                                              FFTW_BACKWARD, FFTW_ESTIMATE);
            fftw_execute(plan);
        }
        Nc = w % real(cc)/(Nx*Ny);
        Nck.fill(0);
        Nck.set_real(Nc);
        for(uword k=0; k<Nz; k++){
            fftw_complex *in = reinterpret_cast<fftw_complex*>
                               (Nck.slice(k).memptr());
            fftw_plan plan = fftw_plan_dft_2d(Nx, Ny, in, in,
                                              FFTW_FORWARD, FFTW_ESTIMATE);
            fftw_execute(plan);
        }
        for(uword i=0; i<Nx; i++)
            for(uword j=0; j<Ny; j++){
                uword si = i + Nx * j;
                cx_colvec vck = ck.tube(i, j);
                cx_colvec vNuk = Nuk.tube(i, j);
                cx_colvec vNak = Nak.tube(i, j);
                cx_colvec vNbk = Nbk.tube(i, j);
                cx_colvec vNck = Nck.tube(i, j);
                cx_colvec vuk = vck + mf3.slice(si) * vNak
                                + mf4.slice(si) * (vNuk + vNck)
                                + mf5.slice(si) * (vNak + vNbk);
                for(uword k=0; k<vuk.n_elem; k++)
                    uk(i, j, k) = vuk(k);
            }

        for(uword k=0; k<Nz; k++){
            fftw_complex *in = reinterpret_cast<fftw_complex*>
                               (uk.slice(k).memptr());
            fftw_complex *out = reinterpret_cast<fftw_complex*>
                                (cu.slice(k).memptr());
            fftw_plan plan = fftw_plan_dft_2d(Nx, Ny, in, out,
                                              FFTW_BACKWARD, FFTW_ESTIMATE);
            fftw_execute(plan);
        }
        u = real(cu)/(Nx*Ny);
        q(s) = u;
    }
    return q;
}

/**
 * 3D Krogstad scheme, using FFTW.
 * The size of u and w is Lx x Ly x Lz, where z is the confined dimension.
 *
 * In current implmentation, for Lx=Ly=32, Lz=33, Ns=101, single solution
 *      This:                       4.4 secs
 *      scheme_krogstad_fftw_raw    4.8 secs
 *      scheme_krogstad_armafft:    5.4 secs
 *
 * TESTED, 2014-06-01.
 *
 */
field<cube> Etdrk4::scheme_krogstad(cube &u, const cube &w, const uword Ns){
    uword Nx = u.n_rows;  // must be equal to _Lx
    uword Ny = u.n_cols;  // must be equal to _Ly
    uword Nz = u.n_slices;

    // Copy initial solution to q
    field<cube> q(Ns);
    q(0) = u;

    cube Nu(Nx, Ny, Nz);
    cx_cube ca(Nx, Ny, Nz);
    cube Na(Nx, Ny, Nz);
    cx_cube cb(Nx, Ny, Nz);
    cube Nb(Nx, Ny, Nz);
    cx_cube cc(Nx, Ny, Nz);
    cube Nc(Nx, Ny, Nz);

    cx_cube cu(Nx, Ny, Nz);
    cx_cube uk(Nx, Ny, Nz);
    cu.fill(0);
    cu.set_real(u);
    for(uword k=0; k<Nz; k++){
        arma_in = cu.slice(k);
        fftw_execute(_p_forward);
        uk.slice(k) = arma_out;
    }
    cx_cube Nuk(Nx, Ny, Nz);
    cx_cube ak(Nx, Ny, Nz);
    cx_cube Nak(Nx, Ny, Nz);
    cx_cube bk(Nx, Ny, Nz);
    cx_cube Nbk(Nx, Ny, Nz);
    cx_cube ck(Nx, Ny, Nz);
    cx_cube Nck(Nx, Ny, Nz);

    for(uword s=1; s<Ns; s++){
        Nu = w % u;
        Nuk.fill(0);
        Nuk.set_real(Nu);
        for(uword k=0; k<Nz; k++){
            arma_in = Nuk.slice(k);
            fftw_execute(_p_forward);
            Nuk.slice(k) = arma_out;
        }
        for(uword i=0; i<Nx; i++)
            for(uword j=0; j<Ny; j++){
                uword si = i + Nx * j;
                cx_colvec vuk = uk.tube(i, j);
                cx_colvec vNuk = Nuk.tube(i, j);
                cx_colvec vak = mE2.slice(si) * vuk + mQ.slice(si) * vNuk;
                for(uword k=0; k<vak.n_elem; k++)
                    ak(i, j, k) = vak(k);
            }

        for(uword k=0; k<Nz; k++){
            arma_in = ak.slice(k);
            fftw_execute(_p_backward);
            ca.slice(k) = arma_out;
        }
        Na = w % real(ca)/(Nx*Ny);
        Nak.fill(0);
        Nak.set_real(Na);
        for(uword k=0; k<Nz; k++){
            arma_in = Nak.slice(k);
            fftw_execute(_p_forward);
            Nak.slice(k) = arma_out;
        }
        for(uword i=0; i<Nx; i++)
            for(uword j=0; j<Ny; j++){
                uword si = i + Nx * j;
                cx_colvec vak = ak.tube(i, j);
                cx_colvec vNak = Nak.tube(i, j);
                cx_colvec vNuk = Nuk.tube(i, j);
                cx_colvec vbk = vak + mf1.slice(si) * (vNak - vNuk);
                for(uword k=0; k<vbk.n_elem; k++)
                    bk(i, j, k) = vbk(k);
            }

        for(uword k=0; k<Nz; k++){
            arma_in = bk.slice(k);
            fftw_execute(_p_backward);
            cb.slice(k) = arma_out;
        }
        Nb = w % real(cb)/(Nx*Ny);
        Nbk.fill(0);
        Nbk.set_real(Nb);
        for(uword k=0; k<Nz; k++){
            arma_in = Nbk.slice(k);
            fftw_execute(_p_forward);
            Nbk.slice(k) = arma_out;
        }
        for(uword i=0; i<Nx; i++)
            for(uword j=0; j<Ny; j++){
                uword si = i + Nx * j;
                cx_colvec vuk = uk.tube(i, j);
                cx_colvec vNuk = Nuk.tube(i, j);
                cx_colvec vNbk = Nbk.tube(i, j);
                cx_colvec vck = mE1.slice(si) * vuk
                                + mf2.slice(si) * vNuk
                                + mf3.slice(si) * (vNbk - vNuk);
                for(uword k=0; k<vck.n_elem; k++)
                    ck(i, j, k) = vck(k);
            }

        for(uword k=0; k<Nz; k++){
            arma_in = ck.slice(k);
            fftw_execute(_p_backward);
            cc.slice(k) = arma_out;
        }
        Nc = w % real(cc)/(Nx*Ny);
        Nck.fill(0);
        Nck.set_real(Nc);
        for(uword k=0; k<Nz; k++){
            arma_in = Nck.slice(k);
            fftw_execute(_p_forward);
            Nck.slice(k) = arma_out;
        }
        for(uword i=0; i<Nx; i++)
            for(uword j=0; j<Ny; j++){
                uword si = i + Nx * j;
                cx_colvec vck = ck.tube(i, j);
                cx_colvec vNuk = Nuk.tube(i, j);
                cx_colvec vNak = Nak.tube(i, j);
                cx_colvec vNbk = Nbk.tube(i, j);
                cx_colvec vNck = Nck.tube(i, j);
                cx_colvec vuk = vck + mf3.slice(si) * vNak
                                + mf4.slice(si) * (vNuk + vNck)
                                + mf5.slice(si) * (vNak + vNbk);
                for(uword k=0; k<vuk.n_elem; k++)
                    uk(i, j, k) = vuk(k);
            }

        for(uword k=0; k<Nz; k++){
            arma_in = uk.slice(k);
            fftw_execute(_p_backward);
            cu.slice(k) = arma_out;
        }
        u = real(cu)/(Nx*Ny);
        q(s) = u;
    }
    return q;
}

/** 1D solution
 * PDE is
 *      du/dt = Lu + w*u
 * NOTE THE PLUS SIGN before w*u.
 * u is the initial condition.
 * The size of u and w must be the same and equal to _Lx.
 * Ns denotes number of time stpes.
 * q carries solutions for all s in [0, Ns*_ds] and of size (_Lx x Ns).
 * The original data of input q will be cleared.
 *
 * Krogstad scheme HAS BEEN THOROUGHLY TESTED AGAINST Chebpy, 2014-05-29.
 * Cox-Matthews scheme has not been tested, 2014-05-29.
 */
void Etdrk4::solve(const colvec &u, const colvec &w, const uword Ns, mat &q){
    BC lbc = _lb.kind();
    BC rbc = _rb.kind();
    q.set_size(_Lx, Ns);
    q.fill(fill::zeros);  // Fill boundary with 0 for DBCs.
    if(lbc==BC::DBC && rbc==BC::DBC){
        colvec uu = u.subvec(1, _Lx-2);
        colvec ww = w.subvec(1, _Lx-2);
        if(_stype == ETDRK4SCHEME::COX)
            q.rows(1, _Lx-2) = scheme_cox(uu, ww, Ns);
        else if(_stype == ETDRK4SCHEME::KROGSTAD)
            q.rows(1, _Lx-2) = scheme_krogstad(uu, ww, Ns);
    }
    else if(lbc==BC::DBC && (rbc==BC::RBC || rbc==BC::NBC)){
        colvec uu = u.subvec(0, _Lx-2);
        colvec ww = w.subvec(0, _Lx-2);
        if(_stype == ETDRK4SCHEME::COX)
            q.rows(1, _Lx-2) = scheme_cox(uu, ww, Ns);
        else if(_stype == ETDRK4SCHEME::KROGSTAD)
            q.rows(0, _Lx-2) = scheme_krogstad(uu, ww, Ns);
    }
    else if((lbc==BC::RBC || lbc==BC::NBC) && rbc==BC::DBC){
        colvec uu = u.subvec(1, _Lx-1);
        colvec ww = w.subvec(1, _Lx-1);
        if(_stype == ETDRK4SCHEME::COX)
            q.rows(1, _Lx-1) = scheme_cox(uu, ww, Ns);
        else if(_stype == ETDRK4SCHEME::KROGSTAD)
            q.rows(1, _Lx-1) = scheme_krogstad(uu, ww, Ns);
    }
    else{
        colvec uu = u;
        colvec ww = w;
        if(_stype == ETDRK4SCHEME::COX)
            q = scheme_cox(uu, ww, Ns);
        else if(_stype == ETDRK4SCHEME::KROGSTAD)
            q = scheme_krogstad(uu, ww, Ns);
    }
}

/** 1D solution
 * PDE is
 *      du/dt = Lu + w*u
 * NOTE THE PLUS SIGN before w*u.
 * u is the initial condition.
 * The size of u and w must be the same and equal to _Lx.
 * Ns denotes number of time stpes.
 * q carries solutions for all s in [0, Ns*_ds] and of size (Ns x _Lx).
 * The original data of input q will be cleared.
 *
 * Krogstad scheme HAS BEEN THOROUGHLY TESTED AGAINST Chebpy, 2014-05-29.
 * Cox-Matthews scheme has not been tested, 2014-05-29.
 */
void Etdrk4::solve(const rowvec &u, const rowvec &w, const uword Ns, mat &q){
    BC lbc = _lb.kind();
    BC rbc = _rb.kind();
    q.set_size(Ns, _Lx);
    q.fill(fill::zeros);  // Fill boundary with 0 for DBCs.
    if(lbc==BC::DBC && rbc==BC::DBC){
        rowvec uu = u.subvec(1, _Lx-2);
        rowvec ww = w.subvec(1, _Lx-2);
        if(_stype == ETDRK4SCHEME::COX)
            q.cols(1, _Lx-2) = scheme_cox(uu, ww, Ns);
        else if(_stype == ETDRK4SCHEME::KROGSTAD)
            q.cols(1, _Lx-2) = scheme_krogstad(uu, ww, Ns);
    }
    else if(lbc==BC::DBC && (rbc==BC::RBC || rbc==BC::NBC)){
        rowvec uu = u.subvec(0, _Lx-2);
        rowvec ww = w.subvec(0, _Lx-2);
        if(_stype == ETDRK4SCHEME::COX)
            q.cols(1, _Lx-2) = scheme_cox(uu, ww, Ns);
        else if(_stype == ETDRK4SCHEME::KROGSTAD)
            q.cols(0, _Lx-2) = scheme_krogstad(uu, ww, Ns);
    }
    else if((lbc==BC::RBC || lbc==BC::NBC) && rbc==BC::DBC){
        rowvec uu = u.subvec(1, _Lx-1);
        rowvec ww = w.subvec(1, _Lx-1);
        if(_stype == ETDRK4SCHEME::COX)
            q.cols(1, _Lx-1) = scheme_cox(uu, ww, Ns);
        else if(_stype == ETDRK4SCHEME::KROGSTAD)
            q.cols(1, _Lx-1) = scheme_krogstad(uu, ww, Ns);
    }
    else{
        rowvec uu = u;
        rowvec ww = w;
        if(_stype == ETDRK4SCHEME::COX)
            q = scheme_cox(uu, ww, Ns);
        else if(_stype == ETDRK4SCHEME::KROGSTAD)
            q = scheme_krogstad(uu, ww, Ns);
    }
}

/** 2D solution
 * PDE is
 *      du/dt = Lu + w*u
 * The system is confined in y direction
 * (must be the first index, i.e. row index).
 * u is the initial condition.
 * The size of u and w must be the same and of size (_Ly x _Lx).
 * Ns denotes number of time stpes.
 * q carries solutions for all s in [0, Ns*_ds] and of size (_Ly x _Lx x Ns).
 * The original data of input q will be cleared.
 */
void Etdrk4::solve(const mat &u, const mat &w, const uword Ns, cube &q){
    BC lbc = _lb.kind();
    BC rbc = _rb.kind();
    q.set_size(_Ly, _Lx, Ns);
    q.fill(0.0);  // Fill boundary with 0 for DBCs.
    if(lbc==BC::DBC && rbc==BC::DBC){
        mat uu = u.rows(1, _Ly-2);
        mat ww = w.rows(1, _Ly-2);
        if(_stype == ETDRK4SCHEME::COX)
            q.tube(span(1, _Ly-2), span::all) = scheme_cox(uu, ww, Ns);
        else if(_stype == ETDRK4SCHEME::KROGSTAD)
            q.tube(span(1, _Ly-2), span::all) = scheme_krogstad(uu, ww, Ns);
    }
    else if(lbc==BC::DBC && (rbc==BC::RBC || rbc==BC::NBC)){
        mat uu = u.rows(0, _Ly-2);
        mat ww = w.rows(0, _Ly-2);
        if(_stype == ETDRK4SCHEME::COX)
            q.tube(span(0, _Ly-2), span::all) = scheme_cox(uu, ww, Ns);
        else if(_stype == ETDRK4SCHEME::KROGSTAD)
            q.tube(span(0, _Ly-2), span::all) = scheme_krogstad(uu, ww, Ns);
    }
    else if((lbc==BC::RBC || lbc==BC::NBC) && rbc==BC::DBC){
        mat uu = u.rows(1, _Ly-1);
        mat ww = w.rows(1, _Ly-1);
        if(_stype == ETDRK4SCHEME::COX)
            q.tube(span(1, _Ly-1), span::all) = scheme_cox(uu, ww, Ns);
        else if(_stype == ETDRK4SCHEME::KROGSTAD)
            q.tube(span(1, _Ly-1), span::all) = scheme_krogstad(uu, ww, Ns);
    }
    else{
        mat uu = u;
        mat ww = w;
        if(_stype == ETDRK4SCHEME::COX)
            q = scheme_cox(uu, ww, Ns);
        else if(_stype == ETDRK4SCHEME::KROGSTAD)
            q = scheme_krogstad(uu, ww, Ns);
    }
}

/** 3D solution
 * PDE is
 *      du/dt = Lu + w*u
 * The system is confined in z direction
 * (must be the last index, i.e. slice index).
 * u is the initial condition.
 * The size of u and w must be the same and of size (_Lx x _Ly x _Lz).
 * Ns denotes number of time stpes.
 * q carries solutions for all s in [0, Ns*_ds] and
 * of size (_Lx x _Ly x _Lz x Ns).
 * The original data of input q will be cleared.
 */
void Etdrk4::solve(const cube &u, const cube &w, const uword Ns,
                   field<cube> &q){
    BC lbc = _lb.kind();
    BC rbc = _rb.kind();

    q.set_size(Ns);
    for(uword i=0; i<Ns; i++)
        q(i) = zeros<cube>(_Lx, _Ly, _Lz);  // Fill boundary with 0 for DBCs.

    if(lbc==BC::DBC && rbc==BC::DBC){
        cube uu = u.slices(1, _Lz-2);
        cube ww = w.slices(1, _Lz-2);
        field<cube> qo;
        if(_stype == ETDRK4SCHEME::COX){
            qo = scheme_cox(uu, ww, Ns);
            for(uword i=0; i<Ns; i++)
                q(i).slices(1, _Lz-2) = qo(i);
        }
        else if(_stype == ETDRK4SCHEME::KROGSTAD){
            qo = scheme_krogstad(uu, ww, Ns);
            for(uword i=0; i<Ns; i++)
                q(i).slices(1, _Lz-2) = qo(i);
        }
    }
    else if(lbc==BC::DBC && (rbc==BC::RBC || rbc==BC::NBC)){
        cube uu = u.slices(0, _Lz-2);
        cube ww = w.slices(0, _Lz-2);
        field<cube> qo;
        if(_stype == ETDRK4SCHEME::COX){
            qo = scheme_cox(uu, ww, Ns);
            for(uword i=0; i<Ns; i++)
                q(i).slices(0, _Lz-2) = qo(i);
        }
        else if(_stype == ETDRK4SCHEME::KROGSTAD){
            qo = scheme_krogstad(uu, ww, Ns);
            for(uword i=0; i<Ns; i++)
                q(i).slices(0, _Lz-2) = qo(i);
        }
    }
    else if((lbc==BC::RBC || lbc==BC::NBC) && rbc==BC::DBC){
        cube uu = u.slices(1, _Lz-1);
        cube ww = w.slices(1, _Lz-1);
        field<cube> qo;
        if(_stype == ETDRK4SCHEME::COX){
            qo = scheme_cox(uu, ww, Ns);
            for(uword i=0; i<Ns; i++)
                q(i).slices(1, _Lz-1) = qo(i);
        }
        else if(_stype == ETDRK4SCHEME::KROGSTAD){
            qo = scheme_krogstad(uu, ww, Ns);
            for(uword i=0; i<Ns; i++)
                q(i).slices(1, _Lz-1) = qo(i);
        }
    }
    else{
        cube uu = u;
        cube ww = w;
        if(_stype == ETDRK4SCHEME::COX)
            q = scheme_cox(uu, ww, Ns);
        else if(_stype == ETDRK4SCHEME::KROGSTAD)
            q = scheme_krogstad(uu, ww, Ns);
    }
}

