#include "RQM4.h"
#include "UnitCell.h"
#include "blitz/array.h"

RQM4::RQM4(const RQM4 &rhs):_laplace(rhs._laplace),_laplace2(rhs._laplace2){
    init_fftw();
}

RQM4::RQM4(const UnitCell &uc, const int Lx, const int Ly, const int Lz,
           const double ds):_laplace(Lx,Ly,Lz), _laplace2(Lx,Ly,Lz){
    // calc_k2_orthogonal acctually compute -k^2.
    blitz::Array<double,3> k2 = uc.calc_k2_orthogonal(Lx, Ly, Lz);
    _laplace = blitz::exp(ds*k2);
    _laplace2 = blitz::exp(0.5*ds*k2);
    init_fftw();
}

RQM4* RQM4::clone() const{
    return new RQM4(*this);
}

void RQM4::init_fftw(){
    int Lx = _laplace.rows();
    int Ly = _laplace.cols();
    int Lz = _laplace.depth();
    _fftw_in = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*Lx*Ly*Lz));
    _fftw_out = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*Lx*Ly*Lz));
    _p_forward = fftw_plan_dft_3d(Lx, Ly, Lz, _fftw_in, _fftw_out,
                                  FFTW_FORWARD, FFTW_ESTIMATE);
    _p_backward = fftw_plan_dft_3d(Lx, Ly, Lz, _fftw_out, _fftw_in,
                                   FFTW_BACKWARD, FFTW_ESTIMATE);
}

RQM4::~RQM4(){
    fftw_free(_fftw_in);
    fftw_free(_fftw_out);
    fftw_destroy_plan(_p_forward);
    fftw_destroy_plan(_p_backward);
}

void RQM4::solve(Propagator &q, const Grid &w){
    int Lx = q.Lx();
    int Ly = q.Ly();
    int Lz = q.Lz();
    double ds = q.ds();
    blitz::Array<double, 4> qs(q.qs());
    int ngrids = Lx * Ly * Lz;

    blitz::Array<double, 3> expq(Lx, Ly, Lz);
    expq = blitz::exp(-0.5 * ds * w.data());
    blitz::Array<double, 3> expq2(Lx, Ly, Lz);
    expq2 = blitz::exp(-0.25 * ds * w.data());

    blitz::Range all = blitz::Range::all();
    blitz::Array<complex<double>,3> bfftw_in(
                            reinterpret_cast<complex<double> *>(_fftw_in),
                            blitz::shape(Lx, Ly, Lz),
                            blitz::neverDeleteData);
    blitz::Array<complex<double>,3> bfftw_out(
                            reinterpret_cast<complex<double> *>(_fftw_out),
                            blitz::shape(Lx, Ly, Lz),
                            blitz::neverDeleteData);

    blitz::Array<double,3> qt(Lx, Ly, Lz);
    // q(r, j), j = s0, s0+1, ..., s1
    for (int s=0; s<q.len()-1; s++){
        blitz::Array<double,3> q1(qs(s, all, all, all));
        blitz::Array<double,3> q2(qs(s+1, all, all, all));

        // Contour step ds
        bfftw_in = blitz::zip(expq*q1, 0, complex<double>());
        fftw_execute(_p_forward);   // fftw_in => fftw_out
        bfftw_out *= _laplace;
        fftw_execute(_p_backward);  // fftw_out => fftw_in
        q2 = (blitz::real(bfftw_in) * expq) / ngrids;

        // Contour step ds/2
        //   First step
        bfftw_in = blitz::zip(expq2*q1, 0, complex<double>());
        fftw_execute(_p_forward);   // fftw_in => fftw_out
        bfftw_out *= _laplace2;
        fftw_execute(_p_backward);  // fftw_out => fftw_in
        qt = (blitz::real(bfftw_in) * expq2) / ngrids;
        //   Second step
        bfftw_in = blitz::zip(expq2*qt, 0, complex<double>());
        fftw_execute(_p_forward);   // fftw_in => fftw_out
        bfftw_out *= _laplace2;
        fftw_execute(_p_backward);  // fftw_out => fftw_in
        qt = (blitz::real(bfftw_in) * expq2) / ngrids;

        q2 = (4.0*qt - q2) / 3.0;
    }
}

