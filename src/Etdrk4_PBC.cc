#include "Etdrk4_PBC.h"
#include "common.h"  // for constant PI

Etdrk4_PBC::Etdrk4_PBC(const Etdrk4_PBC &rhs):_Etdrk4_L(rhs._Etdrk4_L),
                                              _E(rhs._E), _E2(rhs._E2),
                                              _Q(rhs._Q), _f1(rhs._f1),
                                              _f2(rhs._f2), _f3(rhs._f3),
                                              _M(rhs._M){
    init_fftw();
}

Etdrk4_PBC::Etdrk4_PBC(const UnitCell &uc,
                       const int Lx, const int Ly, const int Lz,
                       const double ds,
                       const int M /* = 32 */
                      ):_Etdrk4_L(Lx,Ly,Lz), _E(Lx,Ly,Lz), _E2(Lx,Ly,Lz),
                        _Q(Lx,Ly,Lz), _f1(Lx,Ly,Lz), _f2(Lx,Ly,Lz),
                        _f3(Lx,Ly,Lz), _M(M){
    _Etdrk4_L = uc.calc_k2_orthogonal(Lx, Ly, Lz);
    init_fftw();
    init_coefficient(ds);
}

Etdrk4_PBC* Etdrk4_PBC::clone() const{
    return new Etdrk4_PBC(*this);
}

void Etdrk4_PBC::init_fftw() {
    int Lx = _Etdrk4_L.rows();
    int Ly = _Etdrk4_L.cols();
    int Lz = _Etdrk4_L.depth();
    _fftw_in = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*Lx*Ly*Lz));
    _fftw_out = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex)*Lx*Ly*Lz));
    _p_forward = fftw_plan_dft_3d(Lx, Ly, Lz, _fftw_in, _fftw_out,
                                  FFTW_FORWARD, FFTW_ESTIMATE);
    _p_backward = fftw_plan_dft_3d(Lx, Ly, Lz, _fftw_out, _fftw_in,
                                   FFTW_BACKWARD, FFTW_ESTIMATE);
}

void Etdrk4_PBC::init_coefficient(const double ds) {
	int M = _M;
	int Lx = _Etdrk4_L.rows();
    int Ly = _Etdrk4_L.cols();
    int Lz = _Etdrk4_L.depth();

    Array<complex<double>,1> r(M);           //1i*pi((1:M)-.5)/M
	Array<complex<double>,4> LM(Lx,Ly,Lz,M);  //L(:,ones(M,1))
	Array<complex<double>,4> rN(Lx,Ly,Lz,M);  //r(ones(N,1),:)
	Array<complex<double>,4> LR(Lx,Ly,Lz,M);
	Array<complex<double>,4> res(Lx,Ly,Lz,M);    // store results
	Array<complex<double>,4> eLR(Lx,Ly,Lz,M);  // exp(LR)
    Array<complex<double>,4> LR2(Lx,Ly,Lz,M);    // LR^2
    Array<complex<double>,4> LR3(Lx,Ly,Lz,M);    // LR^3

	firstIndex i2;
	secondIndex j2;
	thirdIndex k2;
	fourthIndex l2;
    Range all = Range::all();

	_E = exp(ds*_Etdrk4_L);

	_E2 = exp(ds*_Etdrk4_L/2.0);

    r = zip(0, PI * (i2 + 0.5) / M, complex<double>());
    r = exp(r);
    for(int l=0; l<M; l++)
        LM(all, all, all, l) = zip(_Etdrk4_L, 0, complex<double>());
	for(int i=0; i<Lx; i++)
	   for(int j=0; j<Ly; j++)
	       for(int k=0; k<Lz; k++)
	           rN(i, j, k, all) = r;

	LR = ds * LM + rN;
    eLR = exp(LR);
    LR2 = LR * LR;
    LR3 = LR2 * LR;

    /** VERY IMPORTANT !! **/
    /* Following constant must be 1.0, 2.0, etc. but not 1, 2
     * due to compilation error of blitz++
     */
    res = (exp(LR/2.0) - 1.0) / LR;  // (exp(x/2)-1)/x
    _Q = ds * mean(real(res)(i2,j2,k2,l2),l2);

    res = (eLR*(LR2-3.0*LR+4.0) - LR - 4.0) / LR3;  // (e^x(x^2-3x+4)-x-4)/x^3
    _f1 = ds * mean(real(res)(i2,j2,k2,l2),l2);

    res = (eLR*(LR-2.0) + LR + 2.0) / LR3;  // (e^x(x-2)+x+2)/x^3
	_f2 = ds * mean(real(res)(i2,j2,k2,l2),l2);

    res = (eLR*(4.0-LR) - LR2 - 3.0*LR - 4.0) / LR3;  //(e^x(4-x)-x^2-3x-4)/x^3
	_f3 = ds * mean(real(res)(i2,j2,k2,l2),l2);
}

Etdrk4_PBC::~Etdrk4_PBC() {
	fftw_free(_fftw_in);
	fftw_free(_fftw_out);
	fftw_destroy_plan(_p_forward);
	fftw_destroy_plan(_p_backward);
}

void Etdrk4_PBC::solve(Propagator &q,const Grid &w) {
	int Lx = q.Lx();
	int Ly = q.Ly();
	int Lz = q.Lz();
	int ngrids = Lx*Ly*Lz;

   	Array<double,4> qs(q.qs());
	Array<double,3> ws(w.data());

	Array<complex<double>,3> Nv(Lx,Ly,Lz);
    Array<complex<double>,3> a(Lx,Ly,Lz);
    Array<complex<double>,3> Na(Lx,Ly,Lz);
    Array<complex<double>,3> b(Lx,Ly,Lz);
	Array<complex<double>,3> Nb(Lx,Ly,Lz);
	Array<complex<double>,3> c(Lx,Ly,Lz);
    Array<complex<double>,3> Nc(Lx,Ly,Lz);
    Array<complex<double>,3> v(Lx,Ly,Lz);

	Array<complex<double>,3> fftw_cin(reinterpret_cast<complex<double> *>(_fftw_in),shape(Lx,Ly,Lz),neverDeleteData);
   	Array<complex<double>,3> fftw_cout(reinterpret_cast<complex<double> *>(_fftw_out),shape(Lx,Ly,Lz),neverDeleteData);

    Range all = Range::all();
	fftw_cin = zip(qs(0,all,all,all), 0, complex<double>());
	fftw_execute(_p_forward);
	v = fftw_cout;

    for(int s=0; s<q.len()-1; s++){
        Array<double, 3> q1(qs(s,all,all,all));
        Array<double, 3> q2(qs(s+1,all,all,all));

	    fftw_cin = zip(-ws*q1, 0, complex<double>());
		fftw_execute(_p_forward);
		Nv = fftw_cout;

		a = _E2*v + _Q*Nv;
		fftw_cout = a;
		fftw_execute(_p_backward);
		fftw_cin /= ngrids;
        fftw_cin = zip(-ws*real(fftw_cin), 0, complex<double>());
		fftw_execute(_p_forward);
		Na = fftw_cout;

		b = _E2*v + _Q*Na;
        fftw_cout = b;
		fftw_execute(_p_backward);
		fftw_cin /= ngrids;
        fftw_cin = zip(-ws*real(fftw_cin), 0, complex<double>());
		fftw_execute(_p_forward);
		Nb = fftw_cout;

        c = _E2*a + _Q*(2.0*Nb-Nv);
        fftw_cout = c;
		fftw_execute(_p_backward);
		fftw_cin /= ngrids;
		fftw_cin = zip(-ws*real(fftw_cin), 0, complex<double>());
		fftw_execute(_p_forward);
		Nc = fftw_cout;

        v = _E*v + _f1*Nv + 2.0*_f2*(Na+Nb) + _f3*Nc;
        fftw_cout = v;
		fftw_execute(_p_backward);
		fftw_cin /= ngrids;
		q2 = real(fftw_cin);
	}
}

















