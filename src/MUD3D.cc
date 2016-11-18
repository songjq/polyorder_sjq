/**
 * MUD3D.cc
 * Created at 2012.4.16
 *
 * Implementation of MUD3D.h.
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

#include "MUD3D.h"

extern "C"{
    void mud3sp_(MUD_INT *iparm,MUD_FLOAT *fparm,MUD_FLOAT *work, \
                 MUD3D::COF_FUNC,MUD3D::COF_FUNC,MUD3D::COF_FUNC, \
                 MUD3D::BND_FUNC, \
                 MUD_FLOAT *rhs,MUD_FLOAT *phi, \
                 MUD_INT *mgopt,MUD_INT *ierror);

    void mud3sa_(MUD_INT *iparm,MUD_FLOAT *fparm,MUD_FLOAT *work, \
                 MUD3D::SIG_FUNC,MUD3D::SIG_FUNC,MUD3D::SIG_FUNC, \
                 MUD3D::SIG_FUNC,MUD3D::BND_FUNC, \
                 MUD_FLOAT *rhs,MUD_FLOAT *phi, \
                 MUD_INT *mgopt,MUD_INT *ierror);

    void mud3cr_(MUD_INT *iparm,MUD_FLOAT *fparm,MUD_FLOAT *work, \
                 MUD3D::COF_CR_FUNC,MUD3D::BND_CR_FUNC, \
                 MUD_FLOAT *rhs,MUD_FLOAT *phi,MUD_INT *mgopt, \
                 MUD_INT *icros,MUD3D::CRS_CR_FUNC, \
                 MUD3D::CRS_CR_FUNC,MUD3D::CRS_CR_FUNC, \
                 MUD_FLOAT *tol,MUD_INT *maxit,MUD_INT *iouter, \
                 MUD_FLOAT *rmax,MUD_INT *ierror);
}

void dummy_cof3(MUD_FLOAT *x,MUD_FLOAT *cxx,MUD_FLOAT *cx,MUD_FLOAT *cex){
    *cxx=1.0;
    *cx=0.0;
    *cex=0.0;
}

double dummy_sig3(MUD_FLOAT *x,MUD_FLOAT *y,MUD_FLOAT *z){
    return 1.0;
}

double dummy_xlmbda3(MUD_FLOAT *x,MUD_FLOAT *y,MUD_FLOAT *z){
    return 0.0;
}

void dummy_bnd3(MUD_INT *kbdy,MUD_FLOAT *xory,MUD_FLOAT *yorz,MUD_FLOAT *alfa,MUD_FLOAT *gbdy){
}

/**
 * For Hexagonal unit cell + position-independent dielectric constant
 *      (4/3)*pxx + (4/3)*pyy + pzz + (4/3)*pxy = r(x,y)
 *
 */
void dummy_cof3_cr(MUD_FLOAT *x,MUD_FLOAT *y,MUD_FLOAT *z,MUD_FLOAT *cxx,MUD_FLOAT *cyy,MUD_FLOAT *czz,MUD_FLOAT *cx,MUD_FLOAT *cy,MUD_FLOAT *cz,MUD_FLOAT *ce){
    *cxx = 4.0 / 3.0;
    *cyy = 4.0 / 3.0;
    *czz = 1.0;
    *cx = 0.0;
    *cy = 0.0;
    *cz = 0.0;
    *ce = 0.0;
}

void dummy_crsxy(MUD_FLOAT *x,MUD_FLOAT *y,MUD_FLOAT *z,MUD_FLOAT *cxy){
    *cxy = 4.0 / 3.0;
}

void dummy_crsxz(MUD_FLOAT *x,MUD_FLOAT *y,MUD_FLOAT *z,MUD_FLOAT *cxz){
    *cxz = 0.0;
}

void dummy_crsyz(MUD_FLOAT *x,MUD_FLOAT *y,MUD_FLOAT *z,MUD_FLOAT *cyz){
    *cyz = 0.0;
}

/**
 * For periodic boundary conditions on all boundaries.
 *
 */
void dummy_bnd3_cr(MUD_INT *kbdy,MUD_FLOAT *xory,MUD_FLOAT *yorz,MUD_FLOAT *a,MUD_FLOAT *b,MUD_FLOAT *c,MUD_FLOAT *g){
}

MUD3D::MUD3D(const MUD3D &rhs):MUD(rhs){
    set_cof(rhs._pcofx,rhs._pcofy,rhs._pcofz);
    set_cof(rhs._pcof_cr,rhs._pcrsxy,rhs._pcrsxz,rhs._pcrsyz);
    set_sig(rhs._psigx,rhs._psigy,rhs._psigz);
    set_xlmbda(rhs._xlmbda);
    set_boundary_function(rhs._pboundary);
    set_boundary_function(rhs._pboundary_cr);
    set_cr_extras(rhs._icros[0],rhs._icros[1],rhs._icros[2],rhs._tol,rhs._maxit);
    init();
}

MUD3D::MUD3D(const string &t,int Lx,int Ly,int Lz,double xb,double yd,double zf,double xa,double yc,double ze,int fft2mg_mode){
    set_type(t);
    set_cof(dummy_cof3,dummy_cof3,dummy_cof3);
    set_cof(dummy_cof3_cr,dummy_crsxy,dummy_crsxz,dummy_crsyz);
    set_sig(dummy_sig3,dummy_sig3,dummy_sig3);
    set_xlmbda(dummy_xlmbda3);
    set_boundary_function(dummy_bnd3);
    set_boundary_function(dummy_bnd3_cr);
    set_boundary_type();
    set_cr_extras();
    _fft2mg_mode = fft2mg_mode;
    set_grid(Lx,Ly,Lz,xb,yd,zf,xa,yc,ze);
    if(_type == MUD_SA)
        set_sig_parm(); // should be called after set_grid
    else
        set_cof_cr_parm();
    set_fft2mg_interp();
    set_method();
    set_guess();
    set_maxcy();
    set_mgopt();
    set_tolmax();

    // discard the length calculated in set_method
    if(_type == MUD_SA) _iparm[20] = 0;
}

void MUD3D::init(){
    int nx = _iparm[13];
    int ny = _iparm[14];
    int nz = _iparm[15];
    blitz::Array<MUD_FLOAT,3> data(nx,ny,nz);
    data = 0.0;

    _iparm[0] = 0;
    int length;
    switch(_type){
        case MUD_SP:
            length = _iparm[19];
            _work.resize(length);
            mud3sp_(_iparm,_fparm,_work.data(), \
                    _pcofx,_pcofy,_pcofz,_pboundary, \
                    data.data(),data.data(),_mgopt,&_ierror);
            break;
        case MUD_SA:
            // Input length = 0 to estimate the actual work length.
            if(_iparm[20] == 0){
                mud3sa_(_iparm,_fparm,_work.data(), \
                        _psigx,_psigy,_psigz,_xlmbda,_pboundary, \
                        data.data(),data.data(),_mgopt,&_ierror);
                length = _iparm[21];
                _iparm[20] = length;
                _work.resize(length);
            }
            // re-discretize
            mud3sa_(_iparm,_fparm,_work.data(), \
                    _psigx,_psigy,_psigz,_xlmbda,_pboundary, \
                    data.data(),data.data(),_mgopt,&_ierror);
            break;
        case MUD_CR:
            // Input length = 0 to estimate the actual work length.
            if(_iparm[20] == 0){
                mud3cr_(_iparm,_fparm,_work.data(), \
                        _pcof_cr,_pboundary_cr, \
                        data.data(),data.data(),_mgopt, \
                        _icros,_pcrsxy,_pcrsxz,_pcrsyz, \
                        &_tol,&_maxit,&_iouter,_rmax.data(), \
                        &_ierror);
                length = _iparm[21];
                _iparm[20] = length;
                _work.resize(length);
            }
            // re-discretize
            mud3cr_(_iparm,_fparm,_work.data(), \
                    _pcof_cr,_pboundary_cr, \
                    data.data(),data.data(),_mgopt, \
                    _icros,_pcrsxy,_pcrsxz,_pcrsyz, \
                    &_tol,&_maxit,&_iouter,_rmax.data(), \
                    &_ierror);
            break;
        default:
            cout<<"Unsupported MUDPACK solver!"<<endl;
            throw("Unsupported MUDPACK solver!");
            break;
    }

    if(_ierror<0){
        //cout<<"Warning: MUDPACK ierror = "<<_ierror<<endl;
    }
    if(_ierror>0){
        cout<<"MUD3D error: "<<_ierror<<endl;
        throw("MUD3D error initialization error!");
    }

    _iparm[0] = 1; // After initialization call
}

/**
 * Set discretized condition
 * In MUDPACK, the number of grids in each dimension has the form:
 *      p * 2^(i-1) + 1
 * while the grids of the input data has the form:
 *      2 * L + 1
 * where L is either Lx or Ly. Thus, we have
 *      p * 2^(i-1) + 1 = 2 * L + 1
 * That is
 *      L = p * 2^(i-2)
 * It requires that L must be even, otherwise p = L should be very large and
 * not suitable for MUDPACK.
 *
 */
void MUD3D::set_grid(int Lx,int Ly,int Lz,double xb,double yd,double zf,double xa,double yc,double ze){
    int nx,ny,nz;
    int ixp,iex;
    int jyq,jey;
    int kzr,kez;
    _decompose_L(Lx,ixp,iex);
    _decompose_L(Ly,jyq,jey);
    _decompose_L(Lz,kzr,kez);
    _iparm[7] = ixp;
    _iparm[8] = jyq;
    _iparm[9] = kzr;

    if (_fft2mg_mode == 0){
        iex -= 1;
        jey -= 1;
        kez -= 1;
        nx = Lx + 1;
        ny = Ly + 1;
        nz = Lz + 1;
    }
    else{
        nx = 2*Lx + 1;
        ny = 2*Ly + 1;
        nz = 2*Lz + 1;
    }
    _iparm[10] = iex;
    _iparm[11] = jey;
    _iparm[12] = kez;
    _iparm[13] = nx;
    _iparm[14] = ny;
    _iparm[15] = nz;

    _fparm[0] = xa;
    _fparm[1] = xb;
    _fparm[2] = yc;
    _fparm[3] = yd;
    _fparm[4] = ze;
    _fparm[5] = zf;
}

/**
 * Determines the method of relaxation
 * (Gauss-Seidel based on alternating points or lines)
 * = 0 for point relaxation
 * = 1 for line relaxation in the x direction
 * = 2 for line relaxation in the y direction
 * = 3 for line relaxation in both the x and y directions
 *
 * From the manual of MUDPACK 5.0.1
 *
 */
void MUD3D::set_method(int imethod){
    int nxa = _iparm[1];
    int nyc = _iparm[3];
    int nze = _iparm[5];
    int nx = _iparm[13];
    int ny = _iparm[14];
    int nz = _iparm[15];
    int isx,jsy,ksz,length;
    switch(imethod){
        case 0: case 8: case 9: case 10:
            isx=0;jsy=0;ksz=0;
            break;
        case 1:
            if(nxa==0) isx=5;
            else isx=3;
            jsy=0;ksz=0;
            break;
        case 2:
            if(nyc==0) jsy=5;
            else jsy=3;
            isx=0;ksz=0;
            break;
        case 3:
            if(nze==0) ksz=5;
            else ksz=3;
            isx=0;jsy=0;
            break;
        case 4:
            if(nxa==0) isx=5;
            else isx=3;
            if(nyc==0) jsy=5;
            else jsy=3;
            ksz=0;
            break;
        case 5:
            if(nxa==0) isx=5;
            else isx=3;
            if(nze==0) ksz=5;
            else ksz=3;
            jsy=0;
            break;
        case 6:
            if(nyc==0) jsy=5;
            else jsy=3;
            if(nze==0) ksz=5;
            else ksz=3;
            isx=0;
            break;
        case 7:
            if(nxa==0) isx=5;
            else isx=3;
            if(nyc==0) jsy=5;
            else jsy=3;
            if(nze==0) ksz=5;
            else ksz=3;
            break;
        default:
            isx=5;jsy=5;ksz=5;
    }

    _iparm[18] = imethod;
    switch(_type){
        case MUD_SP:
            length = 7*(nx+2)*(ny+2)*(nz+2)/2;
            _iparm[19] = length; //work space length 
            break;
        case MUD_SA:
            if(imethod<=7)
                length = (10+isx+jsy+ksz)*(nx+2)*(ny+2)*(nz+2);
            else
                length = 14*(nx+2)*(ny+2)*(nz+2);
            _iparm[19] = 0; // imethod2
            _iparm[20] = 0; // work space length, flag to be re-estimated. 
            break;
        case MUD_CR:
            if(imethod<=7)
                length = (10+isx+jsy+ksz)*(nx+2)*(ny+2)*(nz+2);
            else
                length = 14*(nx+2)*(ny+2)*(nz+2);
            length += 25*nx*ny*nz/7 + 6*(nx*ny+nx*nz+ny*nz);
            _iparm[19] = 0; // imethod2
            _iparm[20] = 0; // work space length, flag to be re-estimated. 
            break;
        default:
            throw("Unsupported MUDPACK solver!");
            break;
    }
}

void MUD3D::set_sig_parm(){
    // only mud3sa_ use sig function
    if(_type != MUD_SA) return;
    // the set_data only works for fft2mg_mode = 0
    if(_fft2mg_mode != 0) return;
    
    MUD_INT nx = _iparm[13];
    MUD_INT ny = _iparm[14];
    MUD_INT nz = _iparm[15];
    MUD_INT ixp = _iparm[7];
    MUD_INT jyq = _iparm[8];
    MUD_INT kzr = _iparm[9];
    MUD_FLOAT xa = _fparm[0];
    MUD_FLOAT xb = _fparm[1];
    MUD_FLOAT yc = _fparm[2];
    MUD_FLOAT yd = _fparm[3];
    MUD_FLOAT ze = _fparm[4];
    MUD_FLOAT zf = _fparm[5];
    MUD_FLOAT lx = xb - xa;
    MUD_FLOAT ly = yd - yc;
    MUD_FLOAT lz = zf - ze;
    MUD_FLOAT dx = lx / (nx - 1);
    MUD_FLOAT dy = ly / (ny - 1);
    MUD_FLOAT dz = lz / (nz - 1);
    gmudsig.dx = 0.5 * dx;
    gmudsig.dy = 0.5 * dy;
    gmudsig.dz = 0.5 * dz;
    gmudsig.xa = xa - lx / (2. * ixp);
    gmudsig.yc = yc - ly / (2. * jyq);
    gmudsig.ze = ze - lz / (2. * kzr);
    gmudsig.xb = xb + lx / (2. * ixp);
    gmudsig.yd = yd + ly / (2. * jyq);
    gmudsig.zf = zf + lz / (2. * kzr);
    gmudsig.nx = (int)((gmudsig.xb - gmudsig.xa) / gmudsig.dx + 0.01) + 1;
    gmudsig.ny = (int)((gmudsig.yd - gmudsig.yc) / gmudsig.dy + 0.01) + 1;
    gmudsig.nz = (int)((gmudsig.zf - gmudsig.ze) / gmudsig.dz + 0.01) + 1;

    // xdata for sigx, sigy, and sigz
    delete gmudsig.xdata;
    gmudsig.xdata = new MUD_FLOAT[gmudsig.nx*gmudsig.ny*gmudsig.nz];

    /*
    // These codes are not neccessary for Poission equation in scft.
    // since xlambda = 0.0 and sigma for x, y, and z are the same,
    // all be the dielectric constant.
    delete gmudsig.ydata;
    delete gmudsig.zdata;
    delete gmudsig.lambda_data;
    gmudsig.ydata = new MUD_FLOAT[gmudsig.nx*gmudsig.ny*gmudsig.nz];
    gmudsig.zdata = new MUD_FLOAT[gmudsig.nx*gmudsig.ny*gmudsig.nz];
    gmudsig.lambda_data = new MUD_FLOAT[gmudsig.nx*gmudsig.ny*gmudsig.nz];
    */
}

void MUD3D::set_cof_cr_parm(){
    // only mud3cr_ use this function
    if(_type != MUD_CR) return;
    // the set_data only works for fft2mg_mode = 0
    if(_fft2mg_mode != 0) return;
    
    MUD_INT nx = _iparm[13];
    MUD_INT ny = _iparm[14];
    MUD_INT nz = _iparm[15];
    MUD_FLOAT xa = _fparm[0];
    MUD_FLOAT xb = _fparm[1];
    MUD_FLOAT yc = _fparm[2];
    MUD_FLOAT yd = _fparm[3];
    MUD_FLOAT ze = _fparm[4];
    MUD_FLOAT zf = _fparm[5];
    MUD_FLOAT lx = xb - xa;
    MUD_FLOAT ly = yd - yc;
    MUD_FLOAT lz = zf - ze;
    MUD_FLOAT dx = lx / (nx - 1);
    MUD_FLOAT dy = ly / (ny - 1);
    MUD_FLOAT dz = lz / (nz - 1);
    gmudsig.dx = dx;
    gmudsig.dy = dy;
    gmudsig.dz = dz;
    gmudsig.xa = xa;
    gmudsig.yc = yc;
    gmudsig.ze = ze;
    gmudsig.xb = xb;
    gmudsig.yd = yd;
    gmudsig.zf = zf;
    gmudsig.nx = nx;
    gmudsig.ny = ny;
    gmudsig.nz = nz;

    delete gmudsig.xdata;
    delete gmudsig.ydata;
    delete gmudsig.zdata;
    delete gmudsig.lambda_data;
    gmudsig.xdata = new MUD_FLOAT[gmudsig.nx*gmudsig.ny*gmudsig.nz];
    gmudsig.ydata = new MUD_FLOAT[gmudsig.nx*gmudsig.ny*gmudsig.nz];
    gmudsig.zdata = new MUD_FLOAT[gmudsig.nx*gmudsig.ny*gmudsig.nz];
    gmudsig.lambda_data = new MUD_FLOAT[gmudsig.nx*gmudsig.ny*gmudsig.nz];
}

void MUD3D::set_data(const blitz::Array<double,3> sig_data_fft){
    // only mud3sa_ use sig function
    if(_type != MUD_SA) return;
    // the set_data only works for fft2mg_mode = 0
    if(_fft2mg_mode != 0) return;
    
    MUD_INT nx = _iparm[13];
    MUD_INT ny = _iparm[14];
    MUD_INT nz = _iparm[15];
    MUD_FLOAT xa = _fparm[0];
    MUD_FLOAT yc = _fparm[2];
    MUD_FLOAT ze = _fparm[4];

    // Here, nx and ny are mg grids.
    // nx = Lx + 1, ny = Ly + 1 in fft2mg_mode = 0
    if(sig_data_fft.rows() != nx-1 || sig_data_fft.cols() != ny-1 || sig_data_fft.depth() != nz-1){
        cout<<"Dielectric constant grid does not match MUD2D!"<<endl;
        throw("Dielectric constant grid does not match MUD2D!");
    }
    blitz::Array<double,3> sig_data_mg1(nx,ny,nz);
    blitz::Array<double,3> sig_data_mg2(2*nx-1,2*ny-1,2*nz-1);

    fft2mg_copy_3d(sig_data_fft,sig_data_mg1);
    interp_trilinear_3d(sig_data_mg1,sig_data_mg2);

    int index;
    int i_xa = (int)((xa - gmudsig.xa) / gmudsig.dx + 0.01);
    int j_yc = (int)((yc - gmudsig.yc) / gmudsig.dy + 0.01);
    int k_ze = (int)((ze - gmudsig.ze) / gmudsig.dz + 0.01);
    int period_x = 2 * (nx - 1);
    int period_y = 2 * (ny - 1);
    int period_z = 2 * (nz - 1);
    int ix,jy,kz;
    for(int i=0;i<gmudsig.nx;i++)
      for(int j=0;j<gmudsig.ny;j++)
        for(int k=0;k<gmudsig.nz;k++){
          ix = i - i_xa;
          if(ix < 0) ix += period_x;
          else if(ix > period_x) ix -= period_x;

          jy = j - j_yc;
          if(jy < 0) jy += period_y;
          else if(jy > period_y) jy -= period_y;

          kz = k - k_ze;
          if(kz < 0) kz += period_z;
          else if(kz > period_z) kz -= period_z;

          index = gmudsig.nz * (gmudsig.ny * i + j) + k;
          gmudsig.xdata[index] = sig_data_mg2(ix,jy,kz);
      }
}

void MUD3D::set_data(const Grid& cof_cr_data_fft){
    // only mud3cr_ use sig function
    if(_type != MUD_CR) return;
    // the set_data only works for fft2mg_mode = 0
    if(_fft2mg_mode != 0) return;
    
    // Here, nx and ny are mg grids.
    // nx = Lx + 1, ny = Ly + 1 in fft2mg_mode = 0
    if(cof_cr_data_fft.Lx() != gmudsig.nx-1 || cof_cr_data_fft.Ly() != gmudsig.ny-1 || cof_cr_data_fft.Lz() != gmudsig.nz-1){
        cout<<"Dielectric constant grid does not match MUD2D!"<<endl;
        throw("Dielectric constant grid does not match MUD2D!");
    }
    blitz::Array<double,3> cof_cr_data_mg1(gmudsig.nx,gmudsig.ny,gmudsig.nz);
    blitz::Array<double,3> cof_cr_data_mg2(gmudsig.nx,gmudsig.ny,gmudsig.nz);
    blitz::Array<double,3> cof_cr_data_mg3(gmudsig.nx,gmudsig.ny,gmudsig.nz);
    blitz::Array<double,3> cof_cr_data_mg4(gmudsig.nx,gmudsig.ny,gmudsig.nz);

    fft2mg_copy_3d(cof_cr_data_fft,cof_cr_data_mg1);
    fft2mg_copy_3d(cof_cr_data_fft.diffx(),cof_cr_data_mg2);
    fft2mg_copy_3d(cof_cr_data_fft.diffy(),cof_cr_data_mg3);
    fft2mg_copy_3d(cof_cr_data_fft.diffz(),cof_cr_data_mg4);

    int index;
    for(int i=0;i<gmudsig.nx;i++)
      for(int j=0;j<gmudsig.ny;j++)
        for(int k=0;k<gmudsig.nz;k++){
          index = gmudsig.nz * (gmudsig.ny * i + j) + k;
          gmudsig.xdata[index] = cof_cr_data_mg1(i,j,k);
          gmudsig.ydata[index] = cof_cr_data_mg2(i,j,k);
          gmudsig.zdata[index] = cof_cr_data_mg3(i,j,k);
          gmudsig.lambda_data[index] = cof_cr_data_mg4(i,j,k);
      }
}

void MUD3D::display() const{
    cout<<"MUD3D mud3"<<type()<<endl;
    cout<<"intl = "<<_iparm[0]<<endl;
    cout<<"nxa = "<<_iparm[1]<<", nxb = "<<_iparm[2];
    cout<<", nyc = "<<_iparm[3]<<", nyd = "<<_iparm[4];
    cout<<", nze = "<<_iparm[5]<<", nzf = "<<_iparm[6]<<endl;

    cout<<"ixp = "<<_iparm[7]<<", jyq = "<<_iparm[8];
    cout<<", kzr = "<<_iparm[9]<<endl;
    cout<<"iex = "<<_iparm[10]<<", jey = "<<_iparm[11];
    cout<<", kez = "<<_iparm[12]<<endl;

    cout<<"nx = "<<_iparm[13]<<", ny = "<<_iparm[14];
    cout<<", nz = "<<_iparm[15]<<endl;
    cout<<"xa = "<<_fparm[0]<<", xb = "<<_fparm[1];
    cout<<", yc = "<<_fparm[2]<<", yd = "<<_fparm[3];
    cout<<", ze = "<<_fparm[4]<<", zf = "<<_fparm[5]<<endl;

    cout<<"iguess = "<<_iparm[16]<<", maxcy = "<<_iparm[17];
    cout<<", tolmax = "<<_fparm[6]<<endl;

    cout<<"method = "<<_iparm[18]<<endl;
    switch(_type){
        case MUD_SP:
            cout<<"work space estimate = "<<_iparm[19]<<endl;
            cout<<"work space minimum = "<<_iparm[20]<<endl;
            break;
        case MUD_SA:
            cout<<"method2 = "<<_iparm[19]<<endl;
            cout<<"work space estimate = "<<_iparm[20]<<endl;
            cout<<"work space minimum = "<<_iparm[21]<<endl;
            break;
        case MUD_CR:
            cout<<"method2 = "<<_iparm[19]<<endl;
            cout<<"work space estimate = "<<_iparm[20]<<endl;
            cout<<"work space minimum = "<<_iparm[21]<<endl;
            cout<<"icrosxy = "<<_icros[0];
            cout<<", icrosxz = "<<_icros[1];
            cout<<", icrosyz = "<<_icros[2]<<endl;
            cout<<"tol = "<<_tol<<", maxit = "<<_maxit;
            cout<<", iouter = "<<_iouter<<endl;
            cout<<"rmax = "<<_rmax;
            break;
        default:
            break;
    }

    cout<<"kcycle = "<<_mgopt[0]<<", iprer = "<<_mgopt[1];
    cout<<", ipost = "<<_mgopt[2]<<", intpol = "<<_mgopt[3]<<endl;

    cout<<"ierror = "<<_ierror<<endl;
    cout<<"fft2mg mode = "<<_fft2mg_mode;
    cout<<", fft2mg interp = "<<_fft2mg_interp<<endl;
    cout<<endl;
}

/**
 * Solve the PDE using MUDPACK
 *      Lu = rhs
 *
 * It is the user who should ensure that
 *      nx = 2*Lx + 1
 *      ny = 2*Ly + 1
 * where
 *      Lx = u.Lx()
 *      Ly = u.Ly()
 *
 */
void MUD3D::solve(Grid &u,const Grid &rhs){
    if(_iparm[0] != 1){
      cout<<"The solver must be initiated before actual solving!"<<endl;
      throw("The solver must be initiated before actual solving!");
    }
    int nx = _iparm[13];
    int ny = _iparm[14];
    int nz = _iparm[15];
    blitz::Array<double,3> mg(nx,ny,nz);
    if(_fft2mg_mode == 0)
        fft2mg_copy_3d(rhs,mg);
    else
        fft2mg_interp_3d(rhs,mg,_fft2mg_interp);

    // C/C++ is row-major while Fortran is column major
    blitz::Array<MUD_FLOAT,3> mg_in(mg.shape(),blitz::fortranArray);
    mg_in = blitz::cast<MUD_FLOAT>(mg);
    blitz::Array<MUD_FLOAT,3> mg_out(mg.shape(),blitz::fortranArray);
    mg_out = 0.0;

    // const_cast here is a must to keep the slove method const. 
    // But the iparm, fparm, work, and ierror may be modified by mud2sp_
    switch(_type){
        case MUD_SP:
            mud3sp_(const_cast<MUD_INT*>(_iparm), \
                    const_cast<MUD_FLOAT*>(_fparm), \
                    const_cast<MUD_FLOAT*>(_work.data()), \
                    _pcofx,_pcofy,_pcofz,_pboundary, \
                    mg_in.data(),mg_out.data(), \
                    const_cast<MUD_INT*>(_mgopt), \
                    const_cast<MUD_INT*>(&_ierror));
            break;
        case MUD_SA:
            init();
            mud3sa_(const_cast<MUD_INT*>(_iparm), \
                    const_cast<MUD_FLOAT*>(_fparm), \
                    const_cast<MUD_FLOAT*>(_work.data()), \
                    _psigx,_psigy,_psigz,_xlmbda,_pboundary, \
                    mg_in.data(),mg_out.data(), \
                    const_cast<MUD_INT*>(_mgopt), \
                    const_cast<MUD_INT*>(&_ierror));
            break;
        case MUD_CR:
            init();
            mud3cr_(const_cast<MUD_INT*>(_iparm), \
                    const_cast<MUD_FLOAT*>(_fparm), \
                    const_cast<MUD_FLOAT*>(_work.data()), \
                    _pcof_cr,_pboundary_cr, \
                    mg_in.data(),mg_out.data(), \
                    const_cast<MUD_INT*>(_mgopt), \
                    _icros,_pcrsxy,_pcrsxz,_pcrsyz, \
                    &_tol,&_maxit,&_iouter,_rmax.data(),
                    const_cast<MUD_INT*>(&_ierror));
            break;
        default:
            cout<<"Unsupported MUDPACK solver!"<<endl;
            throw("Unsupported MUDPACK solver!");
            break;
    }

    mg = blitz::cast<double>(mg_out);
    if(_fft2mg_mode == 0)
        mg2fft_copy_3d(mg,u);
    else
        mg2fft_interp_3d(mg,u);
}

