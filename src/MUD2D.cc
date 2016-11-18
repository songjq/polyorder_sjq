/**
 * MUD2D.cc
 * Created at 2011.6.23
 *
 * Implementation of MUD2D.h.
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

#include "MUD2D.h"

extern "C"{
    void mud2sp_(MUD_INT *iparm,MUD_FLOAT *fparm,MUD_FLOAT *work, \
                 MUD2D::COF_FUNC,MUD2D::COF_FUNC,MUD2D::BND_FUNC, \
                 MUD_FLOAT *rhs,MUD_FLOAT *phi,MUD_INT *mgopt, \
                 MUD_INT *ierror);

    void mud2sa_(MUD_INT *iparm,MUD_FLOAT *fparm,MUD_FLOAT *work, \
                 MUD2D::SIG_FUNC,MUD2D::SIG_FUNC,MUD2D::SIG_FUNC, \
                 MUD2D::BND_FUNC, \
                 MUD_FLOAT *rhs,MUD_FLOAT *phi,MUD_INT *mgopt, \
                 MUD_INT *ierror);

    void mud2cr_(MUD_INT *iparm,MUD_FLOAT *fparm,MUD_FLOAT *work, \
                 MUD2D::COF_CR_FUNC, MUD2D::BND_CR_FUNC, \
                 MUD_FLOAT *rhs,MUD_FLOAT *phi,MUD_INT *mgopt, \
                 MUD_INT *ierror);
}

/**
 * For
 *      pxx + cyy(y)*pyy + cy(y)*py + cey(y)*p(x,y) = r(x,y)
 *
 */
void dummy_cofx(MUD_FLOAT *x,MUD_FLOAT *cxx,MUD_FLOAT *cx,MUD_FLOAT *cex){
    *cxx=1.0;
    *cx=0.0;
    *cex=0.0;
}

/**
 * For
 *      cxx(x)*pyy + pyy + cx(x)*px + cex(x)*p(x,y) = r(x,y)
 * With dummy_cofx, it is for
 *      pxx + pyy = r(x,y)
 * and with dummy_bnd, which suits for
 *      constant eps + Square or Rectangular unit cell
 *
 */
void dummy_cofy(MUD_FLOAT *y,MUD_FLOAT *cyy,MUD_FLOAT *cy,MUD_FLOAT *cey){
    *cyy=1.0;
    *cy=0.0;
    *cey=0.0;
}

/**
 * For mud2sa
 *     d(dp/dx)/dx + d(sigy(x,y)*dp/dy)/dy - xlmbda(x,y)*p(x,y) = r(x,y) 
 *
 */
double dummy_sigx(MUD_FLOAT *x,MUD_FLOAT *y){
    return 1.0;
}

/**
 * For mud2sa
 *     d(sigx(x,y)*dp/dx)/dx + d(dp/dy)/dy - xlmbda(x,y)*p(x,y) = r(x,y) 
 *
 */
double dummy_sigy(MUD_FLOAT *x,MUD_FLOAT *y){
    return 1.0;
}

/**
 * For mud2sa
 *     d(sigx(x,y)*dp/dx)/dx + d(sigy(x,y)*dp/dy)/dy = r(x,y) 
 * with dummy_sigx and dummy_sigy, it is for
 *     pxx + pyy = r(x,y)
 * which reduces to mud2sp
 *
 */
double dummy_xlmbda(MUD_FLOAT *x,MUD_FLOAT *y){
    return 0.0;
}

/**
 * For mud2sp, mud2sa
 *      periodic boundary condition
 *
 */
void dummy_bnd(MUD_INT *kbdy,MUD_FLOAT *xory,MUD_FLOAT *alfa,MUD_FLOAT *gbdy){
}

/**
 * For mud2cr
 *      (4.0/3.0)*(pxx + pxy + pyy)
 * with dummy_bnd_cr, which suits for
 *      constant eps + Hexagonal (2D) unit cell
 *
 */
void dummy_cof_cr(MUD_FLOAT *x,MUD_FLOAT *y,MUD_FLOAT *cxx,MUD_FLOAT *cxy,MUD_FLOAT *cyy,MUD_FLOAT *cx,MUD_FLOAT *cy,MUD_FLOAT *ce){
    *cxx = 4.0 / 3.0;
    *cxy = 4.0 / 3.0;
    *cyy = 4.0 / 3.0;
    *cx = 0.0;
    *cy = 0.0;
    *ce = 0.0;
}

/**
 * For mud2cr
 *      periodic boundary condition
 *
 */
void dummy_bnd_cr(MUD_INT *kbdy,MUD_FLOAT *xory,MUD_FLOAT *alfa,MUD_FLOAT *beta,MUD_FLOAT *gama,MUD_FLOAT *gbdy){
}

MUD2D::MUD2D(const MUD2D &rhs):MUD(rhs){
    set_cof(rhs._pcofx,rhs._pcofy);
    set_cof(rhs._pcof_cr);
    set_sig(rhs._psigx,rhs._psigy);
    set_xlmbda(rhs._xlmbda);
    set_boundary_function(rhs._pboundary);
    set_boundary_function(rhs._pboundary_cr);
    init();
}

MUD2D::MUD2D(string t,int Lx,int Ly,double xb,double yd,double xa,double yc,int fft2mg_mode){
    set_type(t);
    set_cof(dummy_cofx,dummy_cofy);
    set_cof(dummy_cof_cr);
    set_sig(dummy_sigx,dummy_sigy);
    set_xlmbda(dummy_xlmbda);
    set_boundary_function(dummy_bnd);
    set_boundary_function(dummy_bnd_cr);
    set_boundary_type();
    _fft2mg_mode = fft2mg_mode; // this should be set before set_grid
    set_grid(Lx,Ly,xb,yd,xa,yc); // this should call before set_method
    if(_type == MUD_SA)
        set_sig_parm(); // this should be called after set_grid
    else if(_type == MUD_CR)
        set_cof_cr_parm(); // this should be called after set_grid
    set_fft2mg_interp(); // default to linear interpolation
    set_method();
    set_guess();
    set_maxcy();
    set_mgopt();
    set_tolmax();
    //_iparam[15]  //the actual work space lenght output by mud2sp_()
    //_iparam[16]  //outpu for intl=1 calls only
    //_fparam[5]   // output for intl=1 calls with tolmax>0
}

void MUD2D::init(){
    int nx = _iparm[9];
    int ny = _iparm[10];
    blitz::Array<MUD_FLOAT,2> data(nx,ny);
    data = 0.0;

    _iparm[0] = 0;
    int length = _iparm[14];
    _work.resize(length);
    switch(_type){
        case MUD_SP:
            mud2sp_(_iparm,_fparm,_work.data(), \
                    _pcofx,_pcofy,_pboundary, \
                    data.data(),data.data(),_mgopt,&_ierror);
            break;
        case MUD_SA:
            mud2sa_(_iparm,_fparm,_work.data(), \
                    _psigx,_psigy,_xlmbda,_pboundary, \
                    data.data(),data.data(),_mgopt,&_ierror);
            break;
        case MUD_CR:
            mud2cr_(_iparm,_fparm,_work.data(), \
                    _pcof_cr,_pboundary_cr, \
                    data.data(),data.data(),_mgopt,&_ierror);
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
        cout<<"MUD2D error: "<<_ierror<<endl;
        throw("MUD2D error initialization error!");
    }
    _iparm[0]=1; // After initialization call
}

/**
 * Set discretized condition
 * In MUDPACK, the number of grids in each dimension has the form:
 *      p * 2^(i-1) + 1
 *
 * For fft2mg mode 0, the grids of the input data has the form:
 *      L + 1
 * where L is either Lx or Ly. Thus, we have
 *      L = p * 2^(i-1)
 * It requires that L must be even because L + 1 must be odd.
 *
 * For fft2mg mode 1, the grids of the input data has the form:
 *      2 * L + 1
 * Thus, we have
 *      p * 2^(i-1) + 1 = 2 * L + 1
 * That is
 *      L = p * 2^(i-2)
 * It requires that L is better to be even, 
 * otherwise p = L should be very large and not suitable for MUDPACK.
 *
 */
void MUD2D::set_grid(int Lx,int Ly,double xb,double yd,double xa,double yc){
    int nx,ny;
    int ixp,iex;
    int jyq,jey;
    _decompose_L(Lx,ixp,iex);
    _decompose_L(Ly,jyq,jey);
    _iparm[5] = ixp;//ixp;
    _iparm[6] = jyq;//jyq;

    if (_fft2mg_mode == 0){
        iex -= 1;
        jey -= 1;
        nx = Lx + 1;
        ny = Ly + 1;
    }
    else{
        nx = 2*Lx + 1;
        ny = 2*Ly + 1;
    }
    _iparm[7] = iex;//iex;
    _iparm[8] = jey;//jey;
/*
    int nx = 1;
    int ny =1;
    nx <<= (iex-1); // 2^(iex-1)
    nx *= ixp; // nx = ixp*2^(iex-1)
    nx += 1; // nx = ixp*2^(iex-1) + 1
    ny <<= (jey-1); // 2^(iex-1)
    ny *= jyq; // nx = ixp*2^(iex-1)
    ny += 1; // nx = ixp*2^(iex-1) + 1
*/
    _iparm[9] = nx;
    _iparm[10] = ny;

    _fparm[0] = xa;
    _fparm[1] = xb;
    _fparm[2] = yc;
    _fparm[3] = yd;
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
void MUD2D::set_method(int imethod){
    int nxa = _iparm[1];
    int nyc = _iparm[3];
    int nx = _iparm[9];
    int ny = _iparm[10];
    int isx,jsy,length;
    if(imethod == 0){
        isx = 0;
        jsy = 0;
    }
    else if(imethod == 1){
        if(nxa == 0)
          isx = 5;
        else
          isx = 3;
        jsy = 0;
    }
    else if(imethod == 2){
        if(nyc == 0)
          jsy = 5;
        else
          jsy = 3;
        isx = 0;
    }
    else{
        if(nxa == 0)
          isx = 5;
        else
          isx = 3;
        if(nyc == 0)
          jsy = 5;
        else
          jsy = 3;
    }

    switch(_type){
        case MUD_SP:
            length = nx*ny*(5+3*(isx+jsy)/2) + 10*(nx+ny);
            break;
        case MUD_SA:
            length = 4*(nx*ny*(10+isx+jsy) + 8*(nx+ny+2))/3;
            break;
        case MUD_CR:
            length = (7*(nx+2)*(ny+2) + 4*(11+isx+jsy)*nx*ny)/3;
            break;
        default:
            throw("Unsupported MUDPACK solver!");
            break;
    }

    _iparm[13] = imethod;
    _iparm[14] = length; //work space length 
}

/**
 * This must be called after set_grid
 *
 */
void MUD2D::set_sig_parm(){
    // only mud2sa_ use sig function
    if(_type != MUD_SA) return;
    // the set_sig only works for fft2mg_mode = 0
    if(_fft2mg_mode != 0) return;
    
    MUD_INT nx = _iparm[9];
    MUD_INT ny = _iparm[10];
    MUD_INT ixp = _iparm[5];
    MUD_INT jyq = _iparm[6];
    MUD_FLOAT xa = _fparm[0];
    MUD_FLOAT xb = _fparm[1];
    MUD_FLOAT yc = _fparm[2];
    MUD_FLOAT yd = _fparm[3];
    MUD_FLOAT lx = xb - xa;
    MUD_FLOAT ly = yd - yc;
    MUD_FLOAT dx = lx / (nx - 1);
    MUD_FLOAT dy = ly / (ny - 1);
    gmudsig.dx = 0.5 * dx;
    gmudsig.dy = 0.5 * dy;
    gmudsig.xa = xa - lx / (2. * ixp);
    gmudsig.yc = yc - ly / (2. * jyq);
    gmudsig.xb = xb + lx / (2. * ixp);
    gmudsig.yd = yd + ly / (2. * jyq);
    gmudsig.nx = (int)((gmudsig.xb - gmudsig.xa) / gmudsig.dx + 0.01) + 1;
    gmudsig.ny = (int)((gmudsig.yd - gmudsig.yc) / gmudsig.dy + 0.01) + 1;

    // since sigx and sigy are the same in scft+pe case, use sigx for both.
    delete gmudsig.xdata;
    gmudsig.xdata = new MUD_FLOAT[gmudsig.nx*gmudsig.ny];
}

/**
 * This must be called after set_grid
 * For mud2cr usage only
 *
 */
void MUD2D::set_cof_cr_parm(){
    // only mud2cr_ use this function
    if(_type != MUD_CR) return;
    // the set_sig only works for fft2mg_mode = 0
    if(_fft2mg_mode != 0) return;
    
    MUD_INT nx = _iparm[9];
    MUD_INT ny = _iparm[10];
    MUD_FLOAT xa = _fparm[0];
    MUD_FLOAT xb = _fparm[1];
    MUD_FLOAT yc = _fparm[2];
    MUD_FLOAT yd = _fparm[3];
    MUD_FLOAT lx = xb - xa;
    MUD_FLOAT ly = yd - yc;
    MUD_FLOAT dx = lx / (nx - 1);
    MUD_FLOAT dy = ly / (ny - 1);
    gmudsig.dx = dx;
    gmudsig.dy = dy;
    gmudsig.xa = xa;
    gmudsig.yc = yc;
    gmudsig.xb = xb;
    gmudsig.yd = yd;
    gmudsig.nx = nx;
    gmudsig.ny = ny;

    delete gmudsig.xdata;
    delete gmudsig.ydata;
    delete gmudsig.zdata;
    gmudsig.xdata = new MUD_FLOAT[gmudsig.nx*gmudsig.ny];
    gmudsig.ydata = new MUD_FLOAT[gmudsig.nx*gmudsig.ny];
    gmudsig.zdata = new MUD_FLOAT[gmudsig.nx*gmudsig.ny];
}

/**
 * Set global data for sig function use.
 *
 */
void MUD2D::set_data(const blitz::Array<double,2> sig_data_fft){
    // only mud2sa_ use sig function
    if(_type != MUD_SA) return;
    // the set_sig only works for fft2mg_mode = 0
    if(_fft2mg_mode != 0) return;
    
    MUD_INT nx = _iparm[9];
    MUD_INT ny = _iparm[10];
    MUD_FLOAT xa = _fparm[0];
    MUD_FLOAT yc = _fparm[2];

    // Here, nx and ny are mg grids.
    // nx = Lx + 1, ny = Ly + 1 in fft2mg_mode = 0
    if(sig_data_fft.rows() != nx-1 || sig_data_fft.cols() != ny-1){
        cout<<"Dielectric constant grid does not match MUD2D!"<<endl;
        throw("Dielectric constant grid does not match MUD2D!");
    }
    blitz::Array<double,2> sig_data_mg1(nx,ny);
    blitz::Array<double,2> sig_data_mg2(2*nx-1,2*ny-1);

    fft2mg_copy_2d(sig_data_fft,sig_data_mg1);
    interp_bilinear_2d(sig_data_mg1,sig_data_mg2);

    int index;
    int i_xa = (int)((xa - gmudsig.xa) / gmudsig.dx + 0.01);
    int j_yc = (int)((yc - gmudsig.yc) / gmudsig.dy + 0.01);
    int period_x = 2 * (nx - 1);
    int period_y = 2 * (ny - 1);
    int ix,jy;
    for(int i=0;i<gmudsig.nx;i++)
      for(int j=0;j<gmudsig.ny;j++){
          ix = i - i_xa;
          if(ix < 0) ix += period_x;
          else if(ix > period_x) ix -= period_x;

          jy = j - j_yc;
          if(jy < 0) jy += period_y;
          else if(jy > period_y) jy -= period_y;

          index = gmudsig.ny * i + j;
          gmudsig.xdata[index] = sig_data_mg2(ix,jy);
      }
}

/**
 * Set global data for mud2cr cof function use.
 *
 */
void MUD2D::set_data(const Grid& cof_cr_data_fft){
    // only mud2sa_ use cof function
    if(_type != MUD_CR) return;
    // the set_sig only works for fft2mg_mode = 0
    if(_fft2mg_mode != 0) return;
    
    // Here, nx and ny are mg grids.
    // nx = Lx + 1, ny = Ly + 1 in fft2mg_mode = 0
    if(cof_cr_data_fft.Lx() != gmudsig.nx-1 || cof_cr_data_fft.Ly() != gmudsig.ny-1){
        cout<<"Dielectric constant grid does not match MUD2D!"<<endl;
        throw("Dielectric constant grid does not match MUD2D!");
    }
    blitz::Array<double,2> cof_cr_data_mg1(gmudsig.nx,gmudsig.ny);
    blitz::Array<double,2> cof_cr_data_mg2(gmudsig.nx,gmudsig.ny);
    blitz::Array<double,2> cof_cr_data_mg3(gmudsig.nx,gmudsig.ny);

    fft2mg_copy_2d(cof_cr_data_fft,cof_cr_data_mg1);
    fft2mg_copy_2d(cof_cr_data_fft.diffx(),cof_cr_data_mg2);
    fft2mg_copy_2d(cof_cr_data_fft.diffy(),cof_cr_data_mg3);

    int index;
    for(int i=0;i<gmudsig.nx;i++)
      for(int j=0;j<gmudsig.ny;j++){
          index = gmudsig.ny * i + j;
          gmudsig.xdata[index] = cof_cr_data_mg1(i,j);
          gmudsig.ydata[index] = cof_cr_data_mg2(i,j);
          gmudsig.zdata[index] = cof_cr_data_mg3(i,j);
      }
}

void MUD2D::display() const{
    cout<<"MUD2D mud2"<<type()<<endl;
    cout<<"intl = "<<_iparm[0]<<endl;
    cout<<"nxa = "<<_iparm[1]<<", nxb = "<<_iparm[2];
    cout<<", nyc = "<<_iparm[3]<<", nyd = "<<_iparm[4]<<endl;
    cout<<"ixp = "<<_iparm[5]<<", jyq = "<<_iparm[6];
    cout<<", iex = "<<_iparm[7]<<", jey = "<<_iparm[8]<<endl;
    cout<<"nx = "<<_iparm[9]<<", ny = "<<_iparm[10]<<endl;
    cout<<"xa = "<<_fparm[0]<<", xb = "<<_fparm[1];
    cout<<", yc = "<<_fparm[2]<<", yd = "<<_fparm[3]<<endl;
    cout<<"iguess = "<<_iparm[11]<<", maxcy = "<<_iparm[12];
    cout<<", tolmax = "<<_fparm[4]<<endl;
    cout<<"method = "<<_iparm[13]<<endl;
    cout<<"work space estimate = "<<_iparm[14]<<endl;
    cout<<"work space minimum = "<<_iparm[15]<<endl;
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
 * for fft2mg mode 1, and 
 *      nx = Lx + 1
 *      ny = Ly + 1
 * for fft2mg mode 0,
 * where
 *      Lx = u.Lx()
 *      Ly = u.Ly()
 *
 */
void MUD2D::solve(Grid &u,const Grid &rhs){
    if(_iparm[0] != 1){
      cout<<"The solver must be initiated before actual solving!"<<endl;
      throw("The solver must be initiated before actual solving!");
    }
    int nx = _iparm[9];
    int ny = _iparm[10];
    blitz::Array<double,2> mg(nx,ny);
    /** if u and rhs are already in the size of nx x ny: 
    blitz::Range all = blitz::Range::all();
    blitz::Array<double,3> fft_data(rhs.data());
    blitz::Array<double,2> fft_data_2d(fft_data(all,all,0));
    mg = fft_data_2d;
    */
    if(_fft2mg_mode == 0)
        fft2mg_copy_2d(rhs,mg);
    else
        fft2mg_interp_2d(rhs,mg,_fft2mg_interp);

    // C/C++ is row-major while Fortran is column major
    blitz::Array<MUD_FLOAT,2> mg_in(mg.shape(),blitz::fortranArray);
    mg_in = blitz::cast<MUD_FLOAT>(mg);
    blitz::Array<MUD_FLOAT,2> mg_out(mg.shape(),blitz::fortranArray);
    mg_out = 0.0;
    /** if u and rhs are already in the size of nx x ny: 
    blitz::Array<double,3> fft_out(u.data());
    blitz::Array<double,2> fft_out_2d(fft_out(all,all,0));
    mg_out = blitz::cast<MUD_FLOAT>(fft_out_2d);
    */

    // const_cast here is a must to keep the slove method const. 
    // But the iparm, fparm, work, and ierror may be modified by mud2sp_
    switch(_type){
        case MUD_SP:
            mud2sp_(const_cast<MUD_INT*>(_iparm), \
                    const_cast<MUD_FLOAT*>(_fparm), \
                    const_cast<MUD_FLOAT*>(_work.data()), \
                    _pcofx,_pcofy,_pboundary, \
                    mg_in.data(),mg_out.data(), \
                    const_cast<MUD_INT*>(_mgopt), \
                    const_cast<MUD_INT*>(&_ierror));
            break;
        case MUD_SA:
            // In a typical scft calculation, mud2in will change due to
            // the variation of the dielectric constant.
            // Thus requires re-initialization.
            init();
            mud2sa_(const_cast<MUD_INT*>(_iparm), \
                    const_cast<MUD_FLOAT*>(_fparm), \
                    const_cast<MUD_FLOAT*>(_work.data()), \
                    _psigx,_psigy,_xlmbda,_pboundary, \
                    mg_in.data(),mg_out.data(), \
                    const_cast<MUD_INT*>(_mgopt), \
                    const_cast<MUD_INT*>(&_ierror));
            break;
        case MUD_CR:
            // Hexagonal unit cell + position-dependent diecelectric
            // constant needs re-initialization.
            // Here we also let the Hexagonal unit cell + position-
            // independent dielectric constant also re-initialization.
            init();
            mud2cr_(const_cast<MUD_INT*>(_iparm), \
                    const_cast<MUD_FLOAT*>(_fparm), \
                    const_cast<MUD_FLOAT*>(_work.data()), \
                    _pcof_cr,_pboundary_cr, \
                    mg_in.data(),mg_out.data(), \
                    const_cast<MUD_INT*>(_mgopt), \
                    const_cast<MUD_INT*>(&_ierror));
            break;
        default:
            cout<<"Unsupported MUDPACK solver!"<<endl;
            throw("Unsupported MUDPACK solver!");
            break;
    }

    mg = blitz::cast<double>(mg_out);
    /** if u and rhs are already in the size of nx x ny:
    fft_out_2d = mg;
    */
    if(_fft2mg_mode == 0)
        mg2fft_copy_2d(mg,u);
    else
        mg2fft_interp_2d(mg,u);
}

