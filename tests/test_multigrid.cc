/**
 * Test MUD, MUD2D, MUD3D
 *
 * Copyright@ Yi-Xin Liu 2012
 * 
 */

#include "MUD3D.h"

#include <iostream>
#include <string>
#include <ctime>
#include <cmath>

#include "multigrid.h"
#include "common.h"
#include "MUD.h"
#include "MUD2D.h"

using namespace std;

void test_fft2mg_copy(){
    Config cfg("param.ini");
    UnitCell uc(cfg);

    // fft2mg, mg2fft, 1D
    int Lx = 9;
    int Ly = 1;
    int Lz = 1;
    Grid fft(uc,Lx,Ly,Lz);
    blitz::firstIndex i;
    fft.data() = i + 1;
    cout<<"fft= "<<fft.data()<<endl;

    blitz::Array<double,1> mg(Lx+1);
    fft2mg_copy_1d(fft,mg);
    cout<<"mg= "<<mg<<endl;

    fft.data() = 0;
    cout<<"fft= (0) "<<fft.data()<<endl;
    mg2fft_copy_1d(mg,fft);
    cout<<"fft= "<<fft.data()<<endl;

    // fft2mg, mg2fft, 2D
    Lx = 4;
    Ly = 5;
    Lz = 1;
    Grid fft2(uc,Lx,Ly,Lz);
    blitz::secondIndex j;
    fft2.data() = Ly * i + j + 1;
    cout<<"fft2= "<<fft2.data()<<endl;

    blitz::Array<double,2> mg2(Lx+1,Ly+1);
    fft2mg_copy_2d(fft2,mg2);
    cout<<"mg2= "<<mg2<<endl;

    fft2.data() = 0;
    cout<<"fft2= (0) "<<fft2.data()<<endl;
    mg2fft_copy_2d(mg2,fft2);
    cout<<"fft2= "<<fft2.data()<<endl;

    // fft2mg, mg2fft, 3D
    Lx = 3;
    Ly = 3;
    Lz = 3;
    Grid fft3(uc,Lx,Ly,Lz);
    blitz::thirdIndex k;
    fft3.data() = Lz * ((Ly * i) + j) + k + 1;
    cout<<"fft3= "<<fft3.data()<<endl;

    blitz::Array<double,3> mg3(Lx+1,Ly+1,Lz+1);
    fft2mg_copy_3d(fft3,mg3);
    cout<<"mg3= "<<mg3<<endl;

    fft3.data() = 0;
    cout<<"fft3= (0) "<<fft3.data()<<endl;
    mg2fft_copy_3d(mg3,fft3);
    cout<<"fft2= "<<fft3.data()<<endl;
}

void test_fft2mg_interp(){
    Config cfg("param.ini");
    UnitCell uc(cfg);

    // fft2mg, mg2fft, 1D
    int Lx = 9;
    int Ly = 1;
    int Lz = 1;
    Grid fft(uc,Lx,Ly,Lz);
    blitz::firstIndex i;
    fft.data() = i + 1;
    cout<<"fft= "<<fft.data()<<endl;

    blitz::Array<double,1> mg(2*Lx+1);
    fft2mg_interp_1d(fft,mg);
    cout<<"mg= "<<mg<<endl;

    fft.data() = 0;
    cout<<"fft= (0) "<<fft.data()<<endl;
    mg2fft_interp_1d(mg,fft);
    cout<<"fft= "<<fft.data()<<endl;

    // fft2mg, mg2fft, 2D
    Lx = 4;
    Ly = 5;
    Lz = 1;
    Grid fft2(uc,Lx,Ly,Lz);
    blitz::secondIndex j;
    fft2.data() = Ly * i + j + 1;
    cout<<"fft2= "<<fft2.data()<<endl;

    blitz::Array<double,2> mg2(2*Lx+1,2*Ly+1);
    fft2mg_interp_2d(fft2,mg2);
    cout<<"mg2= "<<mg2<<endl;

    fft2.data() = 0;
    cout<<"fft2= (0) "<<fft2.data()<<endl;
    mg2fft_interp_2d(mg2,fft2);
    cout<<"fft2= "<<fft2.data()<<endl;

    // fft2mg, mg2fft, 3D
    Lx = 3;
    Ly = 3;
    Lz = 3;
    Grid fft3(uc,Lx,Ly,Lz);
    blitz::thirdIndex k;
    fft3.data() = Lz * ((Ly * i) + j) + k + 1;
    cout<<"fft3= "<<fft3.data()<<endl;

    blitz::Array<double,3> mg3(2*Lx+1,2*Ly+1,2*Lz+1);
    fft2mg_interp_3d(fft3,mg3);
    cout<<"mg3= "<<mg3<<endl;

    fft3.data() = 0;
    cout<<"fft3= (0) "<<fft3.data()<<endl;
    mg2fft_interp_3d(mg3,fft3);
    cout<<"fft2= "<<fft3.data()<<endl;
}

void test_multigrid_RP_operators(){
    int nc = 8;
    blitz::Array<double,1> uc(nc);
    blitz::firstIndex i;
    uc = i + 1;
    cout<<"uc= "<<uc<<endl;

    blitz::Array<double,1> uf(2*nc-1);
    interp_linear_1d(uc,uf);
    cout<<"interp uf= "<<uf<<endl;

    uc = 0;
    rstrct_injection_1d(uf,uc);
    cout<<"restrict_injection uc= "<<uc<<endl;

    uc = 0;
    rstrct_weight_1d(uf,uc);
    cout<<"restrict_weight uc= "<<uc<<endl;

    int ncx = 4;
    int ncy = 5;
    blitz::Array<double,2> uc2(ncx,ncy);
    blitz::secondIndex j;
    uc2 = ncy * i + j + 1;
    cout<<"uc2= "<<uc2<<endl;

    blitz::Array<double,2> uf2(2*ncx-1,2*ncy-1);
    interp_bilinear_2d(uc2,uf2);
    cout<<"interp_bilinear uf2= "<<uf2<<endl;

    uc2 = 0;
    rstrct_injection_2d(uf2,uc2);
    cout<<"restrict_injection uc2= "<<uc2<<endl;

    uc2 = 0;
    rstrct_hw_2d(uf2,uc2);
    cout<<"restrict_hw uc2= "<<uc2<<endl;

    uc2 = 0;
    rstrct_fw_2d(uf2,uc2);
    cout<<"restrict_fw uc2= "<<uc2<<endl;

    ncx = 5;
    ncy = 4;
    int ncz = 3;
    blitz::Array<double,3> uc3(ncx,ncy,ncz);
    blitz::thirdIndex k;
    uc3 = ncz * (ncy * i + j) + k + 1;
    cout<<"uc3= "<<uc3<<endl;

    blitz::Array<double,3> uf3(2*ncx-1,2*ncy-1,2*ncz-1);
    interp_trilinear_3d(uc3,uf3);
    cout<<"interp_trilinear uf3= "<<uf3<<endl;

    uc3 = 0;
    rstrct_injection_3d(uf3,uc3);
    cout<<"restrict_injection uc3= "<<uc3<<endl;

    uc3 = 0;
    rstrct_hw_3d(uf3,uc3);
    cout<<"restrict_hw uc3= "<<uc3<<endl;

    uc3 = 0;
    rstrct_fw_3d(uf3,uc3);
    cout<<"restrict_fw uc3= "<<uc3<<endl;
}

void cof_cr(MUD_FLOAT* x,MUD_FLOAT* y,MUD_FLOAT* cxx,MUD_FLOAT* cxy,MUD_FLOAT* cyy,MUD_FLOAT* cx,MUD_FLOAT* cy,MUD_FLOAT* ce){
    MUD_FLOAT rx = *x;
    MUD_FLOAT ry = *y;
    *cxx = 1. + ry*ry;
    *cxy = 2.*rx*ry;
    *cyy = 1. + rx*rx;
    *cx = ry;
    *cy = rx;
    *ce = -rx*ry;
}

void bnd_cr(MUD_INT* kbdy,MUD_FLOAT* xory,MUD_FLOAT* alfa,MUD_FLOAT* beta,MUD_FLOAT* gama,MUD_FLOAT* gbdy){
    if(*kbdy == 4){
        MUD_FLOAT x = *xory;
        MUD_FLOAT y = 1.0;
        *alfa = -x;
        *beta = 1. + x;
        *gama = -x;
        MUD_FLOAT px = 5.*x*x*x*x*y*y*y*y*y;
        MUD_FLOAT py = 5.*x*x*x*x*x*y*y*y*y;
        MUD_FLOAT pe = x*x*x*x*x*y*y*y*y*y;
        *gbdy = (*alfa)*px + (*beta)*py + (*gama)*pe;
    }
}

void exact_cr(MUD_FLOAT* x,MUD_FLOAT* y,MUD_FLOAT* pxx,MUD_FLOAT* pxy,MUD_FLOAT* pyy,MUD_FLOAT* px,MUD_FLOAT* py,MUD_FLOAT* pe){
    MUD_FLOAT rx = *x;
    MUD_FLOAT ry = *y;
    *pe = rx*rx*rx*rx*rx*ry*ry*ry*ry*ry;
    *px = 5.*rx*rx*rx*rx*ry*ry*ry*ry*ry;
    *py = 5.*rx*rx*rx*rx*rx*ry*ry*ry*ry;
    *pxx = 20.*rx*rx*rx*ry*ry*ry*ry*ry;
    *pxy = 25.*rx*rx*rx*rx*ry*ry*ry*ry;
    *pyy = 20.*rx*rx*rx*rx*rx*ry*ry*ry;
}

/**
 * Test MUD2D.mud2cr
 *
 * Test case (from tmud2cr.f in MUDPACK):
 *      (1.+y^2)*u_xx + (1.+x^2)*u_yy + 2.*x*y*u_xy +
 *      y*u_x + x*u_y - x*y*u = f(x,y)
 * The exact solution is
 *      u = (x*y)^5
 *
 * Max error = 6.23e-4, the same as tmud2cr.f
 *
 * Note: do not use mg2fft or fft2mg functions.
 *
 */
void test_mud2d_cr(){
    Config cfg("param.ini");
    UnitCell uc(cfg);

    //double epsilon=1.0;
    double lx = 1.0;
    double ly = 1.0; //sqrt(lx/epsilon);
    int Lx = 49;
    int Ly = 65;
    Grid u0(uc,Lx,Ly,1);
    Grid u(uc,Lx,Ly,1);
    Grid f(uc,Lx,Ly,1);
    double dx = lx / (Lx-1);
    double dy = ly / (Ly-1);
    MUD_FLOAT x,y;
    MUD_FLOAT cxx,cxy,cyy,cx,cy,ce;
    MUD_FLOAT pxx,pxy,pyy,px,py,pe;
    for(int i=0;i<Lx;i++)
      for(int j=0;j<Ly;j++){
          x = i * dx;
          y = j * dy;
          cof_cr(&x,&y,&cxx,&cxy,&cyy,&cx,&cy,&ce);
          exact_cr(&x,&y,&pxx,&pxy,&pyy,&px,&py,&pe);
          f(i,j) = cxx*pxx + cxy*pxy + cyy*pyy + cx*px + cy*py + ce*pe;
          u0(i,j) = x*x*x*x*x*y*y*y*y*y; 
          //f(i,j) = -2*(y-y*y)-2*epsilon*(x-x*x);
          //u0(i,j) = (x-x*x)*(y-y*y);
      }

    u.data() = 0.0;
    for(int j=0;j<Ly;j++){
        x = 0.0;
        y = j * dy;
        exact_cr(&x,&y,&pxx,&pxy,&pyy,&px,&py,&pe);
        u(0,j) = pe;
        x = 1.0;
        exact_cr(&x,&y,&pxx,&pxy,&pyy,&px,&py,&pe);
        u(Lx-1,j) = pe;
    }
    for(int i=0;i<Lx;i++){
        x = i * dx;
        y = 0.0;
        exact_cr(&x,&y,&pxx,&pxy,&pyy,&px,&py,&pe);
        u(i,0) = pe;
    }

    MUD2D *pmg=new MUD2D("cr",Lx-1,Ly-1,lx,ly);
    cout<<"Mudpack type "<<pmg->type()<<endl;
    pmg->set_cof(cof_cr);
    pmg->set_boundary_function(bnd_cr);
    pmg->set_boundary_type(1,1,1,2);
    pmg->set_mgopt(2,2,1,3);
    pmg->display();
    pmg->init();
    pmg->display();

    pmg->solve(u,f);

    u.set_name("u");
    u.save("u.mat");
    u0.set_name("u0");
    u0.save("u.mat");

    int nx = Lx;
    int ny = Ly;
    cout<<"u(0,0)(0,ny-1)(nx-1,0)(nx-1,ny-1)  ";
    cout<<u(0,0)<<","<<u(0,ny-1)<<",";
    cout<<u(nx-1,0)<<","<<u(nx-1,ny-1)<<endl;
    cout<<"u0(0,0)(0,ny-1)(nx-1,0)(nx-1,ny-1)  ";
    cout<<u0(0,0)<<","<<u0(0,ny-1)<<",";
    cout<<u0(nx-1,0)<<","<<u0(nx-1,ny-1)<<endl;
    cout<<"u(1,1)(1,np-2)(nt-2,1)(nt-2,np-2)  ";
    cout<<u(1,1)<<","<<u(1,ny-2)<<",";
    cout<<u(nx-2,1)<<","<<u(nx-2,ny-2)<<endl;
    cout<<"u0(1,1)(1,ny-2)(nx-2,1)(nx-2,ny-2)  ";
    cout<<u0(1,1)<<","<<u0(1,ny-2)<<",";
    cout<<u0(nx-2,1)<<","<<u0(nx-2,ny-2)<<endl;
    cout<<"u.mean = () "<<u.mean()<<endl;
    cout<<"u0.mean = () "<<u0.mean()<<endl;
    Grid uu1(u-u0);
    double err_norm1=uu1.abs_mean();
    Grid uu2(uu1*uu1);
    double err_norm2=sqrt(uu2.mean());
    cout<<"max error: "<<blitz::max(blitz::abs(uu1.data()))<<endl;
    cout<<"infinite norm of error: "<<err_norm1<<endl;
    cout<<"L2 norm of error: "<<err_norm2<<endl;
    cout<<endl;
}

double sigx0(MUD_FLOAT *x,MUD_FLOAT *y){
    return sin(*x) * (1.5 + sin(*x) * sin(*x) * cos(*y) * cos(*y));
}

double sigy0(MUD_FLOAT *x,MUD_FLOAT *y){
    if(abs(sin(*x)) > 1.0e-7)
        return (1.5 + sin(*x) * sin(*x) * cos(*y) * cos(*y))/sin(*x);
    else
        return 1.0;
}

double xlambda0(MUD_FLOAT *x,MUD_FLOAT *y){
    return sin(*x) * (1.5 + sin(*x) * sin(*x) * cos(*y) * cos(*y));
}

void bnd(MUD_INT *kbdy,MUD_FLOAT *xory,MUD_FLOAT *alfa,MUD_FLOAT *gbdy){
}

/**
 * Test mud2sa_
 * Test case is from tmud2sa.f in MUDPACK 5.0.1
 * Test passed. 2012.4.16.
 */
void test_mud2d_sa0(){
    Config cfg("param.ini");
    UnitCell uc(cfg);

    double ta = 0.0;
    double tb = PI;
    double pc = 0.0;
    double pd = 2.0 * PI;
    double lt = tb - ta;
    double lp = pd - pc;
    int nt = 40;
    int np = 80;
    Grid u0(uc,nt,np,1);
    Grid u(uc,nt,np,1);
    Grid f(uc,nt,np,1);
    double dlt = lt / (nt-1);
    double dlp = lp / (np-1);
    double t;
    double p;
    double sint,cost,sinp,cosp,sigma,dsigdt,dsigdp;
    double ctt,cpp,ct,cp,ce,tmp,dt,dp,dtt,dpp,ue,ut,up,utt,upp;
    for(int i=0;i<nt;i++)
      for(int j=0;j<np;j++){
          t = ta + (i+0.5) * dlt;
          p = pc + (j+0.5) * dlp;
          sint = sin(t);
          cost = cos(t);
          sinp = sin(p);
          cosp = cos(p);
          sigma = 1.5 + (sint*cosp)*(sint*cosp);
          dsigdt = 2.0*cosp*cosp*sint*cost;
          dsigdp = -2.0*sint*sint*cosp*sinp;
          ctt = sint*sigma;
          cpp = sigma/sint;
          ct = sint*dsigdt + cost*sigma;
          cp = dsigdp/sint;
          ce = -sint*sigma;
          tmp = (sint*cosp*sint*sinp*cost);
          dt = (2.*sint*cost*cost-sint*sint*sint)*(cosp*sinp);
          dp = (cosp*cosp-sinp*sinp)*(sint*sint*cost);
          dtt = (2.*cost*cost*cost-4.*cost*sint*sint-3.*sint*sint*cost)*(cosp*sinp);
          dpp = (-4.*cosp*sinp)*(sint*sint*cost);
          ue = tmp*tmp;
          u0(i,j) = ue;
          ut = 2.*tmp*dt;
          up = 2.*tmp*dp;
          utt = 2.*(dt*dt+tmp*dtt);
          upp = 2.*(dp*dp+tmp*dpp);
          f(i,j) = ctt*utt+cpp*upp+ct*ut+cp*up+ce*ue;
      }
    /*
    for(int j=0;j<np;j++){
        f(0,j) = 0.0;
        f(nt-1,j) = 0.0;
    }
    */

    MUD2D *pmg=new MUD2D("sa",nt,np,tb,pd,ta,pc);
    cout<<"Mudpack type "<<pmg->type()<<endl;
    pmg->set_sig(sigx0,sigy0);
    pmg->set_xlmbda(xlambda0);
    pmg->set_boundary_function(bnd);
    pmg->set_boundary_type(1,1,0,0);
    //pmg->set_fft2mg_interp(1); // use bicubic interpolation for fft2mg
    pmg->set_method(2);
    pmg->set_mgopt(2,2,1,3);
    pmg->display();
    pmg->init();
    pmg->display();

    pmg->solve(u,f);

    u.set_name("u");
    u.save("u.mat");
    u0.set_name("u0");
    u0.save("u.mat");

    cout<<"u(0,0)(0,np-1)(nt-1,0)(nt-1,np-1)  ";
    cout<<u(0,0)<<","<<u(0,np-1)<<",";
    cout<<u(nt-1,0)<<","<<u(nt-1,np-1)<<endl;
    cout<<"u0(0,0)(0,ny-1)(nx-1,0)(nx-1,ny-1)  ";
    cout<<u0(0,0)<<","<<u0(0,np-1)<<",";
    cout<<u0(nt-1,0)<<","<<u0(nt-1,np-1)<<endl;
    cout<<"u(1,1)(1,np-2)(nt-2,1)(nt-2,np-2)  ";
    cout<<u(1,1)<<","<<u(1,np-2)<<",";
    cout<<u(nt-2,1)<<","<<u(nt-2,np-2)<<endl;
    cout<<"u0(1,1)(1,ny-2)(nx-2,1)(nx-2,ny-2)  ";
    cout<<u0(1,1)<<","<<u0(1,np-2)<<",";
    cout<<u0(nt-2,1)<<","<<u0(nt-2,np-2)<<endl;
    cout<<"u.mean = () "<<u.mean()<<endl;
    cout<<"u0.mean = () "<<u0.mean()<<endl;
    Grid uu1(u-u0);
    double err_norm1=uu1.abs_mean();
    Grid uu2(uu1*uu1);
    double err_norm2=sqrt(uu2.mean());
    cout<<"max error: "<<blitz::max(blitz::abs(uu1.data()))<<endl;
    cout<<"infinite norm of error: "<<err_norm1<<endl;
    cout<<"L2 norm of error: "<<err_norm2<<endl;
    cout<<endl;

    delete pmg;
    pmg=new MUD2D("sa",nt,np,tb,pd,ta,pc,1);
    cout<<"Mudpack type "<<pmg->type()<<endl;
    pmg->set_sig(sigx0,sigy0);
    pmg->set_xlmbda(xlambda0);
    pmg->set_boundary_function(bnd);
    pmg->set_boundary_type(1,1,0,0);
    //pmg->set_fft2mg_interp(1); // use bicubic interpolation for fft2mg
    pmg->set_method(2);
    pmg->set_mgopt(2,2,1,3);
    pmg->display();
    pmg->init();
    pmg->display();

    pmg->solve(u,f);

    u.set_name("u");
    u.save("u.mat");
    u0.set_name("u0");
    u0.save("u.mat");

    cout<<"u(0,0)(0,np-1)(nt-1,0)(nt-1,np-1)  ";
    cout<<u(0,0)<<","<<u(0,np-1)<<",";
    cout<<u(nt-1,0)<<","<<u(nt-1,np-1)<<endl;
    cout<<"u0(0,0)(0,ny-1)(nx-1,0)(nx-1,ny-1)  ";
    cout<<u0(0,0)<<","<<u0(0,np-1)<<",";
    cout<<u0(nt-1,0)<<","<<u0(nt-1,np-1)<<endl;
    cout<<"u(1,1)(1,np-2)(nt-2,1)(nt-2,np-2)  ";
    cout<<u(1,1)<<","<<u(1,np-2)<<",";
    cout<<u(nt-2,1)<<","<<u(nt-2,np-2)<<endl;
    cout<<"u0(1,1)(1,ny-2)(nx-2,1)(nx-2,ny-2)  ";
    cout<<u0(1,1)<<","<<u0(1,np-2)<<",";
    cout<<u0(nt-2,1)<<","<<u0(nt-2,np-2)<<endl;
    cout<<"u.mean = () "<<u.mean()<<endl;
    cout<<"u0.mean = () "<<u0.mean()<<endl;
    uu1 = u - u0;
    err_norm1=uu1.abs_mean();
    uu2 = uu1 * uu1;
    err_norm2=sqrt(uu2.mean());
    cout<<"max error: "<<blitz::max(blitz::abs(uu1.data()))<<endl;
    cout<<"infinite norm of error: "<<err_norm1<<endl;
    cout<<"L2 norm of error: "<<err_norm2<<endl;
    cout<<endl;
}

extern "C"{
struct {
    int nx;
    int ny;
    MUD_FLOAT dx;
    MUD_FLOAT dy;
    MUD_FLOAT xa;
    MUD_FLOAT yc;
    MUD_FLOAT xb;
    MUD_FLOAT yd;
    MUD_FLOAT *sinxat;
    MUD_FLOAT *cosyat;
    MUD_FLOAT *data;
} mud2in;

double sig_global(MUD_FLOAT *x,MUD_FLOAT *y){
    MUD_FLOAT dx = mud2in.dx;
    MUD_FLOAT dy = mud2in.dy;
    MUD_FLOAT xa = mud2in.xa;
    MUD_FLOAT yc = mud2in.yc;
    MUD_FLOAT xb = mud2in.xb;
    MUD_FLOAT yd = mud2in.yd;
    int ny = mud2in.ny;
    MUD_FLOAT x0 = xa - (xb - xa) / 4.0;
    MUD_FLOAT y0 = yc - (yd - yc) / 4.0;
    int i = (int)((*x - x0) / dx + 0.01);
    int j = (int)((*y - y0) / dy + 0.01);
    int index = ny * i + j;

    /*
    printf("dx=%f,dy=%f,xa=%f,yc=%f\n",dx,dy,xa,yc);
    printf("sig_nx=%d,sig_ny=%d\n",nx,ny);
    printf("x=%f,y=%f\n",*x,*y);
    printf("i=%d,j=%d\n",i,j);
    */

    /*
    MUD_FLOAT ret = 1.5 + mud2in.sinxat[i]*mud2in.sinxat[i]* \
                    mud2in.cosyat[j]*mud2in.cosyat[j];
    return ret;
    */

    /*
    printf("sinx=%f,cosy=%f\n",sin(*x),cos(*y));
    printf("sinxat=%f,cosyat=%f\n",mud2in.sinxat[i],mud2in.cosyat[j]);
    printf("ret=%f\n",ret);
    */

    return mud2in.data[index];
}

} // end of extern "C"

double sig(MUD_FLOAT *x,MUD_FLOAT *y){
    return 1.5 + sin(*x) * sin(*x) * cos(*y) * cos(*y);
}

double xlambda(MUD_FLOAT *x,MUD_FLOAT *y){
    return 0.0;
}

void test_mud2d_sa_global(){
    Config cfg("param.ini");
    UnitCell uc(cfg);

    double lx = 2.0 * PI;
    double ly = 2.0 * PI;
    int nx = 128;
    int ny = 128;
    double dx = lx / nx;
    double dy = ly / ny;
    double xa = PI * 3.0 / 4.0;
    double xb = xa + lx;
    double yc = PI * 3.0 / 4.0;
    double yd = yc + ly;
    Grid u0(uc,nx,ny,1);
    Grid u(uc,nx,ny,1);
    Grid f(uc,nx,ny,1);
    double x;
    double y;
    double sinx,cosx,siny,cosy,sigma;

    mud2in.dx = 0.5 * dx;
    mud2in.dy = 0.5 * dy;
    mud2in.xa = xa;
    mud2in.yc = yc;
    mud2in.xb = xb;
    mud2in.yd = yd; 
    //        *....@....*....@....*....@....*
    //     xa-lx/4                        xb+lx/4
    mud2in.nx = (int)(1.5 * lx / mud2in.dx + 0.01) + 1;
    mud2in.ny = (int)(1.5 * ly / mud2in.dy + 0.01) + 1;

    mud2in.nx = 3 * nx + 1;
    mud2in.ny = 3 * ny + 1;
    blitz::Array<double,2> bsig_data_mg1(nx+1,ny+1);
    blitz::Array<double,2> bsig_data_mg2(2*nx+1,2*ny+1);
    blitz::Array<double,2> bsig_data_fft(nx,ny);
    bsig_data_fft = 0.0;

    for(int i=0;i<nx;i++)
      for(int j=0;j<ny;j++){
          x = xa + i * dx;
          y = yc + j * dy;
          sinx = sin(x);
          cosx = cos(x);
          siny = sin(y);
          cosy = cos(y);
          u0(i,j) = cosx + siny;
          sigma = 1.5 + sinx * sinx * cosy * cosy;
          f(i,j) = -3. * (sigma - 1.) * u0(i,j);

          bsig_data_fft(i,j) = sigma;
      }

    fft2mg_copy_2d(bsig_data_fft,bsig_data_mg1);
    interp_bilinear_2d(bsig_data_mg1,bsig_data_mg2);
    mud2in.data = new MUD_FLOAT[mud2in.nx*mud2in.ny];
    mud2in.sinxat = new MUD_FLOAT[mud2in.nx];
    mud2in.cosyat = new MUD_FLOAT[mud2in.ny];
    for(int i=0;i<mud2in.nx;i++){
        x = xa - (lx / 4.0) + i * mud2in.dx;
        mud2in.sinxat[i] = sin(x);
    }
    for(int j=0;j<mud2in.ny;j++){
        y = yc - (ly / 4.0) + j * mud2in.dy;
        mud2in.cosyat[j] = cos(y);
    }

    int index;
    MUD_FLOAT x0 = xa - lx / 4.0;
    MUD_FLOAT y0 = yc - ly / 4.0;
    int i_xa = (int)((xa - x0) / mud2in.dx + 0.01);
    int j_yc = (int)((yc - y0) / mud2in.dy + 0.01);
    int period_x = 2 * nx;
    int period_y = 2 * ny;
    int ix,jy;
    for(int i=0;i<mud2in.nx;i++)
      for(int j=0;j<mud2in.ny;j++){
          x = xa - (lx / 4.0) + i * mud2in.dx;
          y = yc - (ly / 4.0) + j * mud2in.dy;
          sinx = sin(x);
          cosy = cos(y);
          index = mud2in.ny * i + j;
          mud2in.data[index] = 1.5 + sinx * sinx * cosy * cosy;
          ix = i - i_xa;
          if(ix < 0) ix += period_x;
          else if(ix > period_x) ix -= period_x;
          jy = j - j_yc;
          if(jy < 0) jy += period_y;
          else if(jy > period_y) jy -= period_y;
          mud2in.data[index] = bsig_data_mg2(ix,jy);
      }

    MUD2D *pmg=new MUD2D("sa",nx,ny,xb,yd,xa,yc);
    cout<<"Mudpack type "<<pmg->type()<<endl;
    //pmg->set_sig(sig,sig);
    pmg->set_sig(sig_global,sig_global);
    pmg->set_xlmbda(xlambda);
    pmg->set_boundary_function(bnd);
    pmg->set_fft2mg_interp(1); // use bicubic interpolation for fft2mg
    //pmg->set_boundary_type(0,0,0,0); // default 0,0,0,0
    //pmg->set_method(0); // default 0
    //pmg->set_mgopt(2,2,1,3); // default 1,2,1,1
    pmg->display();
    pmg->init();
    pmg->display();

    pmg->solve(u,f);

    u.set_name("u");
    u.save("u.mat");
    u0.set_name("u0");
    u0.save("u.mat");

    cout<<"u(0,0)(0,ny-1)(nx-1,0)(nx-1,ny-1)  ";
    cout<<u(0,0)<<","<<u(0,ny-1)<<",";
    cout<<u(nx-1,0)<<","<<u(nx-1,ny-1)<<endl;
    cout<<"u0(0,0)(0,ny-1)(nx-1,0)(nx-1,ny-1)  ";
    cout<<u0(0,0)<<","<<u0(0,ny-1)<<",";
    cout<<u0(nx-1,0)<<","<<u0(nx-1,ny-1)<<endl;
    cout<<"u(1,1)(1,np-2)(nt-2,1)(nt-2,np-2)  ";
    cout<<u(1,1)<<","<<u(1,ny-2)<<",";
    cout<<u(nx-2,1)<<","<<u(nx-2,ny-2)<<endl;
    cout<<"u0(1,1)(1,ny-2)(nx-2,1)(nx-2,ny-2)  ";
    cout<<u0(1,1)<<","<<u0(1,ny-2)<<",";
    cout<<u0(nx-2,1)<<","<<u0(nx-2,ny-2)<<endl;
    cout<<"u.mean = () "<<u.mean()<<endl;
    cout<<"u0.mean = () "<<u0.mean()<<endl;
    Grid uu1(u-u0);
    double err_norm1=uu1.abs_mean();
    Grid uu2(uu1*uu1);
    double err_norm2=sqrt(uu2.mean());
    cout<<"max error: "<<blitz::max(blitz::abs(uu1.data()))<<endl;
    cout<<"infinite norm of error: "<<err_norm1<<endl;
    cout<<"L2 norm of error: "<<err_norm2<<endl;
    cout<<endl;
}

/**
 * Test MUD2D class self-adjoint form
 * test case:
 *      div (e*grad(u)) = f 
 * which can be expanded as:
 *      d(e*du/dx)/dx + d(e*du/dx)/dy = f
 * with
 *      e = 3/2 + sin(x)^2 * cos(y)^2
 * The exact solution is
 *      u = cos(x) + sin(y)
 * Thus
 *      f = -3*(1/2 + sin(x)^2 * cos(y)^2)*(cos(x) + sin(y))
 *        = -3*(e - 1)*u
 *
 * To test this example, we must modify the nxa,nxb,nyc,nyd to 1 
 * in the constructor in MultigridMUD2sp.cpp. 
 * Because here the Homogeneous Dirichlet boundary condition
 * is considered other than periodic boundary condition.
 *
 */
void test_mud2d_sa(){
    Config cfg("param.ini");
    UnitCell uc(cfg);

    double lx = 2.0 * PI;
    double ly = 2.0 * PI;
    int nx = 128;
    int ny = 128;
    double dx = lx / nx;
    double dy = ly / ny;
    double xa = PI * 3.0 / 4.0;
    double xb = xa + lx;
    double yc = PI * 3.0 / 4.0;
    double yd = yc + ly;
    Grid u0(uc,nx,ny,1);
    Grid u(uc,nx,ny,1);
    Grid f(uc,nx,ny,1);
    double x;
    double y;
    double sinx,cosx,siny,cosy,sigma;
    for(int i=0;i<nx;i++)
      for(int j=0;j<ny;j++){
          x = xa + (i + 0.5) * dx;
          y = yc + (j + 0.5) * dy;
          sinx = sin(x);
          cosx = cos(x);
          siny = sin(y);
          cosy = cos(y);
          u0(i,j) = cosx + siny;
          sigma = 1.5 + sinx * sinx * cosy * cosy;
          f(i,j) = -3. * (sigma - 1.) * u0(i,j);
      }

    MUD2D *pmg=new MUD2D("sa",nx,ny,xb+.5*dx,yd+.5*dy,xa+.5*dx,yc+.5*dy);
    cout<<"Mudpack type "<<pmg->type()<<endl;
    pmg->set_sig(sig,sig);
    pmg->set_xlmbda(xlambda);
    pmg->set_boundary_function(bnd);
    //pmg->set_boundary_type(0,0,0,0); // default 0,0,0,0
    //pmg->set_method(0); // default 0
    //pmg->set_mgopt(2,2,1,3); // default 1,2,1,1
    pmg->display();
    pmg->init();
    pmg->display();

    pmg->solve(u,f);

    u.set_name("u");
    u.save("u.mat");
    u0.set_name("u0");
    u0.save("u.mat");

    cout<<"u(0,0)(0,ny-1)(nx-1,0)(nx-1,ny-1)  ";
    cout<<u(0,0)<<","<<u(0,ny-1)<<",";
    cout<<u(nx-1,0)<<","<<u(nx-1,ny-1)<<endl;
    cout<<"u0(0,0)(0,ny-1)(nx-1,0)(nx-1,ny-1)  ";
    cout<<u0(0,0)<<","<<u0(0,ny-1)<<",";
    cout<<u0(nx-1,0)<<","<<u0(nx-1,ny-1)<<endl;
    cout<<"u(1,1)(1,np-2)(nt-2,1)(nt-2,np-2)  ";
    cout<<u(1,1)<<","<<u(1,ny-2)<<",";
    cout<<u(nx-2,1)<<","<<u(nx-2,ny-2)<<endl;
    cout<<"u0(1,1)(1,ny-2)(nx-2,1)(nx-2,ny-2)  ";
    cout<<u0(1,1)<<","<<u0(1,ny-2)<<",";
    cout<<u0(nx-2,1)<<","<<u0(nx-2,ny-2)<<endl;
    cout<<"u.mean = () "<<u.mean()<<endl;
    cout<<"u0.mean = () "<<u0.mean()<<endl;
    Grid uu1(u-u0);
    double err_norm1=uu1.abs_mean();
    Grid uu2(uu1*uu1);
    double err_norm2=sqrt(uu2.mean());
    cout<<"max error: "<<blitz::max(blitz::abs(uu1.data()))<<endl;
    cout<<"infinite norm of error: "<<err_norm1<<endl;
    cout<<"L2 norm of error: "<<err_norm2<<endl;
    cout<<endl;

    delete pmg;
    pmg=new MUD2D("sa",nx,ny,xb,yd,xa,yc,1);
    cout<<"Mudpack type "<<pmg->type()<<endl;
    pmg->set_sig(sig,sig);
    pmg->set_xlmbda(xlambda);
    pmg->set_boundary_function(bnd);
    pmg->set_fft2mg_interp(1); // use bicubic interpolation for fft2mg
    //pmg->set_boundary_type(0,0,0,0); // default 0,0,0,0
    //pmg->set_method(0); // default 0
    //pmg->set_mgopt(2,2,1,3); // default 1,2,1,1
    pmg->display();
    pmg->init();
    pmg->display();

    pmg->solve(u,f);

    u.set_name("u");
    u.save("u.mat");
    u0.set_name("u0");
    u0.save("u.mat");

    cout<<"u(0,0)(0,ny-1)(nx-1,0)(nx-1,ny-1)  ";
    cout<<u(0,0)<<","<<u(0,ny-1)<<",";
    cout<<u(nx-1,0)<<","<<u(nx-1,ny-1)<<endl;
    cout<<"u0(0,0)(0,ny-1)(nx-1,0)(nx-1,ny-1)  ";
    cout<<u0(0,0)<<","<<u0(0,ny-1)<<",";
    cout<<u0(nx-1,0)<<","<<u0(nx-1,ny-1)<<endl;
    cout<<"u(1,1)(1,np-2)(nt-2,1)(nt-2,np-2)  ";
    cout<<u(1,1)<<","<<u(1,ny-2)<<",";
    cout<<u(nx-2,1)<<","<<u(nx-2,ny-2)<<endl;
    cout<<"u0(1,1)(1,ny-2)(nx-2,1)(nx-2,ny-2)  ";
    cout<<u0(1,1)<<","<<u0(1,ny-2)<<",";
    cout<<u0(nx-2,1)<<","<<u0(nx-2,ny-2)<<endl;
    cout<<"u.mean = () "<<u.mean()<<endl;
    cout<<"u0.mean = () "<<u0.mean()<<endl;
    uu1= u - u0;
    err_norm1=uu1.abs_mean();
    uu2 = uu1 * uu1;
    err_norm2=sqrt(uu2.mean());
    cout<<"max error: "<<blitz::max(blitz::abs(uu1.data()))<<endl;
    cout<<"infinite norm of error: "<<err_norm1<<endl;
    cout<<"L2 norm of error: "<<err_norm2<<endl;
    cout<<endl;
}

void cofx(MUD_FLOAT *x,MUD_FLOAT *cxx,MUD_FLOAT *cx,MUD_FLOAT *cex){
    *cxx = 1.0;
    *cx = 0.0;
    *cex = 0.0;
}

void cofy(MUD_FLOAT *y,MUD_FLOAT *cyy,MUD_FLOAT *cy,MUD_FLOAT *cey){
    *cyy = 1.0;
    *cy = 0.0;
    *cey = 0.0;
}

void cofz(MUD_FLOAT *z,MUD_FLOAT *czz,MUD_FLOAT *cz,MUD_FLOAT *cez){
    *czz = 1.0;
    *cz = 0.0;
    *cez = 0.0;
}

/**
 * Test MUD2D class
 *
 * Test case 1:
 * Example from Briggs "A Multigrid tutorial" p.125~128
 *
 *      -u_xx - epsilon*u_yy = f
 * with
 *      f = 2 * (y - y^2) + 2 * epsilon * (x - x^2)
 * The exact solution is
 *      u = (x - x^2) * (y - y^2)
 * To test this example, we must modify the nxa,nxb,nyc,nyd to 1 
 * in the constructor in MultigridMUD2sp.cpp. 
 * Because here the Homogeneous Dirichlet boundary condition
 * is considered other than periodic boundary condition.
 *
 * Test case 2:
 *      u_xx + u_yy = f
 * with
 *      f = -(cos(x) + sin(y))
 * The exact solution is
 *      u = cos(x) + sin(y)
 *
 *
 */
void test_mud2d_sp(){
    Config cfg("param.ini");
    UnitCell uc(cfg);

    //double epsilon=1.0;
    double lx =2.0 * PI;
    double ly = lx; //sqrt(lx/epsilon);
    int Lx = 128;
    int Ly = 128;
    Grid u0(uc,Lx,Ly,1);
    Grid u(uc,Lx,Ly,1);
    Grid f(uc,Lx,Ly,1);
    double dx = lx / Lx;
    double dy = ly / Ly;
    double x;
    double y;
    for(int i=0;i<Lx;i++)
      for(int j=0;j<Ly;j++){
          x = (i + 0.5) * dx;
          y = (j + 0.5) * dy;
          f(i,j) = -(cos(x) + sin(y));
          u0(i,j) = cos(x) + sin(y);
          //f(i,j) = -2*(y-y*y)-2*epsilon*(x-x*x);
          //u0(i,j) = (x-x*x)*(y-y*y);
      }

    MUD2D *pmg=new MUD2D("sp",Lx,Ly,lx,ly);
    cout<<"Mudpack type "<<pmg->type()<<endl;
    pmg->set_cof(cofx,cofy);
    pmg->set_boundary_function(bnd);
    //pmg->set_mgopt(2,2,1,3);
    pmg->display();
    pmg->init();
    pmg->display();

    pmg->solve(u,f);

    u.set_name("u");
    u.save("u.mat");
    u0.set_name("u0");
    u0.save("u.mat");

    int nx = Lx;
    int ny = Ly;
    cout<<"u(0,0)(0,ny-1)(nx-1,0)(nx-1,ny-1)  ";
    cout<<u(0,0)<<","<<u(0,ny-1)<<",";
    cout<<u(nx-1,0)<<","<<u(nx-1,ny-1)<<endl;
    cout<<"u0(0,0)(0,ny-1)(nx-1,0)(nx-1,ny-1)  ";
    cout<<u0(0,0)<<","<<u0(0,ny-1)<<",";
    cout<<u0(nx-1,0)<<","<<u0(nx-1,ny-1)<<endl;
    cout<<"u(1,1)(1,np-2)(nt-2,1)(nt-2,np-2)  ";
    cout<<u(1,1)<<","<<u(1,ny-2)<<",";
    cout<<u(nx-2,1)<<","<<u(nx-2,ny-2)<<endl;
    cout<<"u0(1,1)(1,ny-2)(nx-2,1)(nx-2,ny-2)  ";
    cout<<u0(1,1)<<","<<u0(1,ny-2)<<",";
    cout<<u0(nx-2,1)<<","<<u0(nx-2,ny-2)<<endl;
    cout<<"u.mean = () "<<u.mean()<<endl;
    cout<<"u0.mean = () "<<u0.mean()<<endl;
    Grid uu1(u-u0);
    double err_norm1=uu1.abs_mean();
    Grid uu2(uu1*uu1);
    double err_norm2=sqrt(uu2.mean());
    cout<<"max error: "<<blitz::max(blitz::abs(uu1.data()))<<endl;
    cout<<"infinite norm of error: "<<err_norm1<<endl;
    cout<<"L2 norm of error: "<<err_norm2<<endl;
    cout<<endl;

    delete pmg;
    pmg=new MUD2D("sp",Lx,Ly,lx,ly,.0,.0,1);
    cout<<"Mudpack type "<<pmg->type()<<endl;
    pmg->set_cof(cofx,cofy);
    pmg->set_boundary_function(bnd);
    //pmg->set_mgopt(2,2,1,3);
    pmg->display();
    pmg->init();
    pmg->display();

    pmg->solve(u,f);

    u.set_name("u");
    u.save("u.mat");
    u0.set_name("u0");
    u0.save("u.mat");

    nx = Lx;
    ny = Ly;
    cout<<"u(0,0)(0,ny-1)(nx-1,0)(nx-1,ny-1)  ";
    cout<<u(0,0)<<","<<u(0,ny-1)<<",";
    cout<<u(nx-1,0)<<","<<u(nx-1,ny-1)<<endl;
    cout<<"u0(0,0)(0,ny-1)(nx-1,0)(nx-1,ny-1)  ";
    cout<<u0(0,0)<<","<<u0(0,ny-1)<<",";
    cout<<u0(nx-1,0)<<","<<u0(nx-1,ny-1)<<endl;
    cout<<"u(1,1)(1,np-2)(nt-2,1)(nt-2,np-2)  ";
    cout<<u(1,1)<<","<<u(1,ny-2)<<",";
    cout<<u(nx-2,1)<<","<<u(nx-2,ny-2)<<endl;
    cout<<"u0(1,1)(1,ny-2)(nx-2,1)(nx-2,ny-2)  ";
    cout<<u0(1,1)<<","<<u0(1,ny-2)<<",";
    cout<<u0(nx-2,1)<<","<<u0(nx-2,ny-2)<<endl;
    cout<<"u.mean = () "<<u.mean()<<endl;
    cout<<"u0.mean = () "<<u0.mean()<<endl;
    uu1 = u - u0;
    err_norm1=uu1.abs_mean();
    uu2 = uu1 * uu1;
    err_norm2=sqrt(uu2.mean());
    cout<<"max error: "<<blitz::max(blitz::abs(uu1.data()))<<endl;
    cout<<"infinite norm of error: "<<err_norm1<<endl;
    cout<<"L2 norm of error: "<<err_norm2<<endl;
    cout<<endl;
}

void bnd3(int *kbdy,MUD_FLOAT *xory,MUD_FLOAT *yorz,MUD_FLOAT *alfa,MUD_FLOAT *gbdy){
}

/**
 * Test MUD3D.mud3sp class
 *
 * Test case 1:
 *      u_xx + u_yy + u_zz = f
 * with
 *      f = -(cos(x) + sin(y) + cos(z))
 * The exact solution is
 *      u = cos(x) + sin(y) + cos(z)
 *
 */
void test_mud3d_sp(){
    Config cfg("param.ini");
    UnitCell uc(cfg);

    //double epsilon=1.0;
    double lx = 2.0 * PI;
    double ly = 2.0 * PI; 
    double lz = 2.0 * PI; 
    int Lx = 64;
    int Ly = 64;
    int Lz = 64;
    Grid u0(uc,Lx,Ly,Lz);
    Grid u(uc,Lx,Ly,Lz);
    Grid f(uc,Lx,Ly,Lz);
    double dx = lx / Lx;
    double dy = ly / Ly;
    double dz = lz / Lz;
    double x,y,z;
    for(int i=0;i<Lx;i++)
      for(int j=0;j<Ly;j++)
        for(int k=0;k<Lz;k++){
          x = (i + 0.5) * dx;
          y = (j + 0.5) * dy;
          z = (k + 0.5) * dz;
          f(i,j,k) = -(cos(x) + sin(y) + cos(z));
          u0(i,j,k) = cos(x) + sin(y) + cos(z);
      }

    MUD3D *pmg=new MUD3D("sp",Lx,Ly,Lz,lx,ly,lz);
    cout<<"Mudpack type "<<pmg->type()<<endl;
    pmg->set_cof(cofx,cofy,cofz);
    pmg->set_boundary_function(bnd3);
    //pmg->set_mgopt(2,2,1,3);
    pmg->display();
    pmg->init();
    pmg->display();

    pmg->solve(u,f);

    u.set_name("u");
    u.save("u.mat");
    u0.set_name("u0");
    u0.save("u.mat");

    int nx = Lx;
    int ny = Ly;
    cout<<"u(0,0,0)(0,ny-1,0)(nx-1,0,0)(nx-1,ny-1,0)  ";
    cout<<u(0,0,0)<<","<<u(0,ny-1,0)<<",";
    cout<<u(nx-1,0,0)<<","<<u(nx-1,ny-1,0)<<endl;
    cout<<"u0(0,0,0)(0,ny-1,0)(nx-1,0,0)(nx-1,ny-1,0)  ";
    cout<<u0(0,0,0)<<","<<u0(0,ny-1,0)<<",";
    cout<<u0(nx-1,0,0)<<","<<u0(nx-1,ny-1,0)<<endl;
    cout<<"u(1,1,0)(1,np-2,0)(nt-2,1,0)(nt-2,np-2,0)  ";
    cout<<u(1,1,0)<<","<<u(1,ny-2,0)<<",";
    cout<<u(nx-2,1,0)<<","<<u(nx-2,ny-2,0)<<endl;
    cout<<"u0(1,1,0)(1,ny-2,0)(nx-2,1,0)(nx-2,ny-2,0)  ";
    cout<<u0(1,1,0)<<","<<u0(1,ny-2,0)<<",";
    cout<<u0(nx-2,1,0)<<","<<u0(nx-2,ny-2,0)<<endl;
    cout<<"u.mean = () "<<u.mean()<<endl;
    cout<<"u0.mean = () "<<u0.mean()<<endl;
    Grid uu1(u-u0);
    double err_norm1=uu1.abs_mean();
    Grid uu2(uu1*uu1);
    double err_norm2=sqrt(uu2.mean());
    cout<<"max error: "<<blitz::max(blitz::abs(uu1.data()))<<endl;
    cout<<"infinite norm of error: "<<err_norm1<<endl;
    cout<<"L2 norm of error: "<<err_norm2<<endl;
    cout<<endl;

    delete pmg;
    pmg=new MUD3D("sp",Lx,Ly,Lz,lx,ly,lz,.0,.0,.0,1);
    cout<<"Mudpack type "<<pmg->type()<<endl;
    pmg->set_cof(cofx,cofy,cofz);
    pmg->set_boundary_function(bnd3);
    //pmg->set_mgopt(2,2,1,3);
    pmg->display();
    pmg->init();
    pmg->display();

    pmg->solve(u,f);

    u.set_name("u");
    u.save("u.mat");
    u0.set_name("u0");
    u0.save("u.mat");

    nx = Lx;
    ny = Ly;
    cout<<"u(0,0,0)(0,ny-1,0)(nx-1,0,0)(nx-1,ny-1,0)  ";
    cout<<u(0,0,0)<<","<<u(0,ny-1,0)<<",";
    cout<<u(nx-1,0,0)<<","<<u(nx-1,ny-1,0)<<endl;
    cout<<"u0(0,0,0)(0,ny-1,0)(nx-1,0,0)(nx-1,ny-1,0)  ";
    cout<<u0(0,0,0)<<","<<u0(0,ny-1,0)<<",";
    cout<<u0(nx-1,0,0)<<","<<u0(nx-1,ny-1,0)<<endl;
    cout<<"u(1,1,0)(1,np-2,0)(nt-2,1,0)(nt-2,np-2,0)  ";
    cout<<u(1,1,0)<<","<<u(1,ny-2,0)<<",";
    cout<<u(nx-2,1,0)<<","<<u(nx-2,ny-2,0)<<endl;
    cout<<"u0(1,1,0)(1,ny-2,0)(nx-2,1,0)(nx-2,ny-2,0)  ";
    cout<<u0(1,1,0)<<","<<u0(1,ny-2,0)<<",";
    cout<<u0(nx-2,1,0)<<","<<u0(nx-2,ny-2,0)<<endl;
    cout<<"u.mean = () "<<u.mean()<<endl;
    cout<<"u0.mean = () "<<u0.mean()<<endl;
    uu1 = u - u0;
    err_norm1=uu1.abs_mean();
    uu2 = uu1 * uu1;
    err_norm2=sqrt(uu2.mean());
    cout<<"max error: "<<blitz::max(blitz::abs(uu1.data()))<<endl;
    cout<<"infinite norm of error: "<<err_norm1<<endl;
    cout<<"L2 norm of error: "<<err_norm2<<endl;
    cout<<endl;
}

double sigr(MUD_FLOAT *r,MUD_FLOAT *t,MUD_FLOAT *p){
    return (*r) * (*r) * sin(*t);
}

double sigt(MUD_FLOAT *r,MUD_FLOAT *t,MUD_FLOAT *p){
    return sin(*t);
}

double sigp(MUD_FLOAT *r,MUD_FLOAT *t,MUD_FLOAT *p){
    return 1.0 / sin(*t);
}

double xlm3(MUD_FLOAT *r,MUD_FLOAT *t,MUD_FLOAT *p){
    return (*r) * (*r) * sin(*t);
}

void exact3(double r,double t,double p,double &urr,double &utt,double &upp,double &ur,double &ut,double &up,double &ue){
    double st = sin(t);
    double ct = cos(t);
    double sp = sin(p);
    double cp = cos(p);

    double r6 = r*r*r*r*r*r;
    double ep = (cp*sp) * (cp*sp);
    double dep = 2.*(cp*sp)*(cp*cp-sp*sp);
    double ddep = 2.*((cp*cp-sp*sp)*(cp*cp-sp*sp)-4.*(sp*cp)*(sp*cp));
    double et = st*st*(st*ct)*(st*ct);
    double det = 2.*(2.*(st*ct)*(st*ct)*(st*ct)-ct*st*st*st*st*st);
    double ddet = 12.*(ct*st)*(ct*st)*(ct*ct-st*st) + \
                  2.*st*st*st*st*(st*st-4.*ct*ct);
    ue = r6*et*ep;
    ur = 6.*r*r*r*r*r*et*ep;
    urr = 30.*r*r*r*r*et*ep;
    ut = r6*det*ep;
    utt = r6*ddet*ep;
    up = r6*et*dep;
    upp = r6*et*ddep;
}

extern "C"{
double sigx_global(MUD_FLOAT *x,MUD_FLOAT *y,MUD_FLOAT *z){
    MUD_FLOAT dx = gmudsig.dx;
    MUD_FLOAT dy = gmudsig.dy;
    MUD_FLOAT dz = gmudsig.dz;
    MUD_FLOAT xa = gmudsig.xa;
    MUD_FLOAT yc = gmudsig.yc;
    MUD_FLOAT ze = gmudsig.ze;
    int ny = gmudsig.ny;
    int nz = gmudsig.nz;
    int i = (int)((*x - xa) / dx + 0.01);
    int j = (int)((*y - yc) / dy + 0.01);
    int k = (int)((*z - ze) / dz + 0.01);
    int index = nz * (ny * i + j) + k;
    return gmudsig.xdata[index];
}

double sigy_global(MUD_FLOAT *x,MUD_FLOAT *y,MUD_FLOAT *z){
    MUD_FLOAT dx = gmudsig.dx;
    MUD_FLOAT dy = gmudsig.dy;
    MUD_FLOAT dz = gmudsig.dz;
    MUD_FLOAT xa = gmudsig.xa;
    MUD_FLOAT yc = gmudsig.yc;
    MUD_FLOAT ze = gmudsig.ze;
    int ny = gmudsig.ny;
    int nz = gmudsig.nz;
    int i = (int)((*x - xa) / dx + 0.01);
    int j = (int)((*y - yc) / dy + 0.01);
    int k = (int)((*z - ze) / dz + 0.01);
    int index = nz * (ny * i + j) + k;
    return gmudsig.ydata[index];
}

double sigz_global(MUD_FLOAT *x,MUD_FLOAT *y,MUD_FLOAT *z){
    MUD_FLOAT dx = gmudsig.dx;
    MUD_FLOAT dy = gmudsig.dy;
    MUD_FLOAT dz = gmudsig.dz;
    MUD_FLOAT xa = gmudsig.xa;
    MUD_FLOAT yc = gmudsig.yc;
    MUD_FLOAT ze = gmudsig.ze;
    int ny = gmudsig.ny;
    int nz = gmudsig.nz;
    int i = (int)((*x - xa) / dx + 0.01);
    int j = (int)((*y - yc) / dy + 0.01);
    int k = (int)((*z - ze) / dz + 0.01);
    int index = nz * (ny * i + j) + k;
    return gmudsig.zdata[index];
}

double xlm3_global(MUD_FLOAT *x,MUD_FLOAT *y,MUD_FLOAT *z){
    MUD_FLOAT dx = gmudsig.dx;
    MUD_FLOAT dy = gmudsig.dy;
    MUD_FLOAT dz = gmudsig.dz;
    MUD_FLOAT xa = gmudsig.xa;
    MUD_FLOAT yc = gmudsig.yc;
    MUD_FLOAT ze = gmudsig.ze;
    int ny = gmudsig.ny;
    int nz = gmudsig.nz;
    int i = (int)((*x - xa) / dx + 0.01);
    int j = (int)((*y - yc) / dy + 0.01);
    int k = (int)((*z - ze) / dz + 0.01);
    int index = nz * (ny * i + j) + k;
    return gmudsig.lambda_data[index];
}

} // end of extern "C"

void test_mud3d_sa_global(){
    Config cfg("param.ini");
    UnitCell uc(cfg);

    double ra = 0.50;
    double rb = 1.00;
    double tc = PI / 4.0;
    double td = PI * 3.0 / 4.0;
    double pe = 0.0;
    double pf = 2.0 * PI;
    double lr = rb - ra;
    double lt = td - tc;
    double lp = pf - pe;
    int nr = 24;
    int nt = 16;
    int np = 32;

    Grid u0(uc,nr,nt,np);
    Grid u(uc,nr,nt,np);
    Grid f(uc,nr,nt,np);
    Grid epsr(uc,nr,nt,np);
    Grid epst(uc,nr,nt,np);
    Grid epsp(uc,nr,nt,np);
    Grid epsl(uc,nr,nt,np);
    double dlr = lr / (nr-1);
    double dlt = lt / (nt-1);
    double dlp = lp / (np-1);
    double r,t,p;
    double sint,cost,sinp,cosp;
    double ue,ur,ut,up,urr,utt,upp;
    for(int j=0;j<nt;j++)
      for(int k=0;k<np;k++)
        for(int i=0;i<nr;i++){
          r = ra + i * dlr;
          t = tc + j * dlt;
          p = pe + k * dlp;
          sint = sin(t);
          cost = cos(t);
          sinp = sin(p);
          cosp = cos(p);
          exact3(r,t,p,urr,utt,upp,ur,ut,up,ue);
          f(i,j,k) = r*sint*(r*urr+2.*ur) + \
                     sint*utt + cost*ut + \
                     upp/sint - r*r*sint*ue;
          epsr(i,j,k) = r * r * sint;
          epst(i,j,k) = sint;
          epsp(i,j,k) = 1. / sint;
          epsl(i,j,k) = r * r * sint;
          u0(i,j,k) = ue;
      }

    MUD3D *pmg=new MUD3D("sa",nr,nt,np,rb,td,pf,ra,tc,pe);
    pmg->set_sig(sigx_global,sigy_global,sigz_global);
    pmg->set_xlmbda(xlm3_global);
    //pmg->set_boundary_function(bnd3);
    pmg->set_boundary_type(1,1,1,1,0,0);
    pmg->set_fft2mg_interp(1); // use bicubic interpolation for fft2mg
    pmg->set_method(1);
    pmg->set_mgopt(2,2,1,3);
    pmg->display();
    pmg->init();
    pmg->display();

//    blitz::Array<double,3> sigx_data_mg1(nr+1,nt+1,np+1);
//    blitz::Array<double,3> sigx_data_mg2(2*nr+1,2*nt+1,2*np+1);
    blitz::Array<double,3> sigy_data_mg1(nr+1,nt+1,np+1);
    blitz::Array<double,3> sigy_data_mg2(2*nr+1,2*nt+1,2*np+1);
    blitz::Array<double,3> sigz_data_mg1(nr+1,nt+1,np+1);
    blitz::Array<double,3> sigz_data_mg2(2*nr+1,2*nt+1,2*np+1);
    blitz::Array<double,3> xlm_data_mg1(nr+1,nt+1,np+1);
    blitz::Array<double,3> xlm_data_mg2(2*nr+1,2*nt+1,2*np+1);
    int index;
    int i_ra = (int)((ra - gmudsig.xa) / gmudsig.dx + 0.01);
    int j_tc = (int)((tc - gmudsig.yc) / gmudsig.dy + 0.01);
    int k_pe = (int)((pe - gmudsig.ze) / gmudsig.dz + 0.01);
    int period_r = 2 * nr;
    int period_t = 2 * nt;
    int period_p = 2 * np;
    int ir,jt,kp;

//    fft2mg_copy_3d(epsr,sigx_data_mg1);
//    interp_trilinear_3d(sigx_data_mg1,sigx_data_mg2);

    fft2mg_copy_3d(epst,sigy_data_mg1);
    interp_trilinear_3d(sigy_data_mg1,sigy_data_mg2);

    fft2mg_copy_3d(epsp,sigz_data_mg1);
    interp_trilinear_3d(sigz_data_mg1,sigz_data_mg2);

    fft2mg_copy_3d(epsl,xlm_data_mg1);
    interp_trilinear_3d(xlm_data_mg1,xlm_data_mg2);

    for(int i=0;i<gmudsig.nx;i++)
      for(int j=0;j<gmudsig.ny;j++)
        for(int k=0;k<gmudsig.nz;k++){
            ir = i - i_ra;
            if(ir < 0) ir += period_r;
            else if(ir > period_r) ir -= period_r;

            jt = j - j_tc;
            if(jt < 0) jt += period_t;
            else if(jt > period_t) jt -= period_t;

            kp = k - k_pe;
            if(kp < 0) kp += period_p;
            else if(kp > period_p) kp -= period_p;

            index = gmudsig.nz * (gmudsig.ny * i + j) + k;
 //           gmudsig.xdata[index] = sigx_data_mg2(ir,jt,kp);
            gmudsig.ydata[index] = sigy_data_mg2(ir,jt,kp);
            gmudsig.zdata[index] = sigz_data_mg2(ir,jt,kp);
            gmudsig.lambda_data[index] = xlm_data_mg2(ir,jt,kp);
        }
    pmg->set_data(epsr.data());
    pmg->solve(u,f);

    u.set_name("u");
    u.save("u.mat");
    u0.set_name("u0");
    u0.save("u.mat");

    cout<<"u(0,0,0)(0,ny-1,0)(nx-1,0,0)(nx-1,ny-1,0)  ";
    cout<<u(0,0,0)<<","<<u(0,np-1,0)<<",";
    cout<<u(nr-1,0,0)<<","<<u(nr-1,np-1,0)<<endl;
    cout<<"u0(0,0,0)(0,ny-1,0)(nx-1,0,0)(nx-1,ny-1,0)  ";
    cout<<u0(0,0,0)<<","<<u0(0,np-1,0)<<",";
    cout<<u0(nr-1,0,0)<<","<<u0(nr-1,np-1,0)<<endl;
    cout<<"u(1,1,0)(1,np-2,0)(nt-2,1,0)(nt-2,np-2,0)  ";
    cout<<u(1,1,0)<<","<<u(1,np-2,0)<<",";
    cout<<u(nr-2,1,0)<<","<<u(nr-2,np-2,0)<<endl;
    cout<<"u0(1,1,0)(1,ny-2,0)(nx-2,1,0)(nx-2,ny-2,0)  ";
    cout<<u0(1,1,0)<<","<<u0(1,np-2,0)<<",";
    cout<<u0(nr-2,1,0)<<","<<u0(nr-2,np-2,0)<<endl;
    cout<<"u.mean = () "<<u.mean()<<endl;
    cout<<"u0.mean = () "<<u0.mean()<<endl;
    Grid uu1(u-u0);
    double err_norm1=uu1.abs_mean();
    Grid uu2(uu1*uu1);
    double err_norm2=sqrt(uu2.mean());
    cout<<"max error: "<<blitz::max(blitz::abs(uu1.data()))<<endl;
    cout<<"infinite norm of error: "<<err_norm1<<endl;
    cout<<"L2 norm of error: "<<err_norm2<<endl;
    cout<<endl;
}

/**
 * Test mud3sa_
 * Test case is from tmud2sa.f in MUDPACK 5.0.1
 * Test passed. 2012.4.16.
 */
void test_mud3d_sa(){
    Config cfg("param.ini");
    UnitCell uc(cfg);

    double ra = 0.50;
    double rb = 1.00;
    double tc = PI / 4.0;
    double td = PI * 3.0 / 4.0;
    double pe = 0.0;
    double pf = 2.0 * PI;
    double lr = rb - ra;
    double lt = td - tc;
    double lp = pf - pe;
    int nr = 24;
    int nt = 16;
    int np = 32;

    Grid u0(uc,nr,nt,np);
    Grid u(uc,nr,nt,np);
    Grid f(uc,nr,nt,np);
    double dlr = lr / (nr-1);
    double dlt = lt / (nt-1);
    double dlp = lp / (np-1);
    double r,t,p;
    double sint,cost,sinp,cosp;
    double ue,ur,ut,up,urr,utt,upp;
    for(int j=0;j<nt;j++)
      for(int k=0;k<np;k++)
        for(int i=0;i<nr;i++){
          r = ra + (i+0.5) * dlr;
          t = tc + (j+0.5) * dlt;
          p = pe + (k+0.5) * dlp;
          sint = sin(t);
          cost = cos(t);
          sinp = sin(p);
          cosp = cos(p);
          exact3(r,t,p,urr,utt,upp,ur,ut,up,ue);
          f(i,j,k) = r*sint*(r*urr+2.*ur) + \
                     sint*utt + cost*ut + \
                     upp/sint - r*r*sint*ue;
          u0(i,j,k) = ue;
      }

    MUD3D *pmg=new MUD3D("sa",nr,nt,np,rb,td,pf,ra,tc,pe);
    pmg->set_sig(sigr,sigt,sigp);
    pmg->set_xlmbda(xlm3);
    pmg->set_boundary_function(bnd3);
    pmg->set_boundary_type(1,1,1,1,0,0);
    pmg->set_fft2mg_interp(1); // use bicubic interpolation for fft2mg
    pmg->set_method(1);
    pmg->set_mgopt(2,2,1,3);
    pmg->display();
    pmg->init();
    pmg->display();

    pmg->solve(u,f);

    u.set_name("u");
    u.save("u.mat");
    u0.set_name("u0");
    u0.save("u.mat");

    cout<<"u(0,0,0)(0,ny-1,0)(nx-1,0,0)(nx-1,ny-1,0)  ";
    cout<<u(0,0,0)<<","<<u(0,np-1,0)<<",";
    cout<<u(nr-1,0,0)<<","<<u(nr-1,np-1,0)<<endl;
    cout<<"u0(0,0,0)(0,ny-1,0)(nx-1,0,0)(nx-1,ny-1,0)  ";
    cout<<u0(0,0,0)<<","<<u0(0,np-1,0)<<",";
    cout<<u0(nr-1,0,0)<<","<<u0(nr-1,np-1,0)<<endl;
    cout<<"u(1,1,0)(1,np-2,0)(nt-2,1,0)(nt-2,np-2,0)  ";
    cout<<u(1,1,0)<<","<<u(1,np-2,0)<<",";
    cout<<u(nr-2,1,0)<<","<<u(nr-2,np-2,0)<<endl;
    cout<<"u0(1,1,0)(1,ny-2,0)(nx-2,1,0)(nx-2,ny-2,0)  ";
    cout<<u0(1,1,0)<<","<<u0(1,np-2,0)<<",";
    cout<<u0(nr-2,1,0)<<","<<u0(nr-2,np-2,0)<<endl;
    cout<<"u.mean = () "<<u.mean()<<endl;
    cout<<"u0.mean = () "<<u0.mean()<<endl;
    Grid uu1(u-u0);
    double err_norm1=uu1.abs_mean();
    Grid uu2(uu1*uu1);
    double err_norm2=sqrt(uu2.mean());
    cout<<"max error: "<<blitz::max(blitz::abs(uu1.data()))<<endl;
    cout<<"infinite norm of error: "<<err_norm1<<endl;
    cout<<"L2 norm of error: "<<err_norm2<<endl;
    cout<<endl;

    delete pmg;
    pmg=new MUD3D("sa",nr,nt,np,rb,td,pf,ra,tc,pe,1);
    pmg->set_sig(sigr,sigt,sigp);
    pmg->set_xlmbda(xlm3);
    pmg->set_boundary_function(bnd3);
    pmg->set_boundary_type(1,1,1,1,0,0);
    pmg->set_fft2mg_interp(1); // use bicubic interpolation for fft2mg
    pmg->set_method(1);
    pmg->set_mgopt(2,2,1,3);
    pmg->display();
    pmg->init();
    pmg->display();

    pmg->solve(u,f);

    u.set_name("u");
    u.save("u.mat");
    u0.set_name("u0");
    u0.save("u.mat");

    cout<<"u(0,0,0)(0,ny-1,0)(nx-1,0,0)(nx-1,ny-1,0)  ";
    cout<<u(0,0,0)<<","<<u(0,np-1,0)<<",";
    cout<<u(nr-1,0,0)<<","<<u(nr-1,np-1,0)<<endl;
    cout<<"u0(0,0,0)(0,ny-1,0)(nx-1,0,0)(nx-1,ny-1,0)  ";
    cout<<u0(0,0,0)<<","<<u0(0,np-1,0)<<",";
    cout<<u0(nr-1,0,0)<<","<<u0(nr-1,np-1,0)<<endl;
    cout<<"u(1,1,0)(1,np-2,0)(nt-2,1,0)(nt-2,np-2,0)  ";
    cout<<u(1,1,0)<<","<<u(1,np-2,0)<<",";
    cout<<u(nr-2,1,0)<<","<<u(nr-2,np-2,0)<<endl;
    cout<<"u0(1,1,0)(1,ny-2,0)(nx-2,1,0)(nx-2,ny-2,0)  ";
    cout<<u0(1,1,0)<<","<<u0(1,np-2,0)<<",";
    cout<<u0(nr-2,1,0)<<","<<u0(nr-2,np-2,0)<<endl;
    cout<<"u.mean = () "<<u.mean()<<endl;
    cout<<"u0.mean = () "<<u0.mean()<<endl;
    uu1 = u - u0;
    err_norm1=uu1.abs_mean();
    uu2 = uu1 * uu1;
    err_norm2=sqrt(uu2.mean());
    cout<<"max error: "<<blitz::max(blitz::abs(uu1.data()))<<endl;
    cout<<"infinite norm of error: "<<err_norm1<<endl;
    cout<<"L2 norm of error: "<<err_norm2<<endl;
    cout<<endl;
}

int main(){
//    test_fft2mg_copy();
//    test_fft2mg_interp();
//    test_mud2d_sp();
//    test_mud2d_sa0();
//    test_mud2d_sa();
//    test_mud2d_sa_global();
//    test_mud3d_sp();
//    test_mud3d_sa();
//    test_mud3d_sa_global();
    test_mud2d_cr();

    return 0;
}
