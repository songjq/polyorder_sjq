/**
 * Test FieldE
 *
 * Copyright@ Yi-Xin Liu, 2012
 *
 */

#include "Grid.h"
#include "FieldE.h"

#include <iostream>
#include <string>
#include <ctime>
#include <cmath>

#include "UnitCell.h"
#include "Field.h"
#include "Yita.h"
#include "multigrid.h"
#include "MUD.h"
#include "MUD2D.h"
#include "MUD3D.h"

using namespace std;

void test_fieldE_mud2sa(){
    Config cfg("param.ini");

    double lx=2.0 * PI;
    double ly=2.0 * PI;
    int nx=128;
    int ny=128;
    int nz=1;
    double dx = lx / nx;
    double dy = ly / ny;
    double xa = 0.0;
    //double xb = xa + lx;
    double yc = 0.0;
    //double yd = yc + ly;
    int eps_type = 1;
    cfg.set_integer("Grid","dimension",2);
    cfg.set_integer("Grid","Lx",nx);
    cfg.set_integer("Grid","Ly",ny);
    cfg.set_integer("Grid","Lz",nz);
    cfg.set_string("UnitCell","CrystalSystemType","Square");
    cfg.set_double("UnitCell","a",lx);
    cfg.set_integer("Algorithm","dielectric_constant",eps_type);
    UnitCell uc(cfg);

    FieldE psi("psi",eps_type,cfg,1.0,1.0); // default Updater for 2D

    cout<<"psi.name = (psi) "<<psi.name()<<endl;
    cout<<"psi.dim = (2) "<<psi.dim()<<endl;
    cout<<"psi.(Lx,Ly) = (128,128) "<<"("<<psi.Lx()<<","<<psi.Ly()<<")"<<endl;
    cout<<"psi.(dx,dy) = (0.0490874,0.0490874)"<<"("<<psi.dx()<<","<<psi.dy()<<")"<<endl;
    cout<<endl;

    Grid u0(uc,nx,ny,nz);
    Grid rho(uc,nx,ny,nz);
    Grid eps(uc,nx,ny,nz);
    dx = psi.dx();
    dy = psi.dy();
    double x,y;
    double sinx,cosx,siny,cosy,sigma;
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
          rho(i,j) = -3. * (sigma - 1.) * u0(i,j);

          eps(i,j) = sigma;
      }

    psi.set_eps(eps);
    psi.update(rho);
    //psi.update(rho,up);
    psi.set_eps(eps);
    psi.update(rho);
    psi.save("psi.mat");
    u0.set_name("u0");
    u0.save("psi.mat");

    cout<<"u(0,0)(0,ny-1)(nx-1,0)(nx-1,ny-1)  ";
    cout<<psi(0,0)<<","<<psi(0,ny-1)<<",";
    cout<<psi(nx-1,0)<<","<<psi(nx-1,ny-1)<<endl;
    cout<<"u0(0,0)(0,ny-1)(nx-1,0)(nx-1,ny-1)  ";
    cout<<u0(0,0)<<","<<u0(0,ny-1)<<",";
    cout<<u0(nx-1,0)<<","<<u0(nx-1,ny-1)<<endl;
    cout<<"u(1,1)(1,np-2)(nt-2,1)(nt-2,np-2)  ";
    cout<<psi(1,1)<<","<<psi(1,ny-2)<<",";
    cout<<psi(nx-2,1)<<","<<psi(nx-2,ny-2)<<endl;
    cout<<"u0(1,1)(1,ny-2)(nx-2,1)(nx-2,ny-2)  ";
    cout<<u0(1,1)<<","<<u0(1,ny-2)<<",";
    cout<<u0(nx-2,1)<<","<<u0(nx-2,ny-2)<<endl;
    cout<<"u.mean = () "<<psi.mean()<<endl;
    cout<<"u0.mean = () "<<u0.mean()<<endl;
    Grid uu1(psi-u0);
    double err_norm1 = uu1.abs_mean();
    Grid uu2(uu1*uu1);
    double err_norm2 = sqrt(uu2.sum()/uu2.size());
    cout<<"max error: "<<blitz::max(blitz::abs(uu1.data()))<<endl;
    cout<<"infinite norm of error: "<<err_norm1<<endl;
    cout<<"L2 norm of error: "<<err_norm2<<endl;
    cout<<endl;
}

void test_fieldE_mud2sp(){
    Config cfg("param.ini");

    double lx=2.0 * PI;
    //double ly=2.0 * PI;
    int Lx=128;
    int Ly=128;
    int Lz=1;
    int eps_type = 0;
    cfg.set_integer("Grid","dimension",2);
    cfg.set_integer("Grid","Lx",Lx);
    cfg.set_integer("Grid","Ly",Ly);
    cfg.set_integer("Grid","Lz",Lz);
    cfg.set_string("UnitCell","CrystalSystemType","Square");
    cfg.set_double("UnitCell","a",lx);
    cfg.set_integer("Algorithm","dielectric_constant",0);
    UnitCell uc(cfg);

    FieldE psi("psi",eps_type,cfg,1.0,1.0); // default Updater for 2D

    cout<<"psi.name = (psi) "<<psi.name()<<endl;
    cout<<"psi.dim = (2) "<<psi.dim()<<endl;
    cout<<"psi.(Lx,Ly) = (128,128) "<<"("<<psi.Lx()<<","<<psi.Ly()<<")"<<endl;
    cout<<"psi.(dx,dy) = (0.0490874,0.0490874)"<<"("<<psi.dx()<<","<<psi.dy()<<")"<<endl;
    cout<<endl;

    Grid u0(uc,Lx,Ly,Lz);
    Grid rho(uc,Lx,Ly,Lz);
    double dx=psi.dx();
    double dy=psi.dy();
    double x;
    double y;
    for(int i=0;i<Lx;i++)
      for(int j=0;j<Ly;j++){
          x = (i + 0.5) * dx;
          y = (j + 0.5) * dy;
          rho(i,j) = -(cos(x) + sin(y));
          u0(i,j) = cos(x) + sin(y);
      }
    psi.update(rho);
    //psi.update(rho,up);
    psi.save("psi.mat");
    u0.set_name("u0");
    u0.save("psi.mat");

    int nx = Lx;
    int ny = Ly;
    cout<<"u(0,0)(0,ny-1)(nx-1,0)(nx-1,ny-1)  ";
    cout<<psi(0,0)<<","<<psi(0,ny-1)<<",";
    cout<<psi(nx-1,0)<<","<<psi(nx-1,ny-1)<<endl;
    cout<<"u0(0,0)(0,ny-1)(nx-1,0)(nx-1,ny-1)  ";
    cout<<u0(0,0)<<","<<u0(0,ny-1)<<",";
    cout<<u0(nx-1,0)<<","<<u0(nx-1,ny-1)<<endl;
    cout<<"u(1,1)(1,np-2)(nt-2,1)(nt-2,np-2)  ";
    cout<<psi(1,1)<<","<<psi(1,ny-2)<<",";
    cout<<psi(nx-2,1)<<","<<psi(nx-2,ny-2)<<endl;
    cout<<"u0(1,1)(1,ny-2)(nx-2,1)(nx-2,ny-2)  ";
    cout<<u0(1,1)<<","<<u0(1,ny-2)<<",";
    cout<<u0(nx-2,1)<<","<<u0(nx-2,ny-2)<<endl;
    cout<<"u.mean = () "<<psi.mean()<<endl;
    cout<<"u0.mean = () "<<u0.mean()<<endl;
    Grid uu1(psi-u0);
    double err_norm1 = uu1.abs_mean();
    Grid uu2(uu1*uu1);
    double err_norm2 = sqrt(uu2.sum()/uu2.size());
    cout<<"max error: "<<blitz::max(blitz::abs(uu1.data()))<<endl;
    cout<<"infinite norm of error: "<<err_norm1<<endl;
    cout<<"L2 norm of error: "<<err_norm2<<endl;
    cout<<endl;
}

int main(){
    test_fieldE_mud2sp();
    test_fieldE_mud2sa();

    return 0;
}
