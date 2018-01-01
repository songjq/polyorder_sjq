/*
 * =====================================================================================
 *
 *       Filename:  IniField_fromFredri.cc
 *
 *    Description:  make initial field via definition of Fredrickson in P110 
 *
 *        Version:  1.0
 *        Created:  08/07/2016 07:20:06 PM
 *       Revision:  none
 *       Compiler:  icc
 *
 *         Author:  songjq, 
 *        Company:  
 *
 * =====================================================================================
 */

#include <iostream>
#include "armadillo"
#include <blitz/array.h>
//#include "Cheb.h"
#include "CMatFile.h"

using namespace std;
using namespace arma;
using namespace blitz;
const double kPI = 3.141592653589793;

void make_w_vLam() {
    int Nx = 32;
    int Ny = 15;
    int Nz = 16;
    int lx = 10;
 
    blitz::Range all = blitz::Range::all();
    Array<double, 1> x(Nx);
    Array<double, 1> y(Nx);
    Array<double, 1> f(Nx);
    Array<double, 2> w2d(Nx, Ny);
    Array<double, 3> w3d(Nx, Ny, Nz);
    for (int i=0; i<Nx; i++) 
    	x(i) = 1.0*i/(Nx-1)*lx;
    y = 3.*(x-lx/2)/2;
    y = 2./(exp(y)+exp(-y));
    f = 1-2.*y*y;    

    for (int i=0; i<Nx; i++)
    	w2d(i, all) = f(i);

    for (int i=0; i<Nx; i++)
    	w3d(i, all, all) = f(i);

    CMatFile mat;
    mat.matInit("w_vLam.mat","w");
    mwSize Lx = (mwSize) Nx;
    mwSize Ly = (mwSize) Ny;
    mwSize Lz = (mwSize) Nz;
    mwSize dim_array3d[3]={Lx, Ly, Lz};
    blitz::Array<double, 3> data3d(Lx, Ly, Lz, fortranArray);
    data3d = w3d;
    mat.matPut("w3d",data3d.data(),data3d.size()*sizeof(double),3,dim_array3d,mxDOUBLE_CLASS,mxREAL);

    mwSize dim_array2d[2]={Lx, Ly};
    blitz::Array<double, 2> data2d(Lx, Ly, fortranArray);
    data2d = w2d;
    mat.matPut("w2d",data2d.data(),data2d.size()*sizeof(double),2,dim_array2d,mxDOUBLE_CLASS,mxREAL);
    mat.matRelease();
}

/*void make_w_pLam() {
	int Nx = 64;
	int Ny = 64;
	int Nz = 64;
	int lz = 10;
	blitz::Range all = blitz::Range::all();
	Array<double, 2> w2d(Nx, Ny);
	Array<double, 3> w3d(Nx, Ny, Nz);

	Cheb cheb(Nz);
	vec x = cheb.x();
	x = (1.0-x)/2 * lz;
	//x.print("x = ");
	vec y = 3.0*(x-lz/2)/2;
	y = 2.0/(exp(y)+exp(-y));
	vec f = 1-2.0*y%y;

    for (int j=0; j<Ny; j++) 
    		w2d(all, j) = f(j);

    for (int k=0; k<Nz; k++) 
    			w3d(all, all, k) = f(k);

    CMatFile mat;
    mat.matInit("w_pLam.mat","w");
    mwSize Lx = (mwSize) Nx;
    mwSize Ly = (mwSize) Ny;
    mwSize Lz = (mwSize) Nz;
    mwSize dim_array3d[3]={Lx, Ly, Lz};
    blitz::Array<double, 3> data3d(Lx, Ly, Lz, fortranArray);
    data3d = w3d;
    mat.matPut("w3d",data3d.data(),data3d.size()*sizeof(double),3,dim_array3d,mxDOUBLE_CLASS,mxREAL);

    mwSize dim_array2d[2]={Lx, Ly};
    blitz::Array<double, 2> data2d(Lx, Ly, fortranArray);
    data2d = w2d;
    mat.matPut("w2d",data2d.data(),data2d.size()*sizeof(double),2,dim_array2d,mxDOUBLE_CLASS,mxREAL);
    mat.matRelease();

}*/


int main() {
	make_w_vLam();
	//make_w_pLam();

	return 0;
}