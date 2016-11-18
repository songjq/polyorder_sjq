/*
 * =====================================================================================
 *
 *       Filename:  test_DiffMat.cc
 *
 *    Description:  aim to test the differentiation matrices  
 *
 *        Version:  1.0
 *        Created:  07/21/2016 11:38:39 AM
 *       Revision:  none
 *       Compiler:  icc
 *
 *         Author:  songjq
 *        Company:  
 *
 * =====================================================================================
 */
#include <iostream>
#include "armadillo"
#include <blitz/array.h>
#include "Cheb.h"
#include "Boundary.h"

using namespace std;
using namespace arma;
using namespace blitz;

void test_armadillo() {
	int Nx = 3;
	uword N = Nx;
	mat A = eye<mat>(N, N);
	A.print("A = ");

	mat B = randu<mat>(N, N);
	B.print("B = ");
    Array<double, 2> C(Nx, Nx);
    for(int i=0; i<Nx; i++)
    	for(int j=0; j<Nx; j++)
    		C(i, j) = B(i, j);
    cout << "C = " << C << endl;

	colvec u = linspace(0, N-1, N);
	u.print("u = ");

	double x = A(2,2);
	cout << "x = " << x << endl;

	double y = u(2);
	cout << "y = " << y << endl;

	mat D;
	D = A + B;
	D.print("D+ = ");
	D = A % B;
    D.print("D% = ");
    
    mat U = A * B;
    U.print("U = ");

    vec v = B * u;
    v.print("v = ");


}

void test_diffmat() {
	const double kPI = 3.141592653589793;
	Boundary dbc = Boundary(0, 1, 0);
    Boundary nbc = Boundary(1, 0, 0);
    Boundary rbc = Boundary(1, 1, 0);
	uword N = 20;
    Cheb cheb(N);
    vec x = cheb.x();
    x = (1.0-x)/2 * 3;
    x.print("x = ");
    double d = 0.3;
    int Ld;
    //L_depletion = (int)((Lz-1)/kPI*acos(1-2*deplen/Lc));
    Ld = (int)((N-1)/kPI*acos(1-2*d/3));
    cout << "Ld = " << Ld;
    vec y = zeros<vec>(N);
    for (int i=0; i<Ld; i++)
    	y(i) = (1+cos(kPI * x(i)/d))/2.0;
    y.print("y = ");
    double sum = 0.5 * cheb.quadrature_clencurt(y) * 3;
    cout << "sum is " << sum << endl;
    y = y * 0.15 /sum;
    y.print("y2 = ");

    /*
    //vec y = exp(x) % sin(5*x);
    y.print("y = ");
    vec dy = exp(x) % (sin(5*x)+5*cos(5*x));
    //dy.print("dy = ");
    mat D = cheb.D();
    vec dy_num = -D * y;  //D * y is not right
    //dy_num.print("dy_num = ");
    vec Er = abs(dy-dy_num);
    //Er.print("Er = ");

    mat D_nbc = cheb.D(nbc, nbc);
    vec dy_nbc = -D_nbc * y;
    vec Er_nbc = abs(dy-dy_nbc); //The result will be quite different with specific boundary condition
    //Er_nbc.print("Er_nbc = "); */
    double vave = 0.5 * cheb.quadrature_clencurt(y) * 3;
    cout << "Intergral_of_y is " << vave << endl;
}

int main() {
	//test_armadillo();
	test_diffmat();
	return 0;
}
