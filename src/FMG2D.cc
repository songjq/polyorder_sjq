/**
 * FMG2D.cc
 * Created at 2011.6.21
 *
 * Implementation of FMG2D.h.
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

#include "FMG2D.h"

//////////////////////////////////////////////////////////////////////////
//
//          PUBLIC METHOD
//
//////////////////////////////////////////////////////////////////////////

/* constructor
* On input rho0[0..n-1][0..n-1] contains the right-hand side rho. 
* The dimension n must be of the form 2^j+1 for some integer j. 
* (j is actually the number of grid levels used in the solution, 
* called ng below.) ncycle is the number of V-cycles to be used
* at each level.
*/
FMG2D::FMG2D(MatDoub_IO &rho0, const Doub llx, const Doub lly, const Int nncycle):n(rho0.nrows()),ng(0),lx(llx),ly(lly),ncycle(nncycle),NPRE(1),NPOST(1)
{
	Int nn=n;
	while (nn >>= 1) ng++;
	if ((n-1) != (1 << ng))
		throw("n-1 must be a power of 2 in mglin.");
	nn=n;
	Int ngrid=ng-1;
	rho.resize(ng);
	rho[ngrid] = new MatDoub(nn,nn);
	*rho[ngrid]=rho0;
	while (nn > 3) {
		nn=nn/2+1;
		rho[--ngrid]=new MatDoub(nn,nn);
		rstrct(*rho[ngrid],*rho[ngrid+1]);
	}
}

/* vcycle
* V-cycle Scheme 
*/
/*
void FMG2D::vcycle(MatDoub_IO &u,double &resnorm)
{
    for(Int jpre=0;jpre<NPRE;jpre++) relax(u,rhs);
    resid(res2,u,rhs);
    rstrct(res1,res2);
    interp_bilinear();
}*/

/* mglin
* Full Multigrid Scheme (FMG)
*/
void FMG2D::mglin(MatDoub_IO &u,double &resnorm)
{
	int nn=3;
	uj=new MatDoub(nn,nn);	
	slvsml(*uj,*rho[0]);
	for (Int j=1;j<ng;j++) {
		nn=2*nn-1;
		uj1=uj;
		uj=new MatDoub(nn,nn);
		interp_bicubic(*uj,*uj1);
		delete uj1;	// NOTE: Here uj1 point to the previous uj.
		for (Int jcycle=0;jcycle<ncycle;jcycle++)
			mg(j,*uj,*rho[j]);
	}
	u = *uj;
    resnorm=0.0;

    // Find the norm1 of residual (error)
	MatDoub res(nn,nn);
	resid(res,u,*rho[ng-1]);
	for(int i=0;i<nn;i++)
		for(int j=0;j<nn;j++){
            resnorm+=abs(res[i][j]);
		}
    resnorm/=(nn*nn);

	/* Periodic boundary condition correction
	 if u is a solution, than u+c is also a solution
	 We choose the solution u with <u>=0.
	*/
	Doub sum=0.0;
	for(Int i=0;i<u.nrows();i++)
		for(Int j=0;j<u.nrows();j++)
			sum+=u[i][j];
	for(Int i=0;i<u.nrows();i++)
		for(Int j=0;j<u.nrows();j++)
			u[i][j]-=(sum/u.size());
}

FMG2D::~FMG2D()
{
	if (uj != NULL) delete uj;
	for (Int j=0;j<ng;j++)
		if (rho[j] != NULL) delete rho[j];
}

//////////////////////////////////////////////////////////////////////////
//
//          PRIVATE METHOD
//
//////////////////////////////////////////////////////////////////////////

/* interp
* Coarse-to fine prolongation by bilinear interpolation. If nf is the fine-grid 
* dimension, the coarse-grid solution is input as uc[0..nc-1][0..nc-1], 
* where nc=nf/2+1. The fine-grid solution is returned in uf[0..nf-1][0..nf-1].
*/
void FMG2D::interp_bilinear(MatDoub_O &uf, MatDoub_I &uc)
{
	Int nf=uf.nrows();
	Int nc=nf/2+1;
	// Do elements that are copies
	for (Int ic=0;ic<nc;ic++)
		for (Int jc=0;jc<nc;jc++) uf[2*ic][2*jc]=uc[ic][jc];
	// Do even-numbered columns, interpolating vertically
	for (Int iif=1;iif<nf-1;iif+=2)
	    for (Int jf=0;jf<nf;jf+=2)
			uf[iif][jf]=0.5*(uf[iif+1][jf]+uf[iif-1][jf]);
	// Do odd-numbered columns, interpolating horizontally
	for (Int iif=0;iif<nf;iif++)
	    for (Int jf=1;jf<nf-1;jf+=2)
			uf[iif][jf]=0.5*(uf[iif][jf+1]+uf[iif][jf-1]);
}

/* interp_bicubic
* Coarse-to fine prolongation by bicubic interpolation. If nf is the fine-grid
* dimension, the coarse-grid solution is input as uc[0..nc-1][0..nc-1],
* where nc=nf/2+1. The fine-grid solution is returned in uf[0..nf-1][0..nf-1].
*/
void FMG2D::interp_bicubic(MatDoub_O &uf, MatDoub_I &uc)
{
    Int nf=uf.nrows();
    Int nc=nf/2+1;
    // Do elements that are copies
    for (Int ic=0;ic<nc;ic++)
        for (Int jc=0;jc<nc;jc++) uf[2*ic][2*jc]=uc[ic][jc];
    // Do even-numbered columns, interpolating vertically
    for (Int iif=3;iif<nf-3;iif+=2)
        for (Int jf=0;jf<nf;jf+=2)
            uf[iif][jf]=-0.0625*(uf[iif+3][jf]+uf[iif-3][jf])+0.5625*(uf[iif+1][jf]+uf[iif-1][jf]);
    // boundary grids, using bilinear
    for (Int jf=0;jf<nf;jf+=2){
        uf[1][jf]=0.5*(uf[2][jf]+uf[0][jf]);
        uf[nf-2][jf]=0.5*(uf[nf-1][jf]+uf[nf-3][jf]);
    }
    // Do odd-numbered columns, interpolating horizontally
    for (Int iif=0;iif<nf;iif++)
        for (Int jf=3;jf<nf-2;jf+=2)
            uf[iif][jf]=-0.0625*(uf[iif][jf+3]+uf[iif][jf-3])+0.5625*(uf[iif][jf+1]+uf[iif][jf-1]);
    // boundary grids, using biliear
    for (Int iif=0;iif<nf;iif++){
        uf[iif][1]=0.5*(uf[iif][0]+uf[iif][2]);
        uf[iif][nf-2]=0.5*(uf[iif][nf-1]+uf[iif][nf-3]);
    }
}

/* addint
* Does coarse-to-fine interpolation and adds result to uf. If nf is the 
* fine-grid dimension, the coarse-grid solution is input as 
* uc[0..nc-1][0..nc-1], where nc D nf=2 C 1. The fine-grid solution is 
* returned in uf[0..nf-1][0..nf-1]. res[0..nf-1][0..nf-1] is used
* for temporary storage.
*/
void FMG2D::addint(MatDoub_O &uf, MatDoub_I &uc, MatDoub_O &res)
{
	Int nf=uf.nrows();
	interp_bilinear(res,uc);
	for (Int i=0;i<nf;i++)
	    for (Int j=0;j<nf;j++)
			uf[i][j] += res[i][j];
}

/* slvsml
* Solution of the model problem on the coarsest grid, where h=1/2. 
* The right-hand side is input in rhs[0..2][0..2] and the solution is 
* returned in u[0..2][0..2].
*/
void FMG2D::slvsml(MatDoub_O &u, MatDoub_I &rhs)
{
	Doub hx=lx/2.0;
	Doub h2x=1.0/(hx*hx);
	Doub hy=ly/2.0;
	Doub h2y=1.0/(hy*hy);
	Doub h2=0.5/(h2x+h2y);
	
    u[1][1]=h2*h2x*(u[2][1]+u[0][1])+h2*h2y*(u[1][2]+u[1][0])-h2*rhs[1][1];

	Doub sum=0.0;
	for(int i=0;i<3;i++)
		for(int j=0;j<3;j++)
			sum+=u[i][j];
	for(int i=0;i<3;i++)
		for(int j=0;j<3;j++)
			u[i][j]-=(sum/9.0);
}

/* relax
* Red-black Gauss-Seidel relaxation for model problem. Updates the current value of the
* solution u[0..n-1][0..n-1], using the right-hand side function rhs[0..n-1][0..n-1].
*/
void FMG2D::relax(MatDoub_IO &u, MatDoub_I &rhs)
{
	Int n=u.nrows();
	Doub hx=lx/(n-1);
	Doub h2x=1.0/(hx*hx);
	Doub hy=ly/(n-1);
	Doub h2y=1.0/(hy*hy);
	Doub h2=0.5/(h2x+h2y);
	hx=h2*h2x;
	hy=h2*h2y;
	
    /*
	for (Int ipass=0,jsw=1;ipass<2;ipass++,jsw=3-jsw) {
		for (Int j=1,isw=jsw;j<n-1;j++,isw=3-isw)
			for (Int i=isw;i<n-1;i+=2)
				u[i][j]=0.25*(u[i+1][j]+u[i-1][j]+u[i][j+1]
						+u[i][j-1]-h2*rhs[i][j]);
	}
    */
	for (Int i=1;i<n-1;i+=2)
	    for (Int j=1;j<n-1;j+=2)
		    u[i][j]=hx*(u[i+1][j]+u[i-1][j])+hy*(u[i][j+1]+u[i][j-1])-h2*rhs[i][j];
	for (Int i=2;i<n-2;i+=2)
	    for (Int j=2;j<n-2;j+=2)
		    u[i][j]=hx*(u[i+1][j]+u[i-1][j])+hy*(u[i][j+1]+u[i][j-1])-h2*rhs[i][j];
	for (Int i=1;i<n-1;i+=2)
	    for (Int j=2;j<n-2;j+=2)
		    u[i][j]=hx*(u[i+1][j]+u[i-1][j])+hy*(u[i][j+1]+u[i][j-1])-h2*rhs[i][j];
	for (Int i=2;i<n-2;i+=2)
	    for (Int j=1;j<n-1;j+=2)
		    u[i][j]=hx*(u[i+1][j]+u[i-1][j])+hy*(u[i][j+1]+u[i][j-1])-h2*rhs[i][j];
	// boundary points
	for(int i=1;i<n-1;i++){
		u[0][i]=hy*(u[0][i-1]+u[0][i+1])+hx*(u[1][i]+u[n-2][i])-h2*rhs[0][i];
		u[i][0]=hx*(u[i-1][0]+u[i+1][0])+hy*(u[i][1]+u[i][n-2])-h2*rhs[i][0];
		u[i][n-1]=hx*(u[i-1][n-1]+u[i+1][n-1])+hy*(u[i][n-2]+u[i][1])-h2*rhs[i][n-1];
		u[n-1][i]=hy*(u[n-1][i-1]+u[n-1][i+1])+hx*(u[n-2][i]+u[1][i])-h2*rhs[n-1][i];
	}
	// corner points
	u[0][0]=hy*(u[0][1]+u[0][n-2])+hx*(u[1][0]+u[n-2][0])-h2*rhs[0][0];
	u[0][n-1]=hy*(u[0][n-2]+u[0][1])+hx*(u[n-2][n-1]+u[1][n-1])-h2*rhs[0][n-1];
	u[n-1][0]=hx*(u[n-2][0]+u[1][0])+hy*(u[n-1][1]+u[n-1][n-2])-h2*rhs[n-1][0];
	u[n-1][n-1]=hx*(u[n-2][n-1]+u[1][n-1])+hy*(u[n-1][n-2]+u[n-1][1])-h2*rhs[n-1][n-1];
}

/* resid
* Returns minus the residual for the model problem. Input quantities 
* are u[0..n-1][0..n-1] and rhs[0..n-1][0..n-1], while res[0..n-1][0..n-1]
* is returned.
*/
void FMG2D::resid(MatDoub_O &res, MatDoub_I &u, MatDoub_I &rhs)
{
	Int n=u.nrows();
	Doub hx=lx/(n-1);
	Doub h2x=1.0/(hx*hx);
	Doub hy=ly/(n-1);
	Doub h2y=1.0/(hy*hy);
    Doub h2=2.0*(h2x+h2y);
	// interior points
	for (Int i=1;i<n-1;i++)
	    for (Int j=1;j<n-1;j++)
			res[i][j] = -h2x*(u[i+1][j]+u[i-1][j])-h2y*(u[i][j+1]
						+u[i][j-1])+h2*u[i][j]+rhs[i][j];
	// boundary points 
	for (Int i=1;i<n-1;i++){
		res[0][i]=-h2y*(u[0][i-1]+u[0][i+1])-h2x*(u[1][i]+u[n-2][i])+h2*u[0][i]+rhs[0][i];
		res[n-1][i]=-h2y*(u[n-1][i-1]+u[n-1][i+1])-h2x*(u[n-2][i]+u[1][i])+h2*u[n-1][i]+rhs[n-1][i];
		res[i][0]=-h2x*(u[i-1][0]+u[i+1][0])-h2y*(u[i][1]+u[i][n-2])+h2*u[i][0]+rhs[i][0];
		res[i][n-1]=-h2x*(u[i-1][n-1]+u[i+1][n-1])-h2y*(u[i][n-2]+u[i][1])+h2*u[i][n-1]+rhs[i][n-1];
	}
	// corner points
	res[0][0]=-h2y*(u[0][1]+u[0][n-2])-h2x*(u[1][0]+u[n-2][0])+h2*u[0][0]+rhs[0][0];
	res[0][n-1]=-h2y*(u[0][n-2]+u[0][1])-h2x*(u[1][n-1]+u[n-2][n-1])+h2*u[0][n-1]+rhs[0][n-1];
	res[n-1][0]=-h2y*(u[n-1][1]+u[n-1][n-2])-h2x*(u[n-2][0]+u[1][0])+h2*u[n-1][0]+rhs[n-1][0];
	res[n-1][n-1]=-h2y*(u[n-1][n-2]+u[n-1][1])-h2x*(u[n-2][n-1]+u[1][n-1])+h2*u[n-1][n-1]+rhs[n-1][n-1];
}

/* rstrct
* Half-weighting restriction. If nc is the coarse-grid dimension, 
* the fine-grid solution is input in uf[0..2*nc-2][0..2*nc-2]. The coarse-grid 
* solution obtained by restriction is returned in uc[0..nc-1][0..nc-1].
*/
void FMG2D::rstrct(MatDoub_O &uc, MatDoub_I &uf)
{
	Int nc=uc.nrows();
	Int ncc=2*nc-2;
    /*
	// interior points, half-weighting restriction
	for (Int jf=2,jc=1;jc<nc-1;jc++,jf+=2) {
		for (Int iif=2,ic=1;ic<nc-1;ic++,iif+=2) {
			uc[ic][jc]=0.5*uf[iif][jf]+0.125*(uf[iif+1][jf]+uf[iif-1][jf]
			+uf[iif][jf+1]+uf[iif][jf-1]);
		}
	}
    */
	// interior points, full-weighting restriction
	for (Int jf=2,jc=1;jc<nc-1;jc++,jf+=2)
		for (Int iif=2,ic=1;ic<nc-1;ic++,iif+=2)
			uc[ic][jc]=0.25*uf[iif][jf]+0.125*(uf[iif+1][jf]+uf[iif-1][jf]+uf[iif][jf+1]+uf[iif][jf-1])+0.0625*(uf[iif+1][jf+1]+uf[iif-1][jf-1]+uf[iif-1][jf+1]+uf[iif+1][jf-1]);
	// boundary points
	for (Int jc=2,ic=1;ic<nc-1;ic++,jc+=2) {
		uc[ic][0]=0.5*uf[jc][0]+0.125*(uf[jc-1][0]+uf[jc+1][0]+uf[jc][ncc-1]+uf[jc][1]);
		uc[ic][nc-1]=0.5*uf[jc][ncc]+0.125*(uf[jc-1][ncc]+uf[jc+1][ncc]+uf[jc][ncc-1]+uf[jc][1]);
	}
	for (Int jc=2,ic=1;ic<nc-1;ic++,jc+=2) {
		uc[0][ic]=0.5*uf[0][jc]+0.125*(uf[0][jc-1]+uf[0][jc+1]+uf[1][jc]+uf[ncc-1][jc]);
		uc[nc-1][ic]=0.5*uf[ncc][jc]+0.125*(uf[ncc][jc-1]+uf[ncc][jc+1]+uf[ncc-1][jc]+uf[1][jc]);
	}
	// corner points
	uc[0][0]=0.5*uf[0][0]+0.125*(uf[0][1]+uf[0][ncc-1]+uf[1][0]+uf[ncc-1][0]);
	uc[0][nc-1]=0.5*uf[0][ncc]+0.125*(uf[0][1]+uf[0][ncc-1]+uf[1][ncc]+uf[ncc-1][ncc]);
	uc[nc-1][0]=0.5*uf[ncc][0]+0.125*(uf[ncc][1]+uf[ncc][ncc-1]+uf[1][0]+uf[ncc-1][0]);
	uc[nc-1][nc-1]=0.5*uf[ncc][ncc]+0.125*(uf[ncc][1]+uf[ncc][ncc-1]+uf[ncc-1][ncc]+uf[1][ncc]);
}

/* mg
* Recursive multigrid iteration. On input, j is the current level, u is the 
* current value of the solution, and rhs is the right-hand side. On output 
* u contains the improved solution at the current level.
*/
void FMG2D::mg(Int j, MatDoub_IO &u, MatDoub_I &rhs)
{
	Int nf=u.nrows();
	Int nc=(nf+1)/2;
	if (j == 0)
		slvsml(u,rhs);
	else {
		MatDoub res(nc,nc),v(nc,nc,0.0),temp(nf,nf);
		for (Int jpre=0;jpre<NPRE;jpre++)
			relax(u,rhs);
		resid(temp,u,rhs);
		rstrct(res,temp);
		mg(j-1,v,res);
		addint(u,v,temp);
		for (Int jpost=0;jpost<NPOST;jpost++)
			relax(u,rhs);
	}
}
