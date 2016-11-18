/**
 * multgrid.cc
 * Created at 2012.4.12
 *
 * Implemetation of multigrid.h
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

#include "multigrid.h"

/**
 * Interpolate coare grid uc to fine grid uf by linear method.
 * uc has nc points, the layout looks like:
 *
 * *.........*.........*.........*.........*
 * 0         1         2                  nc-1
 *
 * uf has nf = 2*nc-1 points,the layout looks like:
 *
 * |....|....|....|....|....|....|....|....|
 * 0    1    2    3    4                  2*(nc-1) 
 *
 */
void interp_linear_1d(const blitz::Array<double,1> uc,blitz::Array<double,1> uf){
    int nf = uf.rows();
    int xa = 0;
    int xb = nf - 1;
    blitz::Range rx0(xa,xb,2);
    blitz::Range rx01(xa,xb-2,2);
    blitz::Range rx02(xa+2,xb,2);
    blitz::Range rx1(xa+1,xb-1,2);

    uf(rx0) = uc;
    uf(rx1) = 0.5 * (uf(rx01) + uf(rx02));
}

/**
 * Interpolate coarse grid uc to fine grid uf by cubic method.
 * uc has nc points, the layout looks like:
 * 
 * For uf_i, i = 0, 1, 2, ..., 2*(nc-1)
 * Only the internal grids uf_i (i = 3, 5, 7, ..., xb-3) are cubic
 * interpolated.
 * uf_1 and uf_xb-1 are linear interpolated.
 * 
 */
void interp_cubic_1d(const blitz::Array<double,1> uc,blitz::Array<double,1> uf){
    int nf = uf.rows();
    int xa = 0;
    int xb = nf - 1;
    blitz::Range rx0(xa,xb,2);
    blitz::Range rx01(xa+0,xb-6,2);
    blitz::Range rx02(xa+2,xb-4,2);
    blitz::Range rx03(xa+4,xb-2,2);
    blitz::Range rx04(xa+6,xb-0,2);
    blitz::Range rx1(xa+3,xb-3,2);

    uf(rx0) = uc;
    uf(1) = 0.5 * (uf(0) + uf(2));
    uf(xb-1) = 0.5 * (uf(xb-2) + uf(xb));
    uf(rx1) = 0.0625 * (9.0*(uf(rx02) + uf(rx03)) - (uf(rx01) + uf(rx04)));
}

void rstrct_injection_1d(const blitz::Array<double,1> uf,blitz::Array<double,1> uc){
    int nf = uf.rows();
    int xa = 0;
    int xb = nf - 1;
    blitz::Range rx(xa,xb,2);

    uc = uf(rx);
}

/**
 * Restrict fine grid uf to coarse grid uc by weight method.
 * The scheme is:
 * *.........*.........*.........*.........*
 * 0         1         2                  nc-1
 *       /   |   \  /  |   \
 * |....|....|....|....|....|....|....|....|
 * 0    1    2    3    4                  2*(nc-1) 
 *
 * Note: the boundary points are directly copied without weight.
 *
 */
void rstrct_weight_1d(const blitz::Array<double,1> uf,blitz::Array<double,1> uc){
    int nf = uf.rows();
    int nc = uc.rows();
    int xfa = 0;
    int xfb = nf - 1; // xfb = 2*nc-2, thus it is even
    int xca = 0;
    int xcb = nc - 1;
    blitz::Range rxc(xca+1,xcb-1);
    blitz::Range rx1(xfa+1,xfb-3,2);
    blitz::Range rx2(xfa+2,xfb-2,2);
    blitz::Range rx3(xfa+3,xfb-1,2);

    uc(xca) = uf(xfa);
    uc(xcb) = uf(xfb);
    uc(rxc) = 0.25 * (uf(rx1) + 2.0 * uf(rx2) + uf(rx3));
}

/**
 * Interpolate coarse grid uc to fine grid uf by bilinear method.
 * uc has ncx x ncy points, the layout looks like:
 *
 * *.........*.........*.........*.........*.........*   ncy-1
 * |         |         |         |         |         |
 * |         |         |         |         |         |
 * |         |         |         |         |         |
 * *.........*.........*.........*.........*.........*   
 * |         |         |         |         |         |
 * |         |         |         |         |         |
 * |         |         |         |         |         |
 * *.........*.........*.........*.........*.........*   3
 * |         |         |         |         |         |
 * |         |         |         |         |         |
 * |         |         |         |         |         |
 * *.........*.........*.........*.........*.........*   2
 * |         |         |         |         |         |
 * |         |         |         |         |         |
 * |         |         |         |         |         |
 * *.........*.........*.........*.........*.........*   1
 * |         |         |         |         |         |
 * |         |         |         |         |         |
 * |         |         |         |         |         |
 * *.........*.........*.........*.........*.........*   0
 * 0         1         2                           ncx-1
 * 
 * uf has nfx x nfy = (2*ncx-1) x (2*ncy-1) points.
 * the layout looks like:
 *
 * |....|....|....|....|....|....|....|....|....|....|  2*(ncy-1)
 * |    |    |    |    |    |    |    |    |    |    |
 * |....|....|....|....|....|....|....|....|....|....|  
 * |    |    |    |    |    |    |    |    |    |    |
 * |....|....|....|....|....|....|....|....|....|....|
 * |    |    |    |    |    |    |    |    |    |    |
 * |....|....|....|....|....|....|....|....|....|....|  
 * |    |    |    |    |    |    |    |    |    |    |
 * |....|....|....|....|....|....|....|....|....|....|
 * |    |    |    |    |    |    |    |    |    |    |
 * |....|....|....|....|....|....|....|....|....|....|  
 * |    |    |    |    |    |    |    |    |    |    |
 * |....|....|....|....|....|....|....|....|....|....|  4
 * |    |    |    |    |    |    |    |    |    |    |
 * |....|....|....|....|....|....|....|....|....|....|  3
 * |    |    |    |    |    |    |    |    |    |    |
 * |....|....|....|....|....|....|....|....|....|....|  2
 * |    |    |    |    |    |    |    |    |    |    |
 * |....|....|....|....|....|....|....|....|....|....|  1
 * |    |    |    |    |    |    |    |    |    |    |
 * |....|....|....|....|....|....|....|....|....|....|  0
 * 0    1    2    3    4                        2*(ncx-1)
 *
 */
void interp_bilinear_2d(const blitz::Array<double,2> uc,blitz::Array<double,2> uf){
    int nfx = uf.rows();
    int nfy = uf.cols();
    int xfa = 0;
    int xfb = nfx - 1;
    int yfa = 0;
    int yfb = nfy - 1;
    int ncx = uc.rows();
    int xca = 0;
    int xcb = ncx - 1;
    blitz::Range all = blitz::Range::all();
    blitz::Range rx(xfa,xfb,2);

    for(int i=xca;i<=xcb;i++)
        interp_linear_1d(uc(i,all),uf(2*i,all));
    for(int i=yfa;i<=yfb;i++)
        interp_linear_1d(uf(rx,i),uf(all,i));
}

void interp_bicubic_2d(const blitz::Array<double,2> uc,blitz::Array<double,2> uf){
    int nfx = uf.rows();
    int nfy = uf.cols();
    int xfa = 0;
    int xfb = nfx - 1;
    int yfa = 0;
    int yfb = nfy - 1;
    int ncx = uc.rows();
    int xca = 0;
    int xcb = ncx - 1;
    blitz::Range all = blitz::Range::all();
    blitz::Range rx(xfa,xfb,2);

    for(int i=xca;i<=xcb;i++)
        interp_cubic_1d(uc(i,all),uf(2*i,all));
    for(int i=yfa;i<=yfb;i++)
        interp_cubic_1d(uf(rx,i),uf(all,i));
}

void rstrct_injection_2d(const blitz::Array<double,2> uf,blitz::Array<double,2> uc){
    int nfx = uf.rows();
    int nfy = uf.cols();
    int xfa = 0;
    int xfb = nfx - 1;
    int yfa = 0;
    int yfb = nfy - 1;
    blitz::Range rx(xfa,xfb,2);
    blitz::Range ry(yfa,yfb,2);

    uc = uf(rx,ry);
}

/**
 * Restrict fine grid uf to coarse grid uc by half-weight method.
 * uc has ncx x ncy points.
 * uf has nfx x nfy = (2*ncx-1) x (2*ncy-1) points.
 * the layout looks like:
 *
 * *....|....*....|....*....|....*....|....*....|....*  2*(ncy-1)
 * |    |    |    |    |    |    |    |    |    |    |
 * |....|....@....|....|....|....|....|....|....|....|  
 * |    |    |    |    |    |    |    |    |    |    |
 * *....@....*....@....*....|....*....|....*....|....*
 * |    |    |    |    |    |    |    |    |    |    |
 * |....|....@....|....|....|....|....|....|....|....|  
 * |    |    |    |    |    |    |    |    |    |    |
 * *....|....*....|....*....|....*....|....*....|....*
 * |    |    |    |    |    |    |    |    |    |    |
 * |....|....|....|....|....|....|....|....|....|....|  
 * |    |    |    |    |    |    |    |    |    |    |
 * *....|....*....|....*....|....*....|....*....|....*  4
 * |    |    |    |    |    |    |    |    |    |    |
 * |....|....|....|....|....|....|....|....|....|....|  3
 * |    |    |    |    |    |    |    |    |    |    |
 * *....|....*....|....*....|....*....|....*....|....*  2
 * |    |    |    |    |    |    |    |    |    |    |
 * |....|....|....|....|....|....|....|....|....|....|  1
 * |    |    |    |    |    |    |    |    |    |    |
 * *....|....*....|....*....|....*....|....*....|....*  0
 * 0    1    2    3    4                        2*(ncx-1)
 *
 */
void rstrct_hw_2d(const blitz::Array<double,2> uf,blitz::Array<double,2> uc){
    int nfx = uf.rows();
    int nfy = uf.cols();
    int xfa = 0;
    int xfb = nfx - 1;
    int yfa = 0;
    int yfb = nfy - 1;
    int ncx = uc.rows();
    int ncy = uc.cols();
    int xca = 0;
    int xcb = ncx - 1;
    int yca = 0;
    int ycb = ncy - 1;
    blitz::Range rxc(xca+1,xcb-1);
    blitz::Range ryc(yca+1,ycb-1);
    blitz::Range rxf(xfa,xfb,2);
    blitz::Range ryf(yfa,yfb,2);

    blitz::Range rx1(xfa+1,xfb-3,2);
    blitz::Range rx2(xfa+2,xfb-2,2);
    blitz::Range rx3(xfa+3,xfb-1,2);
    blitz::Range ry1(yfa+1,yfb-3,2);
    blitz::Range ry2(yfa+2,yfb-2,2);
    blitz::Range ry3(yfa+3,yfb-1,2);

    uc = uf(rxf,ryf);
    uc(rxc,ryc) = 0.125 * (uf(rx1,ry2) + uf(rx3,ry2) + \
                           uf(rx2,ry3) + uf(rx2,ry1) + \
                           4.0 * uf(rx2,ry2));
}

/**
 * Restrict fine grid uf to coarse grid uc by full-weight method.
 * uc has ncx x ncy points.
 * uf has nfx x nfy = (2*ncx-1) x (2*ncy-1) points.
 * the layout looks like:
 *
 * *....|....*....|....*....|....*....|....*....|....*  2*(ncy-1)
 * |    |    |    |    |    |    |    |    |    |    |
 * |....@....@....@....|....|....|....|....|....|....|  
 * |    |    |    |    |    |    |    |    |    |    |
 * *....@....*....@....*....|....*....|....*....|....*
 * |    |    |    |    |    |    |    |    |    |    |
 * |....@....@....@....|....|....|....|....|....|....|  
 * |    |    |    |    |    |    |    |    |    |    |
 * *....|....*....|....*....|....*....|....*....|....*
 * |    |    |    |    |    |    |    |    |    |    |
 * |....|....|....|....|....|....|....|....|....|....|  
 * |    |    |    |    |    |    |    |    |    |    |
 * *....|....*....|....*....|....*....|....*....|....*  4
 * |    |    |    |    |    |    |    |    |    |    |
 * |....|....|....|....|....|....|....|....|....|....|  3
 * |    |    |    |    |    |    |    |    |    |    |
 * *....|....*....|....*....|....*....|....*....|....*  2
 * |    |    |    |    |    |    |    |    |    |    |
 * |....|....|....|....|....|....|....|....|....|....|  1
 * |    |    |    |    |    |    |    |    |    |    |
 * *....|....*....|....*....|....*....|....*....|....*  0
 * 0    1    2    3    4                        2*(ncx-1)
 *
 */
void rstrct_fw_2d(const blitz::Array<double,2> uf,blitz::Array<double,2> uc){
    int nfx = uf.rows();
    int nfy = uf.cols();
    int xfa = 0;
    int xfb = nfx - 1;
    int yfa = 0;
    int yfb = nfy - 1;
    int ncx = uc.rows();
    int ncy = uc.cols();
    int xca = 0;
    int xcb = ncx - 1;
    int yca = 0;
    int ycb = ncy - 1;
    blitz::Range rxc(xca+1,xcb-1);
    blitz::Range ryc(yca+1,ycb-1);
    blitz::Range rxf(xfa,xfb,2);
    blitz::Range ryf(yfa,yfb,2);

    // require nfx > 4 and nfy > 4
    blitz::Range rx1(xfa+1,xfb-3,2);
    blitz::Range rx2(xfa+2,xfb-2,2);
    blitz::Range rx3(xfa+3,xfb-1,2);
    blitz::Range ry1(yfa+1,yfb-3,2);
    blitz::Range ry2(yfa+2,yfb-2,2);
    blitz::Range ry3(yfa+3,yfb-1,2);

    uc = uf(rxf,ryf);
    uc(rxc,ryc) = 0.0625 * (uf(rx1,ry1) + uf(rx3,ry3) + \
                            uf(rx1,ry3) + uf(rx3,ry1) + \
                            2.0 * uf(rx2,ry1) + 2.0 * uf(rx2,ry3) + \
                            2.0 * uf(rx1,ry2) + 2.0 * uf(rx3,ry2) + \
                            4.0 * uf(rx2,ry2));
}

void interp_trilinear_3d(const blitz::Array<double,3> uc,blitz::Array<double,3> uf){
    int nfx = uf.rows();
    int nfy = uf.cols();
    int nfz = uf.depth();
    int xfa = 0;
    int xfb = nfx - 1;
    int yfa = 0;
    int yfb = nfy - 1;
    int zfa = 0;
    int zfb = nfz - 1;
    int ncx = uc.rows();
    int xca = 0;
    int xcb = ncx - 1;
    blitz::Range all = blitz::Range::all();
    blitz::Range rx(xfa,xfb,2);
    blitz::Range rz(zfa,zfb,2);

    for(int i=xca;i<=xcb;i++)
        interp_bilinear_2d(uc(i,all,all),uf(2*i,all,all));
    for(int i=yfa;i<=yfb;i++)
        interp_bilinear_2d(uf(rx,i,rz),uf(all,i,all));
}

void interp_tricubic_3d(const blitz::Array<double,3> uc,blitz::Array<double,3> uf){
    int nfx = uf.rows();
    int nfy = uf.cols();
    int nfz = uf.depth();
    int xfa = 0;
    int xfb = nfx - 1;
    int yfa = 0;
    int yfb = nfy - 1;
    int zfa = 0;
    int zfb = nfz - 1;
    int ncx = uc.rows();
    int xca = 0;
    int xcb = ncx - 1;
    blitz::Range all = blitz::Range::all();
    blitz::Range rx(xfa,xfb,2);
    blitz::Range rz(zfa,zfb,2);

    for(int i=xca;i<=xcb;i++)
        interp_bicubic_2d(uc(i,all,all),uf(2*i,all,all));
    for(int i=yfa;i<=yfb;i++)
        interp_bicubic_2d(uf(rx,i,rz),uf(all,i,all));
}

void rstrct_injection_3d(const blitz::Array<double,3> uf,blitz::Array<double,3> uc){
    int nfx = uf.rows();
    int nfy = uf.cols();
    int nfz = uf.depth();
    int xfa = 0;
    int xfb = nfx - 1;
    int yfa = 0;
    int yfb = nfy - 1;
    int zfa = 0;
    int zfb = nfz - 1;
    blitz::Range rx(xfa,xfb,2);
    blitz::Range ry(yfa,yfb,2);
    blitz::Range rz(zfa,zfb,2);

    uc = uf(rx,ry,rz);
}

void rstrct_hw_3d(const blitz::Array<double,3> uf,blitz::Array<double,3> uc){
    int nfx = uf.rows();
    int nfy = uf.cols();
    int nfz = uf.depth();
    int xfa = 0;
    int xfb = nfx - 1;
    int yfa = 0;
    int yfb = nfy - 1;
    int zfa = 0;
    int zfb = nfz - 1;
    int ncx = uc.rows();
    int ncy = uc.cols();
    int ncz = uc.depth();
    int xca = 0;
    int xcb = ncx - 1;
    int yca = 0;
    int ycb = ncy - 1;
    int zca = 0;
    int zcb = ncz - 1;
    blitz::Range rxc(xca+1,xcb-1);
    blitz::Range ryc(yca+1,ycb-1);
    blitz::Range rzc(zca+1,zcb-1);
    blitz::Range rxf(xfa,xfb,2);
    blitz::Range ryf(yfa,yfb,2);
    blitz::Range rzf(zfa,zfb,2);

    blitz::Range rx1(xfa+1,xfb-3,2);
    blitz::Range rx2(xfa+2,xfb-2,2);
    blitz::Range rx3(xfa+3,xfb-1,2);
    blitz::Range ry1(yfa+1,yfb-3,2);
    blitz::Range ry2(yfa+2,yfb-2,2);
    blitz::Range ry3(yfa+3,yfb-1,2);
    blitz::Range rz1(zfa+1,zfb-3,2);
    blitz::Range rz2(zfa+2,zfb-2,2);
    blitz::Range rz3(zfa+3,zfb-1,2);

    uc = uf(rxf,ryf,rzf);
    uc(rxc,ryc,rzc) = (uf(rx1,ry1,rz2) + uf(rx3,ry1,rz2) + \
                       uf(rx1,ry3,rz2) + uf(rx3,ry3,rz2) + \
                       uf(rx1,ry2,rz1) + uf(rx3,ry2,rz1) + \
                       uf(rx1,ry2,rz3) + uf(rx3,ry2,rz3) + \
                       uf(rx2,ry1,rz1) + uf(rx2,ry3,rz1) + \
                       uf(rx2,ry1,rz3) + uf(rx2,ry3,rz3) + \
                       2.0 * uf(rx1,ry2,rz2) + 2.0 * uf(rx3,ry2,rz2) + \
                       2.0 * uf(rx2,ry1,rz2) + 2.0 * uf(rx2,ry3,rz2) + \
                       2.0 * uf(rx2,ry2,rz1) + 2.0 * uf(rx2,ry2,rz3) + \
                       8.0 * uf(rx2,ry2,rz2)) / 32.0;
}

void rstrct_fw_3d(const blitz::Array<double,3> uf,blitz::Array<double,3> uc){
    int nfx = uf.rows();
    int nfy = uf.cols();
    int nfz = uf.depth();
    int xfa = 0;
    int xfb = nfx - 1;
    int yfa = 0;
    int yfb = nfy - 1;
    int zfa = 0;
    int zfb = nfz - 1;
    int ncx = uc.rows();
    int ncy = uc.cols();
    int ncz = uc.depth();
    int xca = 0;
    int xcb = ncx - 1;
    int yca = 0;
    int ycb = ncy - 1;
    int zca = 0;
    int zcb = ncz - 1;
    blitz::Range rxc(xca+1,xcb-1);
    blitz::Range ryc(yca+1,ycb-1);
    blitz::Range rzc(zca+1,zcb-1);
    blitz::Range rxf(xfa,xfb,2);
    blitz::Range ryf(yfa,yfb,2);
    blitz::Range rzf(zfa,zfb,2);

    blitz::Range rx1(xfa+1,xfb-3,2);
    blitz::Range rx2(xfa+2,xfb-2,2);
    blitz::Range rx3(xfa+3,xfb-1,2);
    blitz::Range ry1(yfa+1,yfb-3,2);
    blitz::Range ry2(yfa+2,yfb-2,2);
    blitz::Range ry3(yfa+3,yfb-1,2);
    blitz::Range rz1(zfa+1,zfb-3,2);
    blitz::Range rz2(zfa+2,zfb-2,2);
    blitz::Range rz3(zfa+3,zfb-1,2);

    uc = uf(rxf,ryf,rzf);
    uc(rxc,ryc,rzc) = (uf(rx1,ry1,rz1) + uf(rx1,ry1,rz3) + \
                       uf(rx1,ry3,rz1) + uf(rx1,ry3,rz3) + \
                       uf(rx3,ry1,rz1) + uf(rx3,ry1,rz3) + \
                       uf(rx3,ry3,rz1) + uf(rx3,ry3,rz3) + \
                       2.0 * uf(rx1,ry1,rz2) + 2.0 * uf(rx3,ry1,rz2) + \
                       2.0 * uf(rx1,ry3,rz2) + 2.0 * uf(rx3,ry3,rz2) + \
                       2.0 * uf(rx1,ry2,rz1) + 2.0 * uf(rx3,ry2,rz1) + \
                       2.0 * uf(rx1,ry2,rz3) + 2.0 * uf(rx3,ry2,rz3) + \
                       2.0 * uf(rx2,ry1,rz1) + 2.0 * uf(rx2,ry3,rz1) + \
                       2.0 * uf(rx2,ry1,rz3) + 2.0 * uf(rx2,ry3,rz3) + \
                       4.0 * uf(rx1,ry2,rz2) + 4.0 * uf(rx3,ry2,rz2) + \
                       4.0 * uf(rx2,ry1,rz2) + 4.0 * uf(rx2,ry3,rz2) + \
                       4.0 * uf(rx2,ry2,rz1) + 4.0 * uf(rx2,ry2,rz3) + \
                       8.0 * uf(rx2,ry2,rz2)) / 32.0;
}

/**
 * Transfer fft (scft) Grid data to multigrid data by copying source data.
 * The 1d fft data has Lx points, all points are interior points.
 * the layout looks like:
 * |....*....|....*....|....*....|....*....|....*....|
 *      0         1         2                  Lx-1
 * The 1d multigrid data has Lx+1 points, they can be obtained
 * by copying the fft data and add an extra boundary point 
 * The layout looks like:
 *      #.........#.........#.........#.........#.........@
 *      0         1         2                  Lx-1       Lx = 0 
 *
 */
void fft2mg_copy_1d(const Grid &fft,blitz::Array<double,1> mg){
    blitz::Range all = blitz::Range::all();
    blitz::Array<double,3> fft_data(fft.data());
    blitz::Array<double,1> fft_data_1d(fft_data(all,0,0));
    int Lx = fft.Lx();
    int xa = 0;
    int xb = Lx - 1;
    blitz::Range rx(xa,xb);

    mg(rx) = fft_data_1d;

    // Periodic boundary conditions
    mg(xb+1) = mg(xa);
}

void mg2fft_copy_1d(const blitz::Array<double,1> mg,Grid &fft){
    blitz::Range all = blitz::Range::all();
    blitz::Array<double,3> fft_data(fft.data());
    blitz::Array<double,1> fft_data_1d(fft_data(all,0,0));
    int Lx = fft.Lx();
    int xa = 0;
    int xb = Lx - 1;
    blitz::Range rx(xa,xb);

    fft_data_1d = mg(rx);
}

void fft2mg_copy_1d(const blitz::Array<double,1> fft,blitz::Array<double,1> mg){
    blitz::Range all = blitz::Range::all();
    int Lx = fft.rows();
    int xa = 0;
    int xb = Lx - 1;
    blitz::Range rx(xa,xb);

    mg(rx) = fft;

    // Periodic boundary conditions
    mg(xb+1) = mg(xa);
}

void mg2fft_copy_1d(const blitz::Array<double,1> mg,blitz::Array<double,1> fft){
    blitz::Range all = blitz::Range::all();
    int Lx = fft.rows();
    int xa = 0;
    int xb = Lx - 1;
    blitz::Range rx(xa,xb);

    fft = mg(rx);
}

/**
 * Transfer fft (scft) Grid data to multigrid data.
 * The 1d fft data has Lx points, all points are interior points.
 * the layout looks like:
 * |....*....|....*....|....*....|....*....|....*....|
 *      0         1         2                  Lx-1
 * The 1d multigrid data has 2*Lx+1 points, they can be obtained
 * by interpolation of fft data and adopting the two extra boundary points.
 * The layout looks like:
 * |....|....|....|....|....|....|....|....|....|....|
 * 0    1    2    3    4                     2*Lx-1  2*Lx 
 *
 */
void fft2mg_interp_1d(const Grid &fft,blitz::Array<double,1> mg,int interp_kind){
    blitz::Range all = blitz::Range::all();
    blitz::Array<double,3> fft_data(fft.data());
    blitz::Array<double,1> fft_data_1d(fft_data(all,0,0));
    int Lx = fft.Lx();
    int xa = 1;
    int xb = 2 * Lx - 1;
    blitz::Range rx(xa,xb);

    switch(interp_kind){
        case 0:
            interp_linear_1d(fft_data_1d,mg(rx));
            break;
        case 1:
            interp_cubic_1d(fft_data_1d,mg(rx));
            break;
        default:
            interp_linear_1d(fft_data_1d,mg(rx));
            break;
    }

    // Periodic boundary conditions
    mg(xa-1) = 0.5 * (mg(xa) + mg(xb));
    mg(xb+1) = mg(xa-1);
}

void mg2fft_interp_1d(const blitz::Array<double,1> mg,Grid &fft){
    blitz::Range all = blitz::Range::all();
    blitz::Array<double,3> fft_data(fft.data());
    blitz::Array<double,1> fft_data_1d(fft_data(all,0,0));
    int Lx = fft.Lx();
    int xa = 1;
    int xb = 2 * Lx - 1;
    blitz::Range rx(xa,xb);

    rstrct_injection_1d(mg(rx),fft_data_1d);
    //rstrct_weight_1d(mg(rx),fft_data_1d);
}

void fft2mg_interp_1d(const blitz::Array<double,1> fft,blitz::Array<double,1> mg,int interp_kind){
    int Lx = fft.rows();
    int xa = 1;
    int xb = 2 * Lx - 1;
    blitz::Range rx(xa,xb);

    switch(interp_kind){
        case 0:
            interp_linear_1d(fft,mg(rx));
            break;
        case 1:
            interp_cubic_1d(fft,mg(rx));
            break;
        default:
            interp_linear_1d(fft,mg(rx));
            break;
    }

    // Periodic boundary conditions
    mg(xa-1) = 0.5 * (mg(xa) + mg(xb));
    mg(xb+1) = mg(xa-1);
}

void mg2fft_interp_1d(const blitz::Array<double,1> mg,blitz::Array<double,1> fft){
    int Lx = fft.rows();
    int xa = 1;
    int xb = 2 * Lx - 1;
    blitz::Range rx(xa,xb);

    rstrct_injection_1d(mg(rx),fft);
    //rstrct_weight_1d(mg(rx),fft);
}

/**
 * Transfer fft (scft) Grid data to multigrid data.
 * The 2d fft data has Lx x Ly points, all points are interior points.
 * the layout looks like:
 * |.........|.........|.........|.........|.........|
 * |         |         |         |         |         |
 * |    *    |    *    |    *    |    *    |    *    |  Ly-1
 * |         |         |         |         |         |
 * |.........|.........|.........|.........|.........|
 * |         |         |         |         |         |
 * |    *    |    *    |    *    |    *    |    *    |
 * |         |         |         |         |         |
 * |.........|.........|.........|.........|.........|
 * |         |         |         |         |         |
 * |    *    |    *    |    *    |    *    |    *    |  2
 * |         |         |         |         |         |
 * |.........|.........|.........|.........|.........|
 * |         |         |         |         |         |
 * |    *    |    *    |    *    |    *    |    *    |  1
 * |         |         |         |         |         |
 * |.........|.........|.........|.........|.........|
 * |         |         |         |         |         |
 * |    *    |    *    |    *    |    *    |    *    |  0
 * |         |         |         |         |         |
 * |.........|.........|.........|.........|.........|
 *      0         1         2                  Lx-1
 * The 2d multigrid data has (Lx+1) x (Ly+1) points,
 * they can be obtained by copying fft data 
 * and adding extra boundary points.
 * The layout looks like:
 * |....|....|....|....|....|....|....|....|....|....|....|....| 
 * |    |    |    |    |    |    |    |    |    |    |    |    |
 * |....&....|....@....|....@....|....@....|....@....|....&....| Ly = 0
 * |    |    |    |    |    |    |    |    |    |    |    |    |
 * |....|....|....|....|....|....|....|....|....|....|....|....| 
 * |    |    |    |    |    |    |    |    |    |    |    |    |
 * |....#....|....#....|....#....|....#....|....#....|....@....| Ly-1
 * |    |    |    |    |    |    |    |    |    |    |    |    |
 * |....|....|....|....|....|....|....|....|....|....|....|....|
 * |    |    |    |    |    |    |    |    |    |    |    |    |
 * |....#....|....#....|....#....|....#....|....#....|....@....| 
 * |    |    |    |    |    |    |    |    |    |    |    |    |
 * |....|....|....|....|....|....|....|....|....|....|....|....|
 * |    |    |    |    |    |    |    |    |    |    |    |    |
 * |....#....|....#....|....#....|....#....|....#....|....@....| 2
 * |    |    |    |    |    |    |    |    |    |    |    |    |
 * |....|....|....|....|....|....|....|....|....|....|....|....| 
 * |    |    |    |    |    |    |    |    |    |    |    |    |
 * |....#....|....#....|....#....|....#....|....#....|....@....| 1
 * |    |    |    |    |    |    |    |    |    |    |    |    |
 * |....|....|....#....|....|....|....|....|....|....|....|....| 
 * |    |    |    |    |    |    |    |    |    |    |    |    |
 * |....#....|....#....|....#....|....#....|....#....|....&....| 0
 * |    |    |    |    |    |    |    |    |    |    |    |    |
 * |....|....|....|....|....|....|....|....|....|....|....|....| 
 *      0         1         3                  Lx-1       Lx = 0
 *
 */
void fft2mg_copy_2d(const Grid &fft,blitz::Array<double,2> mg){
    blitz::Range all = blitz::Range::all();
    blitz::Array<double,3> fft_data(fft.data());
    blitz::Array<double,2> fft_data_2d(fft_data(all,all,0));
    int Lx = fft.Lx();
    int Ly = fft.Ly();
    int xa = 0;
    int xb = Lx -1;
    int ya = 0;
    int yb = Ly -1;
    blitz::Range rx(xa,xb);
    blitz::Range ry(ya,yb);

    mg(rx,ry) = fft_data_2d;

    // Periodic boundary conditions
    mg(xb+1,ry) = mg(xa,ry);
    mg(all,yb+1) = mg(all,ya);
}

void mg2fft_copy_2d(const blitz::Array<double,2> mg,Grid &fft){
    blitz::Range all = blitz::Range::all();
    blitz::Array<double,3> fft_data(fft.data());
    blitz::Array<double,2> fft_data_2d(fft_data(all,all,0));
    int Lx = fft.Lx();
    int Ly = fft.Ly();
    int xa = 0;
    int xb = Lx -1;
    int ya = 0;
    int yb = Ly -1;
    blitz::Range rx(xa,xb);
    blitz::Range ry(ya,yb);

    fft_data_2d = mg(rx,ry);
}

void fft2mg_copy_2d(const blitz::Array<double,2> fft,blitz::Array<double,2> mg){
    blitz::Range all = blitz::Range::all();
    int Lx = fft.rows();
    int Ly = fft.cols();
    int xa = 0;
    int xb = Lx -1;
    int ya = 0;
    int yb = Ly -1;
    blitz::Range rx(xa,xb);
    blitz::Range ry(ya,yb);

    mg(rx,ry) = fft;

    // Periodic boundary conditions
    mg(xb+1,ry) = mg(xa,ry);
    mg(all,yb+1) = mg(all,ya);
}

void mg2fft_copy_2d(const blitz::Array<double,2> mg,blitz::Array<double,2> fft){
    blitz::Range all = blitz::Range::all();
    int Lx = fft.rows();
    int Ly = fft.cols();
    int xa = 0;
    int xb = Lx -1;
    int ya = 0;
    int yb = Ly -1;
    blitz::Range rx(xa,xb);
    blitz::Range ry(ya,yb);

    fft = mg(rx,ry);
}

/**
 * Transfer fft (scft) Grid data to multigrid data.
 * The 2d fft data has Lx x Ly points, all points are interior points.
 * the layout looks like:
 * |.........|.........|.........|.........|.........|
 * |         |         |         |         |         |
 * |    *    |    *    |    *    |    *    |    *    |  Ly-1
 * |         |         |         |         |         |
 * |.........|.........|.........|.........|.........|
 * |         |         |         |         |         |
 * |    *    |    *    |    *    |    *    |    *    |
 * |         |         |         |         |         |
 * |.........|.........|.........|.........|.........|
 * |         |         |         |         |         |
 * |    *    |    *    |    *    |    *    |    *    |  2
 * |         |         |         |         |         |
 * |.........|.........|.........|.........|.........|
 * |         |         |         |         |         |
 * |    *    |    *    |    *    |    *    |    *    |  1
 * |         |         |         |         |         |
 * |.........|.........|.........|.........|.........|
 * |         |         |         |         |         |
 * |    *    |    *    |    *    |    *    |    *    |  0
 * |         |         |         |         |         |
 * |.........|.........|.........|.........|.........|
 *      0         1         2                  Lx-1
 * The 2d multigrid data has (2*Lx+1) x (2*Ly+1) points,
 * they can be obtained by interpolation of fft data 
 * and adopting the two extra boundary points.
 * The layout looks like:
 * |....|....|....|....|....|....|....|....|....|....|  2*Ly
 * |    |    |    |    |    |    |    |    |    |    |
 * |....|....|....|....|....|....|....|....|....|....|  2*Ly-1
 * |    |    |    |    |    |    |    |    |    |    |
 * |....|....|....|....|....|....|....|....|....|....|
 * |    |    |    |    |    |    |    |    |    |    |
 * |....|....|....|....|....|....|....|....|....|....|  
 * |    |    |    |    |    |    |    |    |    |    |
 * |....|....|....|....|....|....|....|....|....|....|
 * |    |    |    |    |    |    |    |    |    |    |
 * |....|....|....|....|....|....|....|....|....|....|  
 * |    |    |    |    |    |    |    |    |    |    |
 * |....|....|....|....|....|....|....|....|....|....|  4
 * |    |    |    |    |    |    |    |    |    |    |
 * |....|....|....|....|....|....|....|....|....|....|  3
 * |    |    |    |    |    |    |    |    |    |    |
 * |....|....|....|....|....|....|....|....|....|....|  2
 * |    |    |    |    |    |    |    |    |    |    |
 * |....|....|....|....|....|....|....|....|....|....|  1
 * |    |    |    |    |    |    |    |    |    |    |
 * |....|....|....|....|....|....|....|....|....|....|  0
 * 0    1    2    3    4                     2*Lx-1  2*Lx 
 *
 */
void fft2mg_interp_2d(const Grid &fft,blitz::Array<double,2> mg,int interp_kind){
    blitz::Range all = blitz::Range::all();
    blitz::Array<double,3> fft_data(fft.data());
    blitz::Array<double,2> fft_data_2d(fft_data(all,all,0));
    int Lx = fft.Lx();
    int Ly = fft.Ly();
    int xa = 1;
    int xb = 2 * Lx -1;
    int ya = 1;
    int yb = 2 * Ly -1;
    blitz::Range rx(xa,xb);
    blitz::Range ry(ya,yb);

    switch(interp_kind){
        case 0:
            interp_bilinear_2d(fft_data_2d,mg(rx,ry));
            break;
        case 1:
            interp_bicubic_2d(fft_data_2d,mg(rx,ry));
            break;
        default:
            interp_bilinear_2d(fft_data_2d,mg(rx,ry));
            break;
    }

    // Periodic boundary conditions
    // edge boundaries
    mg(xa-1,ry) = 0.5 * (mg(xa,ry) + mg(xb,ry));
    mg(xb+1,ry) = mg(xa-1,ry);
    mg(rx,ya-1) = 0.5 * (mg(rx,ya) + mg(rx,yb));
    mg(rx,yb+1) = mg(rx,ya-1);
    // corner boundaries
    mg(xa-1,ya-1) = 0.25 * (mg(xa,ya-1) + mg(xb,ya-1) + \
                            mg(xa-1,ya) + mg(xa-1,yb));
    mg(xa-1,yb+1) = mg(xa-1,ya-1);
    mg(xb+1,ya-1) = mg(xa-1,ya-1);
    mg(xb+1,yb+1) = mg(xa-1,ya-1);

    /*
    // Dirichlet boundary conditions
    // edge boundaries
    mg(xa-1,ry) = 0.0;
    mg(xb+1,ry) = 0.0;
//    mg(rx,ya-1) = 0.0;
//    mg(rx,yb+1) = 0.0;
    // corner boundaries
    mg(xa-1,ya-1) = 0.0;
    mg(xa-1,yb+1) = 0.0;
    mg(xb+1,ya-1) = 0.0;
    mg(xb+1,yb+1) = 0.0;
    */
}

void mg2fft_interp_2d(const blitz::Array<double,2> mg,Grid &fft){
    blitz::Range all = blitz::Range::all();
    blitz::Array<double,3> fft_data(fft.data());
    blitz::Array<double,2> fft_data_2d(fft_data(all,all,0));
    int Lx = fft.Lx();
    int Ly = fft.Ly();
    int xa = 1;
    int xb = 2 * Lx -1;
    int ya = 1;
    int yb = 2 * Ly -1;
    blitz::Range rx(xa,xb);
    blitz::Range ry(ya,yb);

    rstrct_injection_2d(mg(rx,ry),fft_data_2d);
    //rstrct_hw_2d(mg(rx,ry),fft_data_2d);
    //rstrct_fw_2d(mg(rx,ry),fft_data_2d);
}

void fft2mg_interp_2d(const blitz::Array<double,2> fft,blitz::Array<double,2> mg,int interp_kind){
    int Lx = fft.rows();
    int Ly = fft.cols();
    int xa = 1;
    int xb = 2 * Lx -1;
    int ya = 1;
    int yb = 2 * Ly -1;
    blitz::Range rx(xa,xb);
    blitz::Range ry(ya,yb);

    switch(interp_kind){
        case 0:
            interp_bilinear_2d(fft,mg(rx,ry));
            break;
        case 1:
            interp_bicubic_2d(fft,mg(rx,ry));
            break;
        default:
            interp_bilinear_2d(fft,mg(rx,ry));
            break;
    }

    // Periodic boundary conditions
    // edge boundaries
    mg(xa-1,ry) = 0.5 * (mg(xa,ry) + mg(xb,ry));
    mg(xb+1,ry) = mg(xa-1,ry);
    mg(rx,ya-1) = 0.5 * (mg(rx,ya) + mg(rx,yb));
    mg(rx,yb+1) = mg(rx,ya-1);
    // corner boundaries
    mg(xa-1,ya-1) = 0.25 * (mg(xa,ya-1) + mg(xb,ya-1) + \
                            mg(xa-1,ya) + mg(xa-1,yb));
    mg(xa-1,yb+1) = mg(xa-1,ya-1);
    mg(xb+1,ya-1) = mg(xa-1,ya-1);
    mg(xb+1,yb+1) = mg(xa-1,ya-1);
}

void mg2fft_interp_2d(const blitz::Array<double,2> mg,blitz::Array<double,2> fft){
    int Lx = fft.rows();
    int Ly = fft.cols();
    int xa = 1;
    int xb = 2 * Lx -1;
    int ya = 1;
    int yb = 2 * Ly -1;
    blitz::Range rx(xa,xb);
    blitz::Range ry(ya,yb);

    rstrct_injection_2d(mg(rx,ry),fft);
    //rstrct_hw_2d(mg(rx,ry),fft);
    //rstrct_fw_2d(mg(rx,ry),fft);
}

void fft2mg_copy_3d(const Grid &fft,blitz::Array<double,3> mg){
    blitz::Array<double,3> fft_data_3d(fft.data());
    int Lx = fft.Lx();
    int Ly = fft.Ly();
    int Lz = fft.Lz();
    int xa = 0;
    int xb = Lx -1;
    int ya = 0;
    int yb = Ly -1;
    int za = 0;
    int zb = Lz -1;
    blitz::Range all = blitz::Range::all();
    blitz::Range rx(xa,xb);
    blitz::Range ry(ya,yb);
    blitz::Range rz(za,zb);

    mg(rx,ry,rz) = fft_data_3d;

    // Periodic boundary conditions
    mg(xb+1,ry,rz) = mg(xa,ry,rz);
    mg(all,yb+1,rz) = mg(all,ya,rz);
    mg(all,all,zb+1) = mg(all,all,za);
}

void mg2fft_copy_3d(const blitz::Array<double,3> mg,Grid &fft){
    blitz::Array<double,3> fft_data_3d(fft.data());
    int Lx = fft.Lx();
    int Ly = fft.Ly();
    int Lz = fft.Lz();
    int xa = 0;
    int xb = Lx -1;
    int ya = 0;
    int yb = Ly -1;
    int za = 0;
    int zb = Lz -1;
    blitz::Range rx(xa,xb);
    blitz::Range ry(ya,yb);
    blitz::Range rz(za,zb);

    fft_data_3d = mg(rx,ry,rz);
}

void fft2mg_copy_3d(const blitz::Array<double,3> fft,blitz::Array<double,3> mg){
    int Lx = fft.rows();
    int Ly = fft.cols();
    int Lz = fft.depth();
    int xa = 0;
    int xb = Lx -1;
    int ya = 0;
    int yb = Ly -1;
    int za = 0;
    int zb = Lz -1;
    blitz::Range all = blitz::Range::all();
    blitz::Range rx(xa,xb);
    blitz::Range ry(ya,yb);
    blitz::Range rz(za,zb);

    mg(rx,ry,rz) = fft;

    // Periodic boundary conditions
    mg(xb+1,ry,rz) = mg(xa,ry,rz);
    mg(all,yb+1,rz) = mg(all,ya,rz);
    mg(all,all,zb+1) = mg(all,all,za);
}

void mg2fft_copy_3d(const blitz::Array<double,3> mg,blitz::Array<double,3> fft){
    int Lx = fft.rows();
    int Ly = fft.cols();
    int Lz = fft.depth();
    int xa = 0;
    int xb = Lx -1;
    int ya = 0;
    int yb = Ly -1;
    int za = 0;
    int zb = Lz -1;
    blitz::Range rx(xa,xb);
    blitz::Range ry(ya,yb);
    blitz::Range rz(za,zb);

    fft = mg(rx,ry,rz);
}

void fft2mg_interp_3d(const Grid &fft,blitz::Array<double,3> mg,int interp_kind){
    blitz::Array<double,3> fft_data_3d(fft.data());
    int Lx = fft.Lx();
    int Ly = fft.Ly();
    int Lz = fft.Lz();
    int xa = 1;
    int xb = 2 * Lx -1;
    int ya = 1;
    int yb = 2 * Ly -1;
    int za = 1;
    int zb = 2 * Lz -1;
    blitz::Range rx(xa,xb);
    blitz::Range ry(ya,yb);
    blitz::Range rz(za,zb);

    switch(interp_kind){
        case 0:
            interp_trilinear_3d(fft_data_3d,mg(rx,ry,rz));
            break;
        case 1:
            interp_tricubic_3d(fft_data_3d,mg(rx,ry,rz));
            break;
        default:
            interp_trilinear_3d(fft_data_3d,mg(rx,ry,rz));
            break;
    }

    // Periodic boundary conditions
    // plane boundaries
    // (y,z) plane
    mg(xa-1,ry,rz) = 0.5 * (mg(xa,ry,rz) + mg(xb,ry,rz));
    mg(xb+1,ry,rz) = mg(xa-1,ry,rz);
    // (x,z) plane
    mg(rx,ya-1,rz) = 0.5 * (mg(rx,ya,rz) + mg(rx,yb,rz));
    mg(rx,yb+1,rz) = mg(rx,ya-1,rz);
    // (x,y) plane
    mg(rx,ry,za-1) = 0.5 * (mg(rx,ry,za) + mg(rx,ry,zb));
    mg(rx,ry,zb+1) = mg(rx,ry,za-1);
    // edge boundaries
    // z edge
    mg(xa-1,ya-1,rz) = 0.25 * (mg(xa-1,ya,rz) + mg(xa-1,yb,rz) + \
                               mg(xa,ya-1,rz) + mg(xb,ya-1,rz));
    mg(xa-1,yb+1,rz) = mg(xa-1,ya-1,rz);
    mg(xb+1,ya-1,rz) = mg(xa-1,ya-1,rz);
    mg(xb+1,yb+1,rz) = mg(xa-1,ya-1,rz);
    // y edge
    mg(xa-1,ry,za-1) = 0.25 * (mg(xa-1,ry,za) + mg(xa-1,ry,zb) + \
                               mg(xa,ry,za-1) + mg(xb,ry,za-1));
    mg(xa-1,ry,zb+1) = mg(xa-1,ry,za-1);
    mg(xb+1,ry,za-1) = mg(xa-1,ry,za-1);
    mg(xb+1,ry,zb+1) = mg(xa-1,ry,za-1);
    // x edge
    mg(rx,ya-1,za-1) = 0.25 * (mg(rx,ya-1,za) + mg(rx,ya-1,zb) + \
                               mg(rx,ya,za-1) + mg(rx,yb,za-1));
    mg(rx,ya-1,zb+1) = mg(rx,ya-1,za-1); 
    mg(rx,yb+1,za-1) = mg(rx,ya-1,za-1); 
    mg(rx,yb+1,zb+1) = mg(rx,ya-1,za-1); 
    // corners
    mg(xa-1,ya-1,za-1) = (mg(xa,ya-1,za-1) + mg(xb,ya-1,za-1) + \
                          mg(xa-1,ya,za-1) + mg(xa-1,yb,za-1) + \
                          mg(xa-1,ya-1,za) + mg(xa-1,ya-1,zb)) / 6.0;
    mg(xb+1,ya-1,za-1) = mg(xa-1,ya-1,za-1);
    mg(xa-1,yb+1,za-1) = mg(xa-1,ya-1,za-1);
    mg(xb+1,yb+1,za-1) = mg(xa-1,ya-1,za-1);

    mg(xa-1,ya-1,zb+1) = mg(xa-1,ya-1,za-1);
    mg(xb+1,ya-1,zb+1) = mg(xa-1,ya-1,za-1);
    mg(xa-1,yb+1,zb+1) = mg(xa-1,ya-1,za-1);
    mg(xb+1,yb+1,zb+1) = mg(xa-1,ya-1,za-1);
}

void mg2fft_interp_3d(const blitz::Array<double,3> mg,Grid &fft){
    blitz::Array<double,3> fft_data_3d(fft.data());
    int Lx = fft.Lx();
    int Ly = fft.Ly();
    int Lz = fft.Lz();
    int xa = 1;
    int xb = 2 * Lx -1;
    int ya = 1;
    int yb = 2 * Ly -1;
    int za = 1;
    int zb = 2 * Lz -1;
    blitz::Range rx(xa,xb);
    blitz::Range ry(ya,yb);
    blitz::Range rz(za,zb);

    rstrct_injection_3d(mg(rx,ry,rz),fft_data_3d);
    //rstrct_hw_2d(mg(rx,ry),fft_data_3d);
    //rstrct_fw_2d(mg(rx,ry),fft_data_3d);
}

void fft2mg_interp_3d(const blitz::Array<double,3> fft,blitz::Array<double,3> mg,int interp_kind){
    int Lx = fft.rows();
    int Ly = fft.cols();
    int Lz = fft.depth();
    int xa = 1;
    int xb = 2 * Lx -1;
    int ya = 1;
    int yb = 2 * Ly -1;
    int za = 1;
    int zb = 2 * Lz -1;
    blitz::Range rx(xa,xb);
    blitz::Range ry(ya,yb);
    blitz::Range rz(za,zb);

    switch(interp_kind){
        case 0:
            interp_trilinear_3d(fft,mg(rx,ry,rz));
            break;
        case 1:
            interp_tricubic_3d(fft,mg(rx,ry,rz));
            break;
        default:
            interp_trilinear_3d(fft,mg(rx,ry,rz));
            break;
    }

    // Periodic boundary conditions
    // plane boundaries
    // (y,z) plane
    mg(xa-1,ry,rz) = 0.5 * (mg(xa,ry,rz) + mg(xb,ry,rz));
    mg(xb+1,ry,rz) = mg(xa-1,ry,rz);
    // (x,z) plane
    mg(rx,ya-1,rz) = 0.5 * (mg(rx,ya,rz) + mg(rx,yb,rz));
    mg(rx,yb+1,rz) = mg(rx,ya-1,rz);
    // (x,y) plane
    mg(rx,ry,za-1) = 0.5 * (mg(rx,ry,za) + mg(rx,ry,zb));
    mg(rx,ry,zb+1) = mg(rx,ry,za-1);
    // edge boundaries
    // z edge
    mg(xa-1,ya-1,rz) = 0.25 * (mg(xa-1,ya,rz) + mg(xa-1,yb,rz) + \
                               mg(xa,ya-1,rz) + mg(xb,ya-1,rz));
    mg(xa-1,yb+1,rz) = mg(xa-1,ya-1,rz);
    mg(xb+1,ya-1,rz) = mg(xa-1,ya-1,rz);
    mg(xb+1,yb+1,rz) = mg(xa-1,ya-1,rz);
    // y edge
    mg(xa-1,ry,za-1) = 0.25 * (mg(xa-1,ry,za) + mg(xa-1,ry,zb) + \
                               mg(xa,ry,za-1) + mg(xb,ry,za-1));
    mg(xa-1,ry,zb+1) = mg(xa-1,ry,za-1);
    mg(xb+1,ry,za-1) = mg(xa-1,ry,za-1);
    mg(xb+1,ry,zb+1) = mg(xa-1,ry,za-1);
    // x edge
    mg(rx,ya-1,za-1) = 0.25 * (mg(rx,ya-1,za) + mg(rx,ya-1,zb) + \
                               mg(rx,ya,za-1) + mg(rx,yb,za-1));
    mg(rx,ya-1,zb+1) = mg(rx,ya-1,za-1); 
    mg(rx,yb+1,za-1) = mg(rx,ya-1,za-1); 
    mg(rx,yb+1,zb+1) = mg(rx,ya-1,za-1); 
    // corners
    mg(xa-1,ya-1,za-1) = (mg(xa,ya-1,za-1) + mg(xb,ya-1,za-1) + \
                          mg(xa-1,ya,za-1) + mg(xa-1,yb,za-1) + \
                          mg(xa-1,ya-1,za) + mg(xa-1,ya-1,zb)) / 6.0;
    mg(xb+1,ya-1,za-1) = mg(xa-1,ya-1,za-1);
    mg(xa-1,yb+1,za-1) = mg(xa-1,ya-1,za-1);
    mg(xb+1,yb+1,za-1) = mg(xa-1,ya-1,za-1);

    mg(xa-1,ya-1,zb+1) = mg(xa-1,ya-1,za-1);
    mg(xb+1,ya-1,zb+1) = mg(xa-1,ya-1,za-1);
    mg(xa-1,yb+1,zb+1) = mg(xa-1,ya-1,za-1);
    mg(xb+1,yb+1,zb+1) = mg(xa-1,ya-1,za-1);
}

void mg2fft_interp_3d(const blitz::Array<double,3> mg,blitz::Array<double,3> fft){
    int Lx = fft.rows();
    int Ly = fft.cols();
    int Lz = fft.depth();
    int xa = 1;
    int xb = 2 * Lx -1;
    int ya = 1;
    int yb = 2 * Ly -1;
    int za = 1;
    int zb = 2 * Lz -1;
    blitz::Range rx(xa,xb);
    blitz::Range ry(ya,yb);
    blitz::Range rz(za,zb);

    rstrct_injection_3d(mg(rx,ry,rz),fft);
    //rstrct_hw_2d(mg(rx,ry),fft);
    //rstrct_fw_2d(mg(rx,ry),fft);
}

