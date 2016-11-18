/**
 * FieldE.cc
 * Created at 2012.4.24
 *
 * Implementation of FieldE.h.
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

#include "FieldE.h"

extern "C"{
double sig_mud2(MUD_FLOAT *x,MUD_FLOAT *y){
    MUD_FLOAT dx = gmudsig.dx;
    MUD_FLOAT dy = gmudsig.dy;
    MUD_FLOAT xa = gmudsig.xa;
    MUD_FLOAT yc = gmudsig.yc;
    int ny = gmudsig.ny;
    int i = (int)((*x - xa) / dx + 0.01);
    int j = (int)((*y - yc) / dy + 0.01);
    int index = ny * i + j;
    return gmudsig.xdata[index];
}

/**
 * For 2D Hexagonal unit cell + position-dependent dielectric constant, 
 * the Poisson Equation is
 *      (4/3)* [eps*(pxx + pxy + pyy) + (d(eps)/dx + 0.5*d(eps)/dy)*px +
 *      (d(eps)/dy + 0.5*d(eps)/dx)*py] = r(x,y)
 * Here,
 *      gmudsig.xdata = eps
 *      gmudsig.ydata = d(eps)/dx
 *      gmudsig.zdata = d(eps)/dy
 *
 */
void cof_mud2cr(MUD_FLOAT *x,MUD_FLOAT *y,MUD_FLOAT *cxx,MUD_FLOAT *cxy,MUD_FLOAT *cyy,MUD_FLOAT *cx,MUD_FLOAT *cy,MUD_FLOAT *ce){
    MUD_FLOAT dx = gmudsig.dx;
    MUD_FLOAT dy = gmudsig.dy;
    MUD_FLOAT xa = gmudsig.xa;
    MUD_FLOAT yc = gmudsig.yc;
    int ny = gmudsig.ny;
    int i = (int)((*x - xa) / dx + 0.01);
    int j = (int)((*y - yc) / dy + 0.01);
    int index = ny * i + j;
    *cxx = 4.0 * gmudsig.xdata[index] / 3.0;
    *cxy = *cxx;
    *cyy = *cxx;
    *cx = 4.0 * (gmudsig.ydata[index] + 0.5 * gmudsig.zdata[index]) / 3.0;
    *cy = 4.0 * (gmudsig.zdata[index] + 0.5 * gmudsig.ydata[index]) / 3.0;
    *ce = 0;
}

double sig_mud3(MUD_FLOAT *x,MUD_FLOAT *y,MUD_FLOAT *z){
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

/**
 * For 3D Hexagonal unit cell + position-dependent dielectric constant, 
 * the Poisson Equation is
 *      (4/3)*eps*(pxx + pxy + pyy) + pzz + 
 *      (4/3)*(d(eps)/dx + 0.5*d(eps)/dy)*px +
 *      (4/3)*(d(eps)/dy + 0.5*d(eps)/dx)*py +
 *      (d(eps)/dz)*pz + = r(x,y,z)
 * Here,
 *      gmudsig.xdata = eps
 *      gmudsig.ydata = d(eps)/dx
 *      gmudsig.zdata = d(eps)/dy
 *      gmudsig.lambda_data = d(eps)/dz
 *
 */
void cof_mud3cr_hex(MUD_FLOAT *x,MUD_FLOAT *y,MUD_FLOAT *z,MUD_FLOAT *cxx,MUD_FLOAT *cyy,MUD_FLOAT *czz,MUD_FLOAT *cx,MUD_FLOAT *cy,MUD_FLOAT *cz,MUD_FLOAT *ce){
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
    *cxx = 4.0 * gmudsig.xdata[index] / 3.0;
    *cyy = *cxx;
    *czz = gmudsig.xdata[index];
    *cx = 4.0 * (gmudsig.ydata[index] + 0.5 * gmudsig.zdata[index]) / 3.0;
    *cy = 4.0 * (gmudsig.zdata[index] + 0.5 * gmudsig.ydata[index]) / 3.0;
    *cz = gmudsig.lambda_data[index];
    *ce = 0;
}

void crs_mud3cr(MUD_FLOAT *x,MUD_FLOAT *y,MUD_FLOAT *z,MUD_FLOAT *c){
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
    *c = 4.0 * gmudsig.xdata[index] / 3.0;
}

/**
 * For Monoclinic unit cell + position-independent dielectric constant
 *      (4/3)*pxx + pyy + (4/3)*pzz + (4/3)*pxz = r(x,y,z)
 *
 */
void cof_mud3cr_mono_ceps(MUD_FLOAT *x,MUD_FLOAT *y,MUD_FLOAT *z,MUD_FLOAT *cxx,MUD_FLOAT *cyy,MUD_FLOAT *czz,MUD_FLOAT *cx,MUD_FLOAT *cy,MUD_FLOAT *cz,MUD_FLOAT *ce){
    *cxx = 4.0 / 3.0;
    *cyy = 1.0;
    *czz = 4.0 / 3.0;
    *cx = 0.0;
    *cy = 0.0;
    *cz = 0.0;
    *ce = 0.0;
}

/**
 * For 3D Monoclinic unit cell + position-dependent dielectric constant, 
 * the Poisson Equation is
 *      (4/3)*eps*(pxx + pxz + pyy) + pyy + 
 *      (4/3)*(d(eps)/dx + 0.5*d(eps)/dz)*px +
 *      (d(eps)/dy)*py + 
 *      (4/3)*(d(eps)/dz + 0.5*d(eps)/dx)*pz = r(x,y,z)
 * Here,
 *      gmudsig.xdata = eps
 *      gmudsig.ydata = d(eps)/dx
 *      gmudsig.zdata = d(eps)/dy
 *      gmudsig.lambda_data = d(eps)/dz
 *
 */
void cof_mud3cr_mono_veps(MUD_FLOAT *x,MUD_FLOAT *y,MUD_FLOAT *z,MUD_FLOAT *cxx,MUD_FLOAT *cyy,MUD_FLOAT *czz,MUD_FLOAT *cx,MUD_FLOAT *cy,MUD_FLOAT *cz,MUD_FLOAT *ce){
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
    *cxx = 4.0 * gmudsig.xdata[index] / 3.0;
    *cyy = gmudsig.xdata[index];
    *czz = *cxx;
    *cx = 4.0 * (gmudsig.ydata[index] + 0.5 * gmudsig.lambda_data[index]) / 3.0;
    *cy = gmudsig.zdata[index];
    *cz = 4.0 * (gmudsig.lambda_data[index] + 0.5 * gmudsig.ydata[index]) / 3.0;
    *ce = 0;
}

} // end of extern "C"

void dummy_crsxy(MUD_FLOAT *x,MUD_FLOAT *y,MUD_FLOAT *z,MUD_FLOAT *cxz);
void dummy_crsxz(MUD_FLOAT *x,MUD_FLOAT *y,MUD_FLOAT *z,MUD_FLOAT *cxz);
void dummy_crsyz(MUD_FLOAT *x,MUD_FLOAT *y,MUD_FLOAT *z,MUD_FLOAT *cxz);

void FieldE::set_updater(const Updater *up){
    if(up){
        _updater=up->clone();
        return;
    }

    if(dim() == 1){
        cout<<"1D multigrid is not supported!"<<endl;
        throw("1D multigrid is not supported!");
    }
    if(dim() == 2){
        if(_eps_type == 0){
            if(_uc.type_key() == HEXAGONAL){
                MUD2D* up2 = new MUD2D(MudpackTypes[MUD_CR],
                                        _Lx,_Ly,lx(),ly());
                up2->init();
                _updater = up2;
            }
            else{
                MUD2D* up2 = new MUD2D(MudpackTypes[MUD_SP],
                                        _Lx,_Ly,lx(),ly());
                //up2->display();
                up2->init();
                //up2->display();
                _updater = up2;
            }
        }
        else{
            if(_uc.type_key() == HEXAGONAL){
                MUD2D * up2 = new MUD2D(MudpackTypes[MUD_CR],
                                        _Lx,_Ly,lx(),ly());
                up2->set_cof(cof_mud2cr);
                up2->init();
                _updater = up2;
            }
            else{
                MUD2D * up2 = new MUD2D(MudpackTypes[MUD_SA],
                                        _Lx,_Ly,lx(),ly());
                up2->set_sig(sig_mud2,sig_mud2);
                //up2->display();
                up2->init();
                //up2->display();
                _updater = up2;
            }
        }
    }
    if(dim() == 3){
        if(_eps_type == 0){
            if(_uc.type_key() == HEXAGONAL){
                MUD3D* up3 = new MUD3D(MudpackTypes[MUD_CR], \
                                    _Lx,_Ly,_Lz,lx(),ly(),lz());
                up3->set_mgopt(0);
                up3->init();
                _updater = up3;
            }
            else if (_uc.type_key() == MONOCLINIC){
                MUD3D* up3 = new MUD3D(MudpackTypes[MUD_CR], \
                                    _Lx,_Ly,_Lz,lx(),ly(),lz());
                // exchange crsxy and crsxz
                up3->set_cof(cof_mud3cr_mono_ceps,
                             dummy_crsxz,dummy_crsxy,dummy_crsyz);
                up3->set_mgopt(0);
                up3->set_cr_extras(0,1,0);
                up3->init();
                _updater = up3;
            }
            else{
                MUD3D* up3 = new MUD3D(MudpackTypes[MUD_SP], \
                                    _Lx,_Ly,_Lz,lx(),ly(),lz());
                //up3->display();
                up3->init();
                //up3->display();
                _updater = up3;
            }
        }
        else{
            if(_uc.type_key() == HEXAGONAL){
                MUD3D* up3 = new MUD3D(MudpackTypes[MUD_CR], \
                                    _Lx,_Ly,_Lz,lx(),ly(),lz());
                up3->set_mgopt(0);
                up3->set_cof(cof_mud3cr_hex,
                             crs_mud3cr,dummy_crsxz,dummy_crsyz);
                up3->init();
                _updater = up3;
            }
            else if(_uc.type_key() == MONOCLINIC){
                MUD3D* up3 = new MUD3D(MudpackTypes[MUD_CR], \
                                    _Lx,_Ly,_Lz,lx(),ly(),lz());
                up3->set_mgopt(0);
                up3->set_cof(cof_mud3cr_mono_veps,
                             dummy_crsxz,crs_mud3cr,dummy_crsyz);
                up3->set_cr_extras(0,1,0);
                up3->init();
                _updater = up3;
            }
            else{
                MUD3D * up3 = new MUD3D(MudpackTypes[MUD_SA], \
                                        _Lx,_Ly,_Lz,lx(),ly(),lz());
                up3->set_sig(sig_mud3,sig_mud3,sig_mud3);
                //up3->display();
                up3->init();
                //up3->display();
                _updater = up3;
            }
        }
    }
}

/**
 * set eps in gmudsig. Only works for _eps_type = 1.
 *
 */
void FieldE::set_eps(const Grid &eps){
    blitz::Range all = blitz::Range::all();
    if(dim() == 1){
    }
    if(dim() == 2){
        if(_uc.type_key() == HEXAGONAL){
            _updater->set_data(eps);
        }
        else{
            blitz::Array<double,3> eps_in(eps.data());
            _updater->set_data(eps_in(all,all,0));
        }
    }
    if(dim() == 3){
      if(_uc.type_key() == HEXAGONAL || _uc.type_key() == MONOCLINIC)
        _updater->set_data(eps);
      else
        _updater->set_data(eps.data());
    }
}

/**
 * Updating electrostatic field by solving the Poisson Equation.
 *
 * The Poisson equation
 *          L(psi) = -(_cc) * rho
 * where _cc = N / epsS for constant dielectric constant, 
 * and _cc = N for position-dependent dielectric constant.
 * It can be rescaled to
 *          p_xx + p_yy = -rho
 * with
 *          p = psi / _cc
 * Thus, the psi is obtained by multiply the _cc to the solution
 *          psi = _cc * p
 *
 */
void FieldE::update(const Grid &rho,Updater *up){
    Grid u(*this);
    up->solve(u,rho);
    // All boundaries are periodic make the PDE singular
    // Add another constraint which set u.mean = 0
    u = u - u.mean();
    _data = _lambda * (_cc * u.data()) + (1.0 - _lambda) * _data;
}

