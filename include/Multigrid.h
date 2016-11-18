/**
 * multigrid.h
 * Created at 2012.4.12
 * 
 * a collection of common functions for constructing
 * multigrid algorithm.
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

#ifndef polyorder_multigrid_h
#define polyorder_multigrid_h

#include <blitz/array.h>

#include "Grid.h"

void interp_linear_1d(const blitz::Array<double,1> uc,blitz::Array<double,1> uf);

void interp_cubic_1d(const blitz::Array<double,1> uc,blitz::Array<double,1> uf);

void rstrct_injection_1d(const blitz::Array<double,1> uf,blitz::Array<double,1> uc);
        
void rstrct_weight_1d(const blitz::Array<double,1> uf,blitz::Array<double,1> uc);

void interp_bilinear_2d(const blitz::Array<double,2> uc,blitz::Array<double,2> uf);

void interp_bicubic_2d(const blitz::Array<double,2> uc,blitz::Array<double,2> uf);

void rstrct_injection_2d(const blitz::Array<double,2> uf,blitz::Array<double,2> uc);

void rstrct_hw_2d(const blitz::Array<double,2> uf,blitz::Array<double,2> uc);

void rstrct_fw_2d(const blitz::Array<double,2> uf,blitz::Array<double,2> uc);

void interp_trilinear_3d(const blitz::Array<double,3> uc,blitz::Array<double,3> uf);

void interp_tricubic_3d(const blitz::Array<double,3> uc,blitz::Array<double,3> uf);

void rstrct_injection_3d(const blitz::Array<double,3> uf,blitz::Array<double,3> uc);

void rstrct_hw_3d(const blitz::Array<double,3> uf,blitz::Array<double,3> uc);

void rstrct_fw_3d(const blitz::Array<double,3> uf,blitz::Array<double,3> uc);

void fft2mg_copy_1d(const Grid &fft,blitz::Array<double,1> mg);

void mg2fft_copy_1d(const blitz::Array<double,1> mg,Grid &fft);

void fft2mg_copy_1d(const blitz::Array<double,1> fft,blitz::Array<double,1> mg);

void mg2fft_copy_1d(const blitz::Array<double,1> mg,blitz::Array<double,1> fft);

void fft2mg_interp_1d(const Grid &fft,blitz::Array<double,1> mg,int interp_kind=0);

void mg2fft_interp_1d(const blitz::Array<double,1> mg,Grid &fft);

void fft2mg_interp_1d(const blitz::Array<double,1> fft,blitz::Array<double,1> mg,int interp_kind=0);

void mg2fft_interp_1d(const blitz::Array<double,1> mg,blitz::Array<double,1> fft);

void fft2mg_copy_2d(const Grid &fft,blitz::Array<double,2> mg);

void mg2fft_copy_2d(const blitz::Array<double,2> mg,Grid &fft);

void fft2mg_copy_2d(const blitz::Array<double,2> fft,blitz::Array<double,2> mg);

void mg2fft_copy_2d(const blitz::Array<double,2> mg,blitz::Array<double,2> fft);

void fft2mg_interp_2d(const Grid &fft,blitz::Array<double,2> mg,int interp_kind=0);

void mg2fft_interp_2d(const blitz::Array<double,2> mg,Grid &fft);

void fft2mg_interp_2d(const blitz::Array<double,2> fft,blitz::Array<double,2> mg,int interp_kind=0);

void mg2fft_interp_2d(const blitz::Array<double,2> mg,blitz::Array<double,2> fft);

void fft2mg_copy_3d(const Grid &fft,blitz::Array<double,3> mg);

void mg2fft_copy_3d(const blitz::Array<double,3> mg,Grid &fft);

void fft2mg_copy_3d(const blitz::Array<double,3> fft,blitz::Array<double,3> mg);

void mg2fft_copy_3d(const blitz::Array<double,3> mg,blitz::Array<double,3> fft);

void fft2mg_interp_3d(const Grid &fft,blitz::Array<double,3> mg,int interp_kind=0);

void mg2fft_interp_3d(const blitz::Array<double,3> mg,Grid &fft);

void fft2mg_interp_3d(const blitz::Array<double,3> fft,blitz::Array<double,3> mg,int interp_kind=0);

void mg2fft_interp_3d(const blitz::Array<double,3> mg,blitz::Array<double,3> fft);

#endif

