/**
 * Helper.cc
 * Created at 2011.6.14
 *
 * Implementation of Help.h
 *
 * Copyright (C) 2012-2014 Yi-Xin Liu <lyx@fudan.edu.cn>
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

#include "Helper.h"

void Helper::init_pattern(Grid &g,const PhasePattern pt,const double c,const double v1,const double v2){
    switch(pt){
        case LAM1_PATTERN:
            init_LAM1(g,c,v1,v2);
            break;
        case LAM2_PATTERN:
            init_LAM2(g,c,v1,v2);
            break;
        case LAM3_PATTERN:
            init_LAM3(g,c,v1,v2);
            break;
        case LAM4_PATTERN:
            init_LAM4(g,c,v1,v2);
            break;
        case LAM5_PATTERN:
            init_LAM5(g,c,v1,v2);
            break;
        case LAM6_PATTERN:
            init_LAM6(g,c,v1,v2);
            break;
        default:
            cout<<"Unknown or unsupported phase pattern!"<<endl;
            exit(1);
    }
}

void Helper::init_LAM1(Grid &g,const double c,const double v1,const double v2){
    for(int i=0;i<g.Lx();i++)
      for(int j=0;j<g.Ly();j++)
        for(int k=0;k<g.Lz();k++){
            if(j<c*g.Ly())
                g(i,j,k)=v1;
            else
                g(i,j,k)=v2;
        }
}

void Helper::init_LAM2(Grid &g,const double c,const double v1,const double v2){
    for(int i=0;i<g.Lx();i++)
      for(int j=0;j<g.Ly();j++)
        for(int k=0;k<g.Lz();k++){
            if(k<c*g.Lz())
                g(i,j,k)=v1;
            else
                g(i,j,k)=v2;
        }
}

void Helper::init_LAM3(Grid &g,const double c,const double v1,const double v2){
    for(int i=0;i<g.Lx();i++)
      for(int j=0;j<g.Ly();j++)
        for(int k=0;k<g.Lz();k++){
            if(i<c*g.Lx())
                g(i,j,k)=v1;
            else
                g(i,j,k)=v2;
        }
}

void Helper::init_LAM4(Grid &g,const double c,const double v1,const double v2){
    for(int i=0;i<g.Lx();i++)
      for(int j=0;j<g.Ly();j++)
        for(int k=0;k<g.Lz();k++){
            if(j<c*g.Ly()/2)
                g(i,j,k)=v1;
            else if(j>=c*g.Ly()/2 && j<g.Ly()/2)
                g(i,j,k)=v2;
            else if(j>=g.Ly()/2 && j<(1+c)*g.Ly()/2)
                g(i,j,k)=v1;
            else
                g(i,j,k)=v2;
        }
}

void Helper::init_LAM5(Grid &g,const double c,const double v1,const double v2){
    for(int i=0;i<g.Lx();i++)
      for(int j=0;j<g.Ly();j++)
        for(int k=0;k<g.Lz();k++){
            if(k<c*g.Lz()/2)
                g(i,j,k)=v1;
            else if(k>=c*g.Lz()/2 && k<g.Lz()/2)
                g(i,j,k)=v2;
            else if(k>=g.Lz()/2 && k<(1+c)*g.Lz()/2)
                g(i,j,k)=v1;
            else
                g(i,j,k)=v2;
        }
}

void Helper::init_LAM6(Grid &g,const double c,const double v1,const double v2){
    for(int i=0;i<g.Lx();i++)
      for(int j=0;j<g.Ly();j++)
        for(int k=0;k<g.Lz();k++){
            if(i<c*g.Lx()/2)
                g(i,j,k)=v1;
            else if(i>=c*g.Lx()/2 && i<g.Lx()/2)
                g(i,j,k)=v2;
            else if(i>=g.Lx()/2 && i<(1+c)*g.Lx()/2)
                g(i,j,k)=v1;
            else
                g(i,j,k)=v2;
        }
}

