/**
 * ABSe_ps_mud_pbc.cc
 * Created at 2011.12.25
 *
 * SCFT for A-B + Solvent + salt with pseudo-spectral algorithm and 
 * full multigrid algorithm with periodic boundary condition 
 * for 1D, 2D, and 3D space.
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

#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include "Model_ABSe.h"
#include "scft.h"

using namespace std;

void run_scft(){    
    clock_t t_b,t_e;
    time_t rawtime;
    struct tm* timeinfo;
    ofstream myfile;
    double timeCost;

    t_b=clock();
    Config *cfg;
    cfg=new Config("param.ini");
    Model_ABSe model(*cfg);
    delete cfg;
    scft sim("param.ini",&model);
    sim.run();
    t_e=clock();

    timeCost=static_cast<double>(t_e-t_b)/CLOCKS_PER_SEC;
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    myfile.open("done.txt");
    myfile<<"Program terminates at: "<<asctime(timeinfo);
    myfile<<"Total time elapsed (hour): "<<timeCost/3600.0<<endl;
    myfile.close();
}

int main(){
    run_scft();
    return 0;
}
