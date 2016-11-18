/**
 * scft_confine.cc
 * Created at 2014.06.14
 *
 * SCFT for confined system.
 *
 * Copyright (C) 2014 Yi-Xin Liu <lyx@fudan.edu.cn>
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

#include "AB_Slab.h"
#include "Config.h"
#include "scft.h"

using namespace std;

void run_scft(){
    clock_t t_b, t_e;
    time_t rawtime;
    struct tm* timeinfo;
    ofstream myfile;
    double timeCost;

    t_b = clock();
    Config *cfg;
    cout<<"Reading confguration file..."<<endl;
    cfg = new Config("param.ini");

    cout<<"Constructing AB_Slab model..."<<endl;
    AB_Slab model(*cfg);
    delete cfg;

    cout<<"Setup simulation instance..."<<endl;
    scft sim("param.ini", &model);
    cout<<"Running simulations..."<<endl;
    sim.run();
    t_e = clock();

    timeCost = static_cast<double>(t_e - t_b) / CLOCKS_PER_SEC;
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
