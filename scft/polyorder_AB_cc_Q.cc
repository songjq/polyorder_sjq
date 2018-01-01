/**
 * polyorder.cc
 * Created at 2014.09.18 by Yi-Xin Liu
 *
 * A sample PolyOrder application demonstrating how to build a SCFT simulation
 * application using PolyOrder library.
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

#include "common.h"
#include "Config.h"
#include "scft.h"
#include "models.h"

using namespace std;

void run_scft(string config_file){
    clock_t t_b, t_e;
    time_t rawtime;
    struct tm* timeinfo_start;
    struct tm* timeinfo_end;
    ofstream myfile;
    double timeCost;

    time(&rawtime);
    timeinfo_start = localtime(&rawtime);
    string start_time(asctime(timeinfo_start));

    cout<<"Checking configuration file ..."<<endl;
    Config cfg(config_file.c_str());
    cout<<"Configuration file: "<<config_file<<" is OK!"<<endl;
    cout<<endl;

    cout<<"Constructing "<<cfg.get_model_type_string()<<" model..."<<endl;
    Model *pmodel;
    if(cfg.model() == ModelType::AB) {
        pmodel = new Model_AB(config_file);
    }
    else if(cfg.model() == ModelType::AB_C) {
        pmodel = new Model_AB_C(config_file);
    }
    else if(cfg.model() == ModelType::A_B) {
        pmodel = new Model_A_B(config_file);
    }
    else if(cfg.model() == ModelType::ABW) {
        pmodel = new Model_ABW(config_file);
    } 
    else if(cfg.model() == ModelType::AS) {
        pmodel = new Model_AS(config_file);
    }
    else if(cfg.model() == ModelType::AB_S) {
        pmodel = new Model_AB_S(config_file);
    }   
    else{
        cerr<<cfg.get_model_type_string()<<" model is not available."<<endl;
        exit(1);
    }
    cout<<cfg.get_model_type_string()<<" model is constructed."<<endl;
    cout<<endl;

    cout<<"Setup simulation instance..."<<endl;
    scft sim(config_file, pmodel);
    cout<<"Simulation instance is ready."<<endl;
    cout<<endl;

    cout<<"Running simulations..."<<endl;
    t_b = clock();
    sim.run();
    t_e = clock();
    timeCost = static_cast<double>(t_e - t_b) / CLOCKS_PER_SEC;

    time(&rawtime);
    timeinfo_end = localtime(&rawtime);

    myfile.open("done.txt");
    myfile<<"Program starts at:\t\t\t"<<start_time;
    myfile<<"Program terminates at:\t\t\t"<<asctime(timeinfo_end);
    myfile<<"Total time elapsed in second:\t"<<timeCost<<endl;
    myfile<<"Total time elapsed in minute:\t"<<timeCost/60.0<<endl;
    myfile<<"Total time elapsed in hour:\t"<<timeCost/3600.0<<endl;
    myfile.close();
}

int main(int argc, char* argv[]){
    // the default configuration file is in the current working directory
    string config_file = "param.ini";
    // Specify configuration file through input arguments.
    if(argc > 1){
        config_file = argv[1];
    }
    run_scft(config_file);
    return 0;
}
