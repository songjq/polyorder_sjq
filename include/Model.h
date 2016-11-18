/**
 * Model.h
 * Created at 2011.6.6
 *
 * Model is the base class abstrating the actual polymer model
 * that is simulated. It provides the interfacce of updating
 * and saving the underlying potential fields (field in short)
 * and density fields (density in short).
 *
 * HISTORY:
 * 2012.4.1
 *   1. From now on, the history of this file is tracked by Mercurial.
 *   2. Package name: polyorder.
 * 2011.6.6
 *   1. original version
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

#ifndef polyorder_model_h
#define polyorder_model_h

#include <string>
#include "Config.h"
#include <blitz/array.h>
const double kPI = 3.141592653589793;

using std::string;

class Model{
public:
    Model(){}
	Model(const string config_file):_cfg(config_file.c_str()){}
	virtual void init()=0;
    virtual void reset(const string& config_data)=0;
	virtual void update()=0;
    virtual double Hw() const=0;
    virtual double Hs() const=0;
    virtual double H() const=0;
    virtual double residual_error() const=0;
    virtual double incomp() const=0;
    virtual void display() const=0;
    virtual void display_parameters() const=0;
    virtual void save(const string file)=0;
    virtual void save_model(const string file)=0;
    virtual void save_field(const string file)=0;
    virtual void save_density(const string file)=0;
    virtual void save_q(const string file)=0;
    //virtual void input_AField(blitz::Array<double, 3>) = 0;  //added by songjq for string method in 20161001
    //virtual void input_BField(blitz::Array<double, 3>) = 0;  //added by songjq for string method in 20161001
    //virtual void init_data_field() = 0; //added by songjq for string method in 20161001
    //virtual blitz::Array<double, 4> output_data() = 0; //added by songjq for string method in 20161001

protected:
    virtual void init_field()=0;
    virtual void init_density()=0;
    virtual void init_propagator()=0;
    virtual void release_memory()=0;

protected:
    // To initialize _cfg,
    // derived class should call the constructor Model(config_file)
    // in its constructor's initialization list.
    Config _cfg;  // Configurational object.
};

#endif

