/**
 * scft.cc
 * Created at 2011.6.6
 *
 * Implementation of scft.h
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

#include "scft.h"
#include "UnitCell.h"

#include "armadillo"

scft::scft(const string config_file, Model* pmodel)
                        :_cfg(config_file.c_str()), _pmodel(pmodel),
                         _num_scft(0), _iter(0){
    init();
}

/**
 * NOTE: only the size of simulation cell is supported in batch processing.
 */
void scft::run(){
    AlgorithmCellType actype = _cfg.algo_cell_optimization_type();
    if(actype == AlgorithmCellType::BRENT){
        /** update cell size by Brent's method **/
        bracket();
        cout<<"aa = "<<aa<<" ba = "<<ba<<" ca = "<<ca<<endl;
        cout<<"Faa = "<<Faa<<" Fba = "<<Fba<<" Fca = "<<Fca<<endl;

        optimization_Brent();
        arma::uword i = Fs.n_elem - 1;

        // Update the final result
        arma::uword curr_iter = Fs.n_elem;
        as.resize(curr_iter + 1);
        bs.resize(curr_iter + 1);
        cs.resize(curr_iter + 1);
        Fs.resize(curr_iter + 1);
        _cfg.a(amin);
        _cfg.b(bmin);
        _cfg.c(cmin);
        // it is necessary to use uc.lx, uc.ly, uc.lz instead of
        // uc.a, uc.b, uc.c or a, b, c to correctly handle
        // 1D and 2D situations.
        UnitCell uc(_cfg);
        as(curr_iter) = uc.lx();
        bs(curr_iter) = uc.ly();
        cs(curr_iter) = uc.lz();
        Fs(curr_iter) = Fmin;
        string s;
        if(_cfg.data_file().empty())
            s = _cfg.base_dir() + DEFAULT_DATA_FILE;
        else
            s = _cfg.base_dir() + _cfg.data_file();
        save_cell(s+"_cell.mat");
        cout<<"amin = "<<uc.lx()<<" bmin = "<<uc.ly()<<" cmin = "<<uc.lz()<<endl;
        cout<<"Fmin = "<<Fmin<<endl;
    }
    else if(actype == AlgorithmCellType::SINGLE){
        double a0 = _cfg.a();
        double b0 = _cfg.b();
        double c0 = _cfg.c();
        solve_scft(a0, b0, c0);
    }
    else if(actype == AlgorithmCellType::MANUAL){
        /** Manually update cell size **/
        arma::vec min = _cfg.batch_cell_min();
        arma::vec max = _cfg.batch_cell_max();
        arma::vec step = _cfg.batch_cell_step();
        for(double a=min(0); a<=max(0); a+=step(0))
            for(double b=min(1); b<=max(1); b+=step(1))
                for(double c=min(2); c<=max(2); c+=step(2)){
                    reset_model(a, b, c);
                    _pmodel->display_parameters();
                    init_record();
                    save_param();
                    relax();
                }
    }
    else{
        cerr<<"Cell optimization algorithm: ";
        cerr<<_cfg.get_algo_cell_optimization_type_string();
        cerr<<" is not available."<<endl;
        exit(1);
    }
}

void scft::update_cell(const double a, double& b, double& c){
    arma::vec min = _cfg.batch_cell_min();
    double a0 = min(0);
    double b0 = min(1);
    double c0 = min(2);
    b = b0;
    c = c0;

    if(_cfg.dim() == 2 && _cfg.gtypey() == GridType::REGULAR)
        b = b0 * (a / a0);

    if(_cfg.dim() == 3 && _cfg.gtypez() == GridType::REGULAR){
        b = b0 * (a / a0);
        c = c0 * (a / a0);
    }
}

void scft::bracket(){
    const double GOLD = 1.618034;
    const double GLIMIT = 100.0;
    const double TINY = 1.0e-20;

    arma::vec min = _cfg.batch_cell_min();
    arma::vec step = _cfg.batch_cell_step();

    double a0 = min(0);
    double b0 = min(1);
    double c0 = min(2);
    double b, c;
    aa = a0;
    Faa = solve_scft(aa, b0, c0);

    ba = aa + step(0);
    update_cell(ba, b, c);
    Fba = solve_scft(ba, b, c);

    double Fua;

    if(Fba > Faa){
        double t = aa;
        aa = ba;
        ba = t;
        t = Faa;
        Faa = Fba;
        Fba = t;
    }

    ca = ba + GOLD * (ba - aa);
    update_cell(ca, b, c);
    Fca = solve_scft(ca, b, c);

    while(Fba > Fca){
        double r = (ba - aa) * (Fba - Fca);
        double q = (ba - ca) * (Fba - Faa);
        double p = q - r;
        if(abs(p) < TINY)
            p = p > 0.0 ? TINY : -TINY;
        double ua = ba - ((ba - ca)*q - (ba - aa)*r) / (2 * p);
        double ualim = ba + GLIMIT * (ca - ba);
        if((ba - ua) * (ua - ca) > 0.0){
            update_cell(ua, b, c);
            Fua = solve_scft(ua, b, c);
            if(Fua < Fca){
                aa = ba;
                ba = ua;
                Faa = Fba;
                Fba = Fua;
                return;
            }
            else if(Fua > Fba){
                ca = ua;
                Fca = Fua;
                return;
            }
            ua = ca + GOLD * (ca - ba);
            update_cell(ua, b, c);
            Fua = solve_scft(ua, b, c);
        }
        else if((ca - ua) * (ua - ualim) > 0.0){
            update_cell(ua, b, c);
            Fua = solve_scft(ua, b, c);
            if(Fua < Fca){
                ba = ca;
                ca = ua;
                ua = ua + GOLD * (ua - ca);
                Fba = Fca;
                Fca = Fua;
                update_cell(ua, b, c);
                Fua = solve_scft(ua, b, c);
            }
        }
        else if((ua - ualim) * (ualim - ca) >= 0.0){
            ua = ualim;
            update_cell(ua, b, c);
            Fua = solve_scft(ua, b, c);
        }
        else{
            ua = ca + GOLD * (ca - ba);
            update_cell(ua, b, c);
            Fua = solve_scft(ua, b, c);
        }
        aa = ba;
        ba = ca;
        ca = ua;
        Faa = Fba;
        Fba = Fca;
        Fca = Fua;
    }
}

void scft::optimization_Brent(){
    const double CGOLD = 0.3819660;
    const double ITMAX = max_iter_cell;
    const double ZEPS = 1.0e-20;
    const double TOL = _cfg.tol_cell();

    arma::vec min = _cfg.batch_cell_min();
    double a0 = min(0);
    double b0 = min(1);
    double c0 = min(2);
    double b, c;

    double axa, bxa, u, v, w, x, xm;
    double p, q, r;
    double Fu, Fv, Fw, Fx;
    double e = 0.0, d = 0.0, et, tol1, tol2;

    axa = aa < ca ? aa : ca;
    bxa = aa > ca ? aa : ca;
    x = v = w = ba;
    Fx = Fv = Fw = Fba;
    for(arma::uword iter=0; iter < ITMAX; iter++){
        cout<<"iter (Brent's method) = "<<iter<<endl;
        xm = 0.5 * (axa + bxa);
        tol1 = TOL * abs(x);
        tol2 = 2.0 * (tol1 + ZEPS);
        if(abs(x - xm) <= (tol2 - 0.5*(bxa-axa))){
            Fmin = Fx;
            amin = x;
            bmin = b0 * (x / a0);
            cmin = c0 * (x / a0);
            return;
        }
        if(abs(e) > tol1){
            r = (x - w) * (Fx - Fv);
            q = (x - v) * (Fx - Fw);
            p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if(q > 0) p = -p;
            q = abs(q);
            et = e;
            e = d;
            if(abs(p) >= abs(0.5*q*et) || p <= q*(axa-x) || p >= q * (bxa-x)){
                e = x > xm ? axa - x : bxa - x;
                d = CGOLD * e;
            }
            else{
                d = p / q;
                u = x + d;
                if(u - axa < tol2 || bxa - u < tol2)
                    d = x > xm ? -tol1 : tol1;
            }
        }
        else{
            e = x > xm ? axa - x : bxa - x;
            d = CGOLD * e;
        }
        double t = d > 0.0 ? tol1 : -tol1;
        u = abs(d) >= tol1 ? x + d : x + t;
        //b = b0 * (u / a0);
        //c = c0 * (u / a0);
        update_cell(u, b, c);
        Fu = solve_scft(u, b, c);  // only evaluate function once per iter
        if(Fu <= Fx){
            if(u >= x)
                axa = x;
            else
                bxa = x;
            v = w;
            w = x;
            x = u;
            Fv = Fw;
            Fw = Fx;
            Fx = Fu;
        }
        else{
            if(u < x)
                axa = u;
            else
                bxa = u;
            if(Fu <= Fw || w == x){
                v = w;
                w = u;
                Fv = Fw;
                Fw = Fu;
            }
            else if(Fu <= Fv || v == x || v == w){
                v = u;
                Fv = Fu;
            }
        }
    }
}

double scft::solve_scft(double a, double b, double c){
    blitz::Range all = blitz::Range::all();
    if(_num_scft > 0)
        reset_model(a, b, c);
    _pmodel->display_parameters();
    init_record();
    save_param();
    relax();
    double F = _pmodel->H();

    arma::uword curr_iter = as.n_elem;
    as.resize(curr_iter + 1);
    bs.resize(curr_iter + 1);
    cs.resize(curr_iter + 1);
    Fs.resize(curr_iter + 1);
    as(curr_iter) = _curr_a;
    bs(curr_iter) = _curr_b;
    cs(curr_iter) = _curr_c;
    Fs(curr_iter) = F;

    _num_scft++;

    return F;
}

void scft::reset_model(double a, double b, double c){
    _cfg.a(a);
    _cfg.b(b);
    _cfg.c(c);
    _cfg.set_grid_init_type(GridInitType::FILE_INIT);
    _cfg.set_field_data_file(prev_data_file);
    // it is necessary to use uc.lx, uc.ly, uc.lz instead of
    // uc.a, uc.b, uc.c or a, b, c to correctly handle
    // 1D and 2D situations.
    UnitCell uc(_cfg);
    _curr_a = uc.lx();
    _curr_b = uc.ly();
    _curr_c = uc.lz();

    string config_data;
    if(!_cfg.get_config_data(config_data)){
        cerr<<"Getting config data failed. Abort."<<endl;
        exit(1);
    }
    _pmodel->reset(config_data);
}

void scft::relax(){
    clock_t t_0, t_b, t_e;
    double t;
    string s;
    int i_save = -1;
    bool is_break = false;

    if(_cfg.data_file().empty())
        s = _cfg.base_dir() + DEFAULT_DATA_FILE;
    else
        s = _cfg.base_dir() + _cfg.data_file();

    t_0 = clock();
    t_b = t_0;
    for(int i=1; i<=_cfg.max_iter(); i++){
        _iter = i;
        _pmodel->update();

        if(_cfg.is_display() && i%_cfg.display_interval() == 0){
            t_e = clock();
            t = static_cast<double>(t_e - t_b) / CLOCKS_PER_SEC;
            t /= _cfg.display_interval();
            display(t);
            t_b = clock();
        }

        if(_cfg.is_save_data()){
            if(i%_cfg.record_interval() == 0){
                i_save++;
                _t(i_save) = _iter;
                t_e = clock();
                t = static_cast<double>(t_e - t_0) / CLOCKS_PER_SEC;
                _time(i_save) = t;
                _H(i_save) = _pmodel->H();
                _residual_error(i_save) = _pmodel->residual_error();
                //_incomp(i_save) = _pmodel->incomp();
            }
            is_break = save_data(s, i_save);
        }

        if(is_break) break;
    }
}

void scft::init(){
    _num_iters = 1.0 * _cfg.max_iter() / _cfg.record_interval() + 1;
    max_iter_cell = _cfg.max_iter_cell();
    prev_data_file = _cfg.field_data_file();
    if(_cfg.is_save_data())
        init_record();
    UnitCell uc(_cfg);
    _curr_a = uc.lx();
    _curr_b = uc.ly();
    _curr_c = uc.lz();
}

void scft::init_record(){
    _t.resize(_num_iters);
    _t = 0.0;
    _time.resize(_num_iters);
    _time = 0.0;
    _H.resize(_num_iters);
    _H = 0.0;
    _residual_error.resize(_num_iters);
    _residual_error = 0.0;
    _incomp.resize(_num_iters);
    _incomp = 0.0;
}

void scft::display(const double t) const{
    cout<<_iter<<endl;
    cout<<"t = "<<t<<endl;
    //cout.precision(6);
    //cout<<"\tF_min = "<<_minH;
    //cout<<"\t(a, b, c) = "<<"("<<_min_a<<", "<<_min_b<<", "<<_min_c<<")"<<endl;
    _pmodel->display();
}

void scft::save_param(){
    string s;
    if(_cfg.param_file().empty())
        s = _cfg.base_dir() + DEFAULT_PARAM_FILE;
    else
        s = _cfg.base_dir() + _cfg.param_file();
    stringstream ss;
    ss<<s<<"_a"<<_curr_a<<"_b"<<_curr_b<<"_c"<<_curr_c<<".mat";
    _pmodel->save_model(ss.str());
    save_scft(ss.str());
}

void scft::save_record(const string file){
    CMatFile mat;
    mwSize dims1[1] = {(mwSize)_num_iters};

    mat.matInit(file,"u");
    if(!mat.queryStatus()){
        mat.matPut("t", _t.data(), \
                    _t.size()*sizeof(double), \
                    1, dims1, mxDOUBLE_CLASS, mxREAL);
        mat.matPut("time", _time.data(), \
                    _time.size()*sizeof(double), \
                    1, dims1, mxDOUBLE_CLASS, mxREAL);
        mat.matPut("F", _H.data(), \
                    _H.size()*sizeof(double), \
                    1, dims1, mxDOUBLE_CLASS, mxREAL);
        mat.matPut("err_residual", _residual_error.data(), \
                    _residual_error.size()*sizeof(double), \
                    1, dims1, mxDOUBLE_CLASS, mxREAL);
        mat.matPut("incomp", _incomp.data(), \
                    _incomp.size()*sizeof(double), \
                    1, dims1, mxDOUBLE_CLASS, mxREAL);
        mat.matRelease();
    }
}

void scft::save_scft(const string file){
    CMatFile mat;
    mwSize dims1[1] = {3};

    mat.matInit(file,"u");
    if(!mat.queryStatus()){
        mat.matPutString("base_dir", _cfg.base_dir());
        mat.matPutString("data_file", _cfg.data_file());
        mat.matPutString("param_file", _cfg.param_file());
        mat.matPutScalar("min_iter", _cfg.min_iter());
        mat.matPutScalar("max_iter", _cfg.max_iter());
        mat.matPutScalar("record_interval", _cfg.record_interval());
        mat.matPutScalar("save_interval", _cfg.save_interval());
        mat.matPutScalar("thresh_H", _cfg.thresh_H());
        mat.matPutScalar("thresh_residual", _cfg.thresh_residual());
        mat.matPutScalar("thresh_incomp", _cfg.thresh_incomp());
        mat.matPutScalar("tol_cell", _cfg.tol_cell());
        mat.matPutScalar("max_iter_cell", _cfg.max_iter_cell());
        arma::vec min = _cfg.batch_cell_min();
        mat.matPut("batch_cell_min", min.memptr(), \
                    min.n_elem*sizeof(double), \
                    1, dims1, mxDOUBLE_CLASS, mxREAL);
        arma::vec max = _cfg.batch_cell_max();
        mat.matPut("batch_cell_max", max.memptr(), \
                    max.n_elem*sizeof(double), \
                    1, dims1, mxDOUBLE_CLASS, mxREAL);
        arma::vec step = _cfg.batch_cell_step();
        mat.matPut("batch_cell_step", step.memptr(), \
                    step.n_elem*sizeof(double), \
                    1, dims1, mxDOUBLE_CLASS, mxREAL);
        mat.matRelease();
    }
}

/**
 *  Input argument: s = base_dir/data_file
 */
bool scft::save_data(const string s, const int i_save){
    stringstream ss;
    ss<<s<<"_a"<<_curr_a<<"_b"<<_curr_b<<"_c"<<_curr_c<<"_"<<_iter<<".mat";

    // diverged
    if(!isFiniteNumber(_H(i_save))){
        cout<<endl;
        cout<<"Error: diverge"<<endl;
        cout<<"Iter: "<<_iter<<endl;
        cout<<"H = "<<_H[i_save]<<endl;
        return true;
    }

    // converged but energy is larger than the found minimum energy
    
    /*if(_iter > _cfg.min_iter() &&
                i_save>2 && \
                abs(_H(i_save-1) - _H(i_save-2)) < _cfg.thresh_H() && \
                _residual_error(i_save) < _cfg.thresh_residual() && \
                _H(i_save) > _minH){
        _pmodel->save(ss.str());
        save_record(ss.str());
        if(_cfg.is_save_q()) _pmodel->save_q(ss.str());
        return true;
    }*/
    

    // converged
    if(_iter > _cfg.min_iter() && \
                i_save > 0 && \
                _residual_error(i_save) < _cfg.thresh_residual()){
        _pmodel->save(ss.str());
        save_record(ss.str());
        if(_cfg.is_save_q()) _pmodel->save_q(ss.str());
        prev_data_file = ss.str();
        return true;
    }

    // max_iter has reached
    if(_iter == _cfg.max_iter()){
        _pmodel->save(ss.str());
        save_record(ss.str());
        if(_cfg.is_save_q()) _pmodel->save_q(ss.str());
        prev_data_file = ss.str();
        return true;
    }

    // save_interval has reached
    if(_iter%_cfg.save_interval() == 0){
        _pmodel->save(ss.str());
        save_record(ss.str());
        if(_cfg.is_save_q()) _pmodel->save_q(ss.str());
    }

    return false;
}

void scft::save_cell(const string file){
    CMatFile mat;
    mwSize dims1[1] = {Fs.n_elem};

    mat.matInit(file,"u");
    if(!mat.queryStatus()){
        mat.matPut("a", as.memptr(), \
                    as.n_elem*sizeof(double), \
                    1, dims1, mxDOUBLE_CLASS, mxREAL);
        mat.matPut("b", bs.memptr(), \
                    bs.n_elem*sizeof(double), \
                    1, dims1, mxDOUBLE_CLASS, mxREAL);
        mat.matPut("c", cs.memptr(), \
                    cs.n_elem*sizeof(double), \
                    1, dims1, mxDOUBLE_CLASS, mxREAL);
        mat.matPut("F", Fs.memptr(), \
                    Fs.n_elem*sizeof(double), \
                    1, dims1, mxDOUBLE_CLASS, mxREAL);
        mat.matRelease();
    }
}
