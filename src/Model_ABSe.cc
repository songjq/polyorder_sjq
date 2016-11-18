/**
 * Model_ABSe.cc
 * Created at 2011.6.25
 *
 * Implementation of Model_ABSe.h.
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

#include "Model_ABSe.h"

using namespace std;

void Model_ABSe::init(const Config &cfg){
    _number_of_component = cfg.get_integer("Model","number_of_component");

    _N = cfg.get_integer("Model","N");
    _fA = cfg.get_double("Model","fA");
    _a = cfg.get_double("Model","a");
    _Rg = _a*sqrt(_N/6.0);

    _Ms = cfg.get_integer("Model","Ms");
    _ds = 1.0 / (_Ms -1);
    int L = _Ms - 1;
    int LA = int(_fA * L);
    _sA = LA + 1;
    int LB = L - LA;
    _sB = LB + 1;

    _fA = 1.0 * LA / L;
    _fB = 1 - _fA;
    _NA = _fA * _N;
    _NB = _N - _NA;

    _chiN = cfg.get_double("Model","chiN");
    _chiAB = _chiN/_N;
    _chiASN = cfg.get_double("Model","chiASN");
    _chiBSN = cfg.get_double("Model","chiBSN");

    _cs = cfg.get_double("Model","cs");

    _alphaA = cfg.get_double("Model","alphaA");
    _alphaB = cfg.get_double("Model","alphaB");

    _upsA = cfg.get_integer("Model","upsA");
    _upsB = cfg.get_integer("Model","upsB");
    _upsP = cfg.get_integer("Model","upsP");
    _upsN = cfg.get_integer("Model","upsN");

    _epsA = cfg.get_double("Model","epsA");
    _epsB = cfg.get_double("Model","epsB");
    _epsS = cfg.get_double("Model","epsS");

    _phiC_avg = cfg.get_double("Model","phiC");
    _phiA_avg = _fA*_phiC_avg;
    _phiB_avg = _fB*_phiC_avg;
    _phiS_avg = 1.0-_phiC_avg;
    _phiP_avg = _cs-_upsA*_alphaA*_phiA_avg/_upsP;
    _phiN_avg = -(_upsP*_phiP_avg + _upsA*_alphaA*_phiA_avg + \
                _upsB*_alphaB*_phiB_avg)/_upsN;

    _charge_distribution = cfg.get_integer("Algorithm", \
                                        "charge_distribution");
    _dielectric_constant = cfg.get_integer("Algorithm", \
                                        "dielectric_constant");
    _density_integration = cfg.get_integer("Algorithm", \
                                        "density_integration");

    init_field(cfg);
    init_density(cfg);
    init_propagator(cfg);
}

void Model_ABSe::update(){
    Grid wAeff(*_wA);
    Grid wBeff(*_wB);

    // Smeared: 0; Annealed: 1
    if(_charge_distribution == 0){
        if(_upsA != 0) wAeff += (_N * _alphaA * _upsA) * (*_psi);
        if(_upsB != 0) wBeff += (_N * _alphaB * _upsB) * (*_psi);
    }
    else{
        if(_upsA != 0)
          wAeff -= _N * log(1. - _alphaA + _alphaA * exp(_upsA* (*_psi)));
        if(_upsB != 0)
          wBeff -= _N * log(1. - _alphaB + _alphaB * exp(-_upsB * (*_psi)));
    }

    _qA->update(wAeff);
    _qB->set_head(_qA->get_tail());
    _qB->update(wBeff);
    _qBc->update(wBeff);
    _qAc->set_head(_qBc->get_tail());
    _qAc->update(wAeff);

    _phiA->update((*_qA),(*_qAc));
    _phiB->update((*_qB),(*_qBc));
    _phiS->update((1.0/_N)*(*_wS));
    _phiP->update((1.0/_N)*(*_wP));
    _phiN->update((1.0/_N)*(*_wN));

    // yita must be updated before wA,wB, because laters depend on yita.
    _yita->update((*_phiA)+(*_phiB)+(*_phiS)-1.0);
    if(_dielectric_constant == 1){
        Grid diff2 = _psi->diff2();
        _wA->update(_chiN*(*_phiB) + _chiASN*(*_phiS) + (*_yita) - \
                    0.5*_epsA*diff2);
        _wB->update(_chiN*(*_phiA) + _chiBSN*(*_phiS) + (*_yita) - \
                    0.5*_epsB*diff2);
        _wS->update(_chiASN*(*_phiA) + _chiBSN*(*_phiB) + (*_yita) -\
                    0.5*_epsS*diff2);
    }
    else{
        _wA->update(_chiN*(*_phiB) + _chiASN*(*_phiS) + (*_yita));
        _wB->update(_chiN*(*_phiA) + _chiBSN*(*_phiS) + (*_yita));
        _wS->update(_chiASN*(*_phiA) + _chiBSN*(*_phiB) + (*_yita));
    }

    Grid g(_upsP * (*_phiP) + _upsN * (*_phiN));

    // Smeared: 0; Annealed: 1
    if(_charge_distribution == 0){
        if(_upsA != 0) g += (_alphaA * _upsA) * (*_phiA);
        if(_upsB != 0) g += (_alphaB * _upsB) * (*_phiB);
    }
    else{
        if(_upsA != 0)
          g += _upsA * _alphaA * (*_phiA) / (_alphaA + (1. - _alphaA) * \
               exp(_upsA * (*_psi)));
        if(_upsB != 0)
          g += _upsB * _alphaB * (*_phiB) / (_alphaB + (1. - _alphaB) * \
               exp(_upsB * (*_psi)));
    }
/*
 * External charges
    int Lx=_wA->Ly();
    int Ly=_wA->Lz();
    g(Lx/5,Ly/5) += 16.0;
    g(Lx/5-1,Ly/5-1) += 4.0;
    g(Lx/5-1,Ly/5+1) += 4.0;
    g(Lx/5+1,Ly/5+1) += 4.0;
    g(Lx/5+1,Ly/5-1) += 4.0;
    g(Lx/2,Ly/3) -= 8.0;
    g(Lx/2,2*Ly/3) -= 8.0;
    g(Lx/3,Ly/2) -= 8.0;
    g(2*Lx/3,Ly/2) -= 8.0;
*/
    if(_dielectric_constant == 1)
        _psi->set_eps(_epsA * (*_phiA) + \
                      _epsB * (*_phiB) + \
                      _epsS * (*_phiS));

    _psi->update(-1.0 * g);
    _wP->update((_upsP*_N)*(*_psi));
    _wN->update((_upsN*_N)*(*_psi));
}

double Model_ABSe::Hw() const{
    Grid g=_chiN * (*_phiA) * (*_phiB) + \
           _chiASN * (*_phiA) * (*_phiS) + \
           _chiBSN * (*_phiB) * (*_phiS) + \
           (*_wA) * (*_phiA) - \
           (*_wB) * (*_phiB) - \
           (*_wS) * (*_phiS);

    if(_dielectric_constant == 0)
        g -= 0.5 * _epsS * _psi->diff2();
    else
        g -= 0.5 * (_epsA * (*_phiA) + \
                    _epsB * (*_phiB) + \
                    _epsS * (*_phiS)) * \
                   _psi->diff2();

    return g.mean();
}

double Model_ABSe::Hs() const{
    double hs=0.0;
    hs -= _phiC_avg*log(_qB->Qt()/_phiC_avg);
    if(_phiS_avg>0) hs -= _N*_phiS_avg*log(_wS->Q(_N)/_phiS_avg);
    if(_phiP_avg>0) hs -= _N*_phiP_avg*log(_wP->Q(_N)/_phiP_avg);
    if(_phiN_avg>0) hs -= _N*_phiN_avg*log(_wN->Q(_N)/_phiN_avg);
    return hs;
}

double Model_ABSe::incomp() const{
    Grid g=(*_phiA)+(*_phiB)+(*_phiS)-1.0;
    return g.abs_mean();
}

double Model_ABSe::residual_error() const{
    double res=0.0;
    double r,x,b;
    Grid g1,g2,diff2;

    if(_dielectric_constant == 1) diff2 = _psi->diff2();
    // wA
    if(_dielectric_constant == 1)
        g1 = _chiN*(*_phiB) + _chiASN*(*_phiS) + (*_yita) - \
             0.5*_epsA*diff2;
    else
        g1 = _chiN*(*_phiB) + _chiASN*(*_phiS) + (*_yita);
    g2 = g1 - (*_wA);
    r = g2.abs_mean();
    x = _wA->abs_mean();
    b = g1.abs_mean();
    res += r / (x + b);
    // wB
    if(_dielectric_constant == 1)
        g1 = _chiN*(*_phiA) + _chiBSN*(*_phiS) + (*_yita) - \
             0.5*_epsB*diff2;
    else
        g1 = _chiN*(*_phiA) + _chiBSN*(*_phiS) + (*_yita);
    g2 = g1 - (*_wB);
    r = g2.abs_mean();
    x = _wB->abs_mean();
    b = g1.abs_mean();
    res += r / (x + b);
    // wS
    if(_dielectric_constant == 1)
        g1 = _chiASN*(*_phiA) + _chiBSN*(*_phiB) + (*_yita) - \
             0.5*_epsS*diff2;
    else
        g1 = _chiASN*(*_phiA) + _chiBSN*(*_phiB) + (*_yita);
    g2 = g1 - (*_wS);
    r = g2.abs_mean();
    x = _wS->abs_mean();
    b = g1.abs_mean();
    res += r / (x + b);
    // Yita
    g1 = (*_phiA) + (*_phiB) + (*_phiS);
    g2 = g1 - 1.0;
    r = g2.abs_mean();
    x = 1.0;
    b = g1.abs_mean();
    res += r / (x + b);

    return res/4;
}

/*
double Model_ABSe::density_error() const{
    double err;
    double errA=0.0;
    double errB=0.0;
    Grid g=(*_phiA)-(*_phiA0);
    errA = g.abs_mean();
    g=(*_phiB)-(*_phiB0);
    errB = g.abs_mean();
    err=errA;
    if(err<errB) err=errB;
    return err;
}
*/

void Model_ABSe::display() const{
    cout<<"\tUnitCell: "<<_wA->uc().type()<<endl;
    cout.setf(ios::fixed,ios::floatfield);
    cout.precision(4);
    cout<<"\t(lx,ly,lz)  = ";
    cout<<_wA->lx()<<","<<_wA->ly()<<","<<_wA->lz()<<endl;

    cout.setf(ios::fixed,ios::floatfield);
    cout.precision(6);
    cout<<"\tH    = "<<H()<<" = "<<Hw()<<Hs()<<endl; //Hs is always negative

    cout.setf(ios::scientific,ios::floatfield);
    cout.precision(2);
    cout<<"\tIncompressibility = "<<incomp()<<endl;
    cout<<"\tResidual Error    = "<<residual_error()<<endl;

    cout.setf(ios::fixed,ios::floatfield);
    cout.precision(4);
    cout<<"\tphiA = "<<_phiA->mean();
    cout<<"\t["<<_phiA->min()<<","<<_phiA->max()<<"]"<<endl;
    cout<<"\tphiB = "<<_phiB->mean();
    cout<<"\t["<<_phiB->min()<<","<<_phiB->max()<<"]"<<endl;
    cout<<"\tphiS = "<<_phiS->mean();
    cout<<"\t["<<_phiS->min()<<","<<_phiS->max()<<"]"<<endl;
    cout<<"\tphiP = "<<_phiP->mean();
    cout<<"\t["<<_phiP->min()<<","<<_phiP->max()<<"]"<<endl;
    cout<<"\tphiN = "<<_phiN->mean();
    cout<<"\t["<<_phiN->min()<<","<<_phiN->max()<<"]"<<endl;

    cout<<"\twA   = "<<_wA->mean();
    cout<<"\t["<<_wA->min()<<","<<_wA->max()<<"]"<<endl;
    cout<<"\twB   = "<<_wB->mean();
    cout<<"\t["<<_wB->min()<<","<<_wB->max()<<"]"<<endl;
    cout<<"\twS   = "<<_wS->mean();
    cout<<"\t["<<_wS->min()<<","<<_wS->max()<<"]"<<endl;
    cout<<"\twP   = "<<_wP->mean();
    cout<<"\t["<<_wP->min()<<","<<_wP->max()<<"]"<<endl;
    cout<<"\twN   = "<<_wN->mean();
    cout<<"\t["<<_wN->min()<<","<<_wN->max()<<"]"<<endl;
    cout<<"\tpsi  = "<<_psi->mean();
    cout<<"\t["<<_psi->min()<<","<<_psi->max()<<"]"<<endl;

    cout.unsetf(ios::floatfield);
    cout.precision(6);
}

void Model_ABSe::save_model(const string file){
    CMatFile mat;
    mat.matInit(file,"u");
    if(!mat.queryStatus()){
        mat.matPutScalar("number_of_component",_number_of_component);
        mat.matPutScalar("NA",_NA);
        mat.matPutScalar("NB",_NB);
        mat.matPutScalar("N",_N);
        mat.matPutScalar("fA",_fA);
        mat.matPutScalar("fB",_fB);
        mat.matPutScalar("a",_a);
        mat.matPutScalar("Rg",_Rg);
        mat.matPutScalar("chiAB",_chiAB);
        mat.matPutScalar("chiN",_chiN);
        mat.matPutScalar("chiASN",_chiASN);
        mat.matPutScalar("chiBSN",_chiBSN);
        mat.matPutScalar("phiC",_phiC_avg);
        mat.matPutScalar("phiA_avg",_phiA_avg);
        mat.matPutScalar("phiB_avg",_phiB_avg);
        mat.matPutScalar("charge_distribution_type",_charge_distribution);
        mat.matPutScalar("dielectric_constant_type",_dielectric_constant);
        mat.matPutScalar("density_integration_type",_density_integration);
        mat.matPutScalar("cs",_cs);
        mat.matPutScalar("alphaA",_alphaA);
        mat.matPutScalar("alphaB",_alphaB);
        mat.matPutScalar("upsA",_upsA);
        mat.matPutScalar("upsB",_upsB);
        mat.matPutScalar("upsP",_upsP);
        mat.matPutScalar("upsN",_upsN);
        mat.matPutScalar("epsA",_epsA);
        mat.matPutScalar("epsB",_epsB);
        mat.matPutScalar("epsS",_epsS);
        mat.matPutScalar("phiS_avg",_phiS_avg);
        mat.matPutScalar("phiP_avg",_phiP_avg);
        mat.matPutScalar("phiN_avg",_phiN_avg);
        mat.matPutScalar("Ms",_Ms);
        mat.matPutScalar("ds",_ds);
        mat.matPutScalar("sA",_sA);
        mat.matPutScalar("sB",_sB);
        mat.matPutScalar("seedA",_wA->seed());
        mat.matPutScalar("seedB",_wB->seed());

        mat.matPutScalar("dim",_wA->dim());
        mat.matPutScalar("Lx",_wA->Lx());
        mat.matPutScalar("Ly",_wA->Ly());
        mat.matPutScalar("Lz",_wA->Lz());
        mat.matPutScalar("lx",_wA->lx());
        mat.matPutScalar("ly",_wA->ly());
        mat.matPutScalar("lz",_wA->lz());
        mat.matPutString("crystal_system_type",_wA->uc().type());
        //mat.matPutString("gridType",GridTypes[_wA->grid_type()]);
        mat.matPutString("gridInitType",GridInitTypes[_wA->grid_init_type()]);
        //mat.matPutString("phasePattern",PhasePatterns[_wA->phase_pattern()]);
        mat.matRelease();
    }
}

void Model_ABSe::release_memory(){
    delete _wA;
    delete _wB;
    delete _wS;
    delete _wP;
    delete _wN;
    delete _yita;
    delete _psi;
    delete _phiA;
    delete _phiB;
    delete _phiS;
    delete _phiP;
    delete _phiN;
    delete _qA;
    delete _qB;
    delete _qAc;
    delete _qBc;
}

void Model_ABSe::init_field(const Config &cfg){
    double lamA=cfg.get_double("Grid","lamA");
    double lamB=cfg.get_double("Grid","lamB");
    double lamS=cfg.get_double("Grid","lamS");
    double lamP=cfg.get_double("Grid","lamP");
    double lamN=cfg.get_double("Grid","lamN");
    double lamYita=cfg.get_double("Grid","lamYita");
    double lamPsi=cfg.get_double("Grid","lamPsi");

    _wS=new Field("wS",cfg,lamS);
    _wP=new Field("wP",cfg,lamP);
    _wN=new Field("wN",cfg,lamN);
    _yita=new Yita("yita",cfg,lamYita);

    if(_dielectric_constant == 0){
        // default Updater invoked
        // 2D, mud2sp, for u_xx + u_yy = f
        // 3D, mud3sp, for u_xx + u_yy + u_zz = f
        _psi = new FieldE("psi",_dielectric_constant,cfg,lamPsi,_N/_epsS);
    }
    else{
        _psi = new FieldE("psi",_dielectric_constant,cfg,lamPsi,_N);
    }

    switch(cfg.get_grid_init_type()){
        case RANDOM_INIT:
        {
            double low = cfg.get_double("Grid","v1");
            double high = cfg.get_double("Grid","v2");
            int seed = cfg.get_double("Model","seed");
            _wA=new Field("wA",cfg,lamA,low,high,seed);
            _wB=new Field("wB",cfg,lamB,low,high,seed);
        } // brackets are needed to declare variable inside case
            break;
        case CONSTANT_INIT:
        {
            double vA=cfg.get_double("Grid","v1");
            double vB=cfg.get_double("Grid","v2");
            _wA=new Field("wA",cfg,vA,lamA);
            _wB=new Field("wB",cfg,vB,lamB);
        }
            break;
        case FILE_INIT:
        {
            string file = cfg.get_string("Grid","field_data");
            _wA=new Field("wA",cfg,file,lamA);
            _wB=new Field("wB",cfg,file,lamB);
        }
            break;
        case PATTERN_INIT:
        {
            double c=_fA<_fB?_fA:_fB;
            double v1=cfg.get_double("Grid","v1");
            double v2=cfg.get_double("Grid","v2");
            PhasePattern pt=cfg.get_phase_pattern();
            _wA=new Field("wA",cfg,lamA);
            _wB=new Field("wB",cfg,lamB);
            Helper::init_pattern((*_wA),pt,c,v1,v2);
            Helper::init_pattern((*_wB),pt,c,v2,v1);
        }
            break;
        default:
            cout<<"Unkonwn or unsupported grid init type!"<<endl;
            exit(1);
    }
}

void Model_ABSe::init_density(const Config &cfg){
    if(_density_integration == 1 && (_sA-1)%2 == 0)
        _phiA = new Density("phiA",cfg,new Simpson);
    else
        _phiA = new Density("phiA",cfg);

    if(_density_integration == 1 && (_sB-1)%2 == 0)
        _phiB = new Density("phiB",cfg,new Simpson);
    else
        _phiB = new Density("phiB",cfg);

    _phiS = new DensitySmall("phiS",cfg,_phiS_avg);
    _phiP = new DensitySmall("phiP",cfg,_phiP_avg);
    _phiN = new DensitySmall("phiN",cfg,_phiN_avg);
}

void Model_ABSe::init_propagator(const Config &cfg){
    Grid one(UnitCell(cfg),_wA->Lx(),_wA->Ly(),_wA->Lz(),1.0);
    _qA=new Propagator("qA",cfg,_sA,_ds,one);
    _qB=new Propagator("qB",cfg,_sB,_ds);
    _qAc=new Propagator("qAc",cfg,_sA,_ds);
    _qBc=new Propagator("qBc",cfg,_sB,_ds,one);
}

