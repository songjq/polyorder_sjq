#include "Model_AB.h"
#include "CMatFile.h"
#include "Helper.h"

using namespace arma;
using namespace std;
using namespace blitz;

Model_AB::Model_AB(const string config_file):Model(config_file){
    init();
}

void Model_AB::reset(const string& config_data){
    release_memory();
    bool rc = _cfg.set_config_data(config_data);
    if(!rc){
        cerr<<"Setting config data failed. Abort."<<endl;
        exit(1);
    }
    init();
}

void Model_AB::init(){
    is_compressible = _cfg.is_compressible();

    vec vf = _cfg.f();
    f_brush = vf(0);
    fA = vf(1);
    fB = vf(2);
    fB = 1.0 - fA;
    vec vchiN = _cfg.chiN();
    chiNab = vchiN(0);
    chiNac = vchiN(1);
    chiNbc = vchiN(2);
    vec va = _cfg.segment_length();
    aA = va(0);
    aB = va(1);
    aC = va(2);

    vec vds = _cfg.ds();
    dsA = vds(0);
    dsB = vds(1);
    dsC = vds(2);

    umat vMs = _cfg.Ms();
    sA = vMs(0);
    sB = vMs(1);
    sC = vMs(2);
    dsA = 1.0/(sA-1);
    dsB = 1.0/(sB-1);
    dsC = 1.0/(sC-1);
    
    switch (_cfg.dim()) {
        case 1 :
            graft_density = f_brush * _cfg.a() / sC;  // default the length of brush chain is sC
            break;
        case 2 :
            graft_density = f_brush * _cfg.b() / sC;
            break;
        case 3 :
            graft_density = f_brush * _cfg.c() / sC;
            break;
        default :
            cout << "Please input correct dimension !" << endl;
            break;
    }
    //graft_density = fB * _cfg.Ly() / sB;
    cout << "graft_density is  = " << graft_density << endl;
    //uword LA = sA - 1;
    //uword LB = sB - 1;

    //fA = 1.0 * LA / (LA + LB); 
    //fA = 1.0 * LA / (LA + LB);


    /*uword LA = fA / dsA;
    sA = LA + 1;
    uword LB = fB / dsB;
    sB = LB + 1;

    fA = 1.0 * LA / (LA + LB);
    fB = 1.0 * LB / (LA + LB);

    fA_shiftvalue = _cfg.get_double("Model","fA_shiftvalue");   //added by songjq in 20141103 
    fA = fA + fA_shiftvalue;                                   //added by songjq in 20141103
    fB = 1 - fA;                                               //added by songjq in 20141103
    dsA = fA / LA;                                             //added by songjq in 20141103
    dsB = fB / LB; */                                           //added by songjq in 20141103

    vec lam = _cfg.lam();
    lamA = lam(0);
    lamB = lam(1);
    lamC = lam(2);
    lamYita = lam(3);

    init_field();
    init_density();
    init_propagator();
}

void Model_AB::update(){
    blitz::Range all = blitz::Range::all();
    int L_graft = 0;
    uword Lx = _cfg.Lx();
    uword Ly = _cfg.Ly();
    uword Lz = _cfg.Lz();
    blitz::Array<double, 4> qc(qC->qs());
    blitz::Array<double, 4> qcc(qCc->qs());
    if(_cfg.algo_scft_type() == AlgorithmSCFTType::ANDERSON){
        qA->update(*wAx);
        qB->set_head( qA->get_tail() );
        qB->update(*wBx);

        qBc->update(*wBx);
        qAc->set_head( qBc->get_tail() );
        qAc->update(*wAx);

        qCc->update(*wCx); 
        switch (_cfg.dim()) {
        case 1 :
            L_graft = (int) (_cfg.graft_area() / _cfg.a() * Lx);
            qc(0, blitz::Range(0, L_graft-1), all, all) = graft_density / qcc(sC-1, blitz::Range(0, L_graft-1), all, all);  
            qc(0, blitz::Range(L_graft-2, Lx-L_graft-1), all, all) = 0.0;    
            qc(0, blitz::Range(Lx-L_graft, Lx-1), all, all) = graft_density / qcc(sC-1, blitz::Range(Lx-L_graft, Lx-1), all, all);
            break;
        case 2 :
            L_graft = (int) (_cfg.graft_area() / _cfg.b() * Ly);
            qc(0, all, blitz::Range(0, L_graft-1), all) = graft_density / qcc(sC-1, all, blitz::Range(0, L_graft-1), all);  
            qc(0, all, blitz::Range(L_graft-2, Ly-L_graft-1), all) = 0.0;    
            qc(0, all, blitz::Range(Ly-L_graft, Lx-1), all) = graft_density / qcc(sC-1, all, blitz::Range(Ly-L_graft, Lx-1), all);
            break;
        case 3 :
            L_graft = (int) (_cfg.graft_area() / _cfg.c() * Lz);
            qc(0, all, all, blitz::Range(0, L_graft-1)) = graft_density / qcc(sC-1, all, all, blitz::Range(0, L_graft-1));  
            qc(0, all, all, blitz::Range(L_graft-2, Lz-L_graft-1)) = 0.0;    
            qc(0, all, all, blitz::Range(Lz-L_graft, Lz-1)) = graft_density / qcc(sC-1, all, all, blitz::Range(Lz-L_graft, Lz-1));
            break;
        default :
            cout << "Please input correct dimension !" << endl;
            break;
        }    
        qC->update(*wCx);
    }
    else{
        qA->update(*wA);
        qB->set_head( qA->get_tail() );
        qB->update(*wB);

        qBc->update(*wB);
        qAc->set_head( qBc->get_tail() );
        qAc->update(*wA);

        qCc->update(*wC);
        switch (_cfg.dim()) {
        case 1 :
            L_graft = (int) (_cfg.graft_area() / _cfg.a() * Lx);
            qc(0, blitz::Range(0, L_graft-1), all, all) = graft_density / qcc(sC-1, blitz::Range(0, L_graft-1), all, all);  
            qc(0, blitz::Range(L_graft-2, Lx-L_graft-1), all, all) = 0.0;    
            qc(0, blitz::Range(Lx-L_graft, Lx-1), all, all) = graft_density / qcc(sC-1, blitz::Range(Lx-L_graft, Lx-1), all, all);
            break;
        case 2 :
            L_graft = (int) (_cfg.graft_area() / _cfg.b() * Ly);
            qc(0, all, blitz::Range(0, L_graft-1), all) = graft_density / qcc(sC-1, all, blitz::Range(0, L_graft-1), all);  
            qc(0, all, blitz::Range(L_graft-2, Ly-L_graft-1), all) = 0.0;    
            qc(0, all, blitz::Range(Ly-L_graft, Lx-1), all) = graft_density / qcc(sC-1, all, blitz::Range(Ly-L_graft, Lx-1), all);
            break;
        case 3 :
            L_graft = (int) (_cfg.graft_area() / _cfg.c() * Lz);
            qc(0, all, all, blitz::Range(0, L_graft-1)) = graft_density / qcc(sC-1, all, all, blitz::Range(0, L_graft-1));  
            qc(0, all, all, blitz::Range(L_graft-2, Lz-L_graft-1)) = 0.0;    
            qc(0, all, all, blitz::Range(Lz-L_graft, Lz-1)) = graft_density / qcc(sC-1, all, all, blitz::Range(Lz-L_graft, Lz-1));
            break;
        default :
            cout << "Please input correct dimension !" << endl;
            break;
        }   
        qC->update(*wC);
    }

    double Q = qB->Qt();
    phiA->set_cc(1.0/Q*fA*(1-f_brush));
    phiA->update(*qA, *qAc);
    phiB->set_cc(1.0/Q*fB*(1-f_brush));
    phiB->update(*qB, *qBc);

    double QC = qC->Qt();
    phiC->set_cc(1.0/QC*f_brush);
    phiC->update(*qC, *qCc);

    /**
     * It is essential to combine A and B blocks to prodcue
     * Anderson mixing coefficient C.
     * Updating by Anderson mixing for A and B separately fails.
     */
    if(is_compressible){                                              //this part could not be used
        Grid eta = lamYita * ((*phiA) + (*phiB) + (*phiC)- 1.0);
        if(_cfg.algo_scft_type() == AlgorithmSCFTType::ANDERSON){
            wAx->pre_update(chiNab * (*phiB) + chiNac * (*phiC) + eta);
            wBx->pre_update(chiNab * (*phiA) + chiNbc * (*phiC) + eta);
            wCx->pre_update(chiNac * (*phiA) + chiNbc * (*phiB) + eta);
            mat UA = wAx->calc_U();
            vec VA = wAx->calc_V();
            mat UB = wBx->calc_U();
            vec VB = wBx->calc_V();
            mat U = UA + UB;
            vec V = VA + VB;
            //U.print("U =");
            //V.print("V =");
            vec C;
            if(!U.is_empty() and !V.is_empty())
                C = solve(U, V);
            //C.print("C =");
            wAx->update(C);
            wBx->update(C);
        }
        /*else{
            wA->update( chiN * (*phiB) + eta );
            wB->update( chiN * (*phiA) + eta );
        }*/
    }
    else{                                             //this part could not be used
        if(_cfg.algo_scft_type() == AlgorithmSCFTType::ANDERSON){
            //(*yita) = 0.5 * ((*wAx) + (*wBx) - chiN);
            //cout<<"wAx k = "<<wAx->n_iteration()<<endl;
            //cout<<"wAx cur_pos = "<<wAx->current_index()<<endl;
            //cout<<"wBx k = "<<wBx->n_iteration()<<endl;
            //cout<<"wBx cur_pos = "<<wBx->current_index()<<endl;
            //wAx->pre_update(chiN * (*phiB) + (*yita));
            //wBx->pre_update(chiN * (*phiA) + (*yita));
            mat UA = wAx->calc_U();
            vec VA = wAx->calc_V();
            mat UB = wBx->calc_U();
            vec VB = wBx->calc_V();
            mat U = UA + UB;
            vec V = VA + VB;
            //U.print("U =");
            //V.print("V =");
            vec C;
            if(!U.is_empty() and !V.is_empty())
                C = solve(U, V);
            //C.print("C =");
            wAx->update(C);
            wBx->update(C);
        }
        else{
            // For good convergence, yita must be updated before wA and wB.
            yita->update( (*phiA) + (*phiB) + (*phiC) - 1.0 );
            wA->update( chiNab * (*phiB) + chiNac * (*phiC) + (*yita) );
            wB->update( chiNab * (*phiA) + chiNbc * (*phiC) + (*yita) );
            wC->update( chiNac * (*phiA) + chiNbc * (*phiB) + (*yita) );
        }
    }
}

double Model_AB::Hw() const{
    double ret;
    if(is_compressible){               //this part could not be used
        /*if(_cfg.algo_scft_type() == AlgorithmSCFTType::ANDERSON){
            Grid g = chiN * (*phiA) * (*phiB) +
                     lamYita * ((*phiA) + (*phiB) - 1.0) -
                     (*wAx) * (*phiA) - (*wBx) * (*phiB);
            ret = g.quadrature();
        }
        else{
            Grid g = chiN * (*phiA) * (*phiB) +
                     lamYita * ((*phiA) + (*phiB) - 1.0) -
                     (*wA) * (*phiA) - (*wB) * (*phiB);
            ret = g.quadrature();
        }*/
    }
    else{                                //this part could not be used
        if(_cfg.algo_scft_type() == AlgorithmSCFTType::ANDERSON){
            /*Grid g = chiN * (*phiA) * (*phiB) -
                     (*wAx) * (*phiA) - (*wBx) * (*phiB);
            ret = g.quadrature();*/
        }
        else{
            Grid g = chiNab * (*phiA) * (*phiB) + chiNac * (*phiA) * (*phiC) +
                     chiNbc * (*phiB) * (*phiC) - (*wA) * (*phiA) - (*wB) * (*phiB) - (*wC) * (*phiC);
            ret = g.quadrature();
        }
    }
    return ret;
}

double Model_AB::Hs() const{
    return -(1-f_brush)*log(qB->Qt())-f_brush*log(qC->Qt());
}

double Model_AB::H() const{
    return Hw() + Hs();
}

double Model_AB::incomp() const{
    Grid g = (*phiA) + (*phiB) + (*phiC) - 1.0;
    return g.abs_quadrature();
}

double Model_AB::residual_error() const{
    double res = 0.0;
    if(_cfg.algo_scft_type() == AlgorithmSCFTType::ANDERSON){  //this part could not be used
        /* For FieldAX ONLY */
        double eA1 = wAx->current_d2_quadrature();
        double eA2 = wAx->current_w2_quadrature();
        double eB1 = wBx->current_d2_quadrature();
        double eB2 = wBx->current_w2_quadrature();
        res = sqrt((eA1+eB1)/(eA2+eB2));
        /*
        double eA1 = wAx->current_d_abs_quadrature();
        double eA2 = wAx->current_w_abs_quadrature();
        double eB1 = wBx->current_d_abs_quadrature();
        double eB2 = wBx->current_w_abs_quadrature();
        res = (eA1+eB1) / (eA2+eB2);
        */
    }
    else{
        /* For Field ONLY */
        double r, x, b;
        Grid g1, g2;
        if(!is_compressible){
            // wA
            g1 = chiNab * (*phiB) +chiNac * (*phiC) + (*yita);
            g2 = g1 - (*wA);
            r = g2.abs_mean();
            x = wA->abs_mean();
            b = g1.abs_mean();
            res += r / (x+b);
            // wB
            g1 = chiNab * (*phiA) + chiNbc * (*phiC) + (*yita);
            g2 = g1 - (*wB);
            r = g2.abs_mean();
            x = wB->abs_mean();
            b = g1.abs_mean();
            res += r / (x+b);
            //wC
            g1 = chiNac * (*phiA) + chiNbc * (*phiB) + (*yita);
            g2 = g1 - (*wC);
            r = g2.abs_mean();
            x = wC->abs_mean();
            b = g1.abs_mean();
            res += r / (x+b);
            // Yita
            g1 = (*phiA) + (*phiB) + (*phiC);
            g2 = g1 - 1.0;
            r = g2.abs_mean();
            x = 1.0;
            b = g1.abs_mean();
            res += r / (x+b);
            res /= 4.0;
        }
        /*else{                         //this part could not be used
            // wA
            g1 = chiN * (*phiB) + lamYita * ((*phiA) + (*phiB) - 1.0);
            g2 = g1 - (*wA);
            r = g2.abs_quadrature();
            x = wA->abs_quadrature();
            b = g1.abs_quadrature();
            res += r / (x+b);
            // wB
            g1 = chiN * (*phiA) + lamYita * ((*phiA) + (*phiB) - 1.0);
            g2 = g1 - (*wB);
            r = g2.abs_quadrature();
            x = wB->abs_quadrature();
            b = g1.abs_quadrature();
            res += r / (x+b);
            res /= 2.0;
        }*/
    }
    return res;
}

/*
double Model_AB::density_error() const{
    double err=0.0;
    Grid g=(*_phiA)-(*_phiA0);
    err=g.abs_mean();
    g=(*_phiB)-(*_phiB0);
    err += g.abs_mean();
    return err/2;
}*/

void Model_AB::display() const{
    cout<<"\tUnit Cell: "<<phiA->uc().type()<<endl;
    cout.setf(ios::fixed, ios::floatfield);
    cout.precision(4);
    cout<<"\t(lx,ly,lz) = ";
    cout<<phiA->lx()<<","<<phiA->ly()<<","<<phiA->lz()<<endl;

    cout.setf(ios::fixed, ios::floatfield);
    cout.precision(6);
    cout<<"\tH    = "<<H()<<" = "<<Hw()<<" + "<<Hs()<<endl;

    cout.setf(ios::fixed, ios::floatfield);
    cout.precision(4);
    cout<<"\tphiA = "<<phiA->quadrature();
    cout<<"\t["<<phiA->min()<<", "<<phiA->max()<<"]"<<endl;
    cout<<"\tphiB = "<<phiB->quadrature();
    cout<<"\t["<<phiB->min()<<", "<<phiB->max()<<"]"<<endl;
    cout<<"\tphiC = "<<phiC->quadrature();
    cout<<"\t["<<phiC->min()<<", "<<phiC->max()<<"]"<<endl;

    if(_cfg.algo_scft_type() == AlgorithmSCFTType::ANDERSON){  //this part could not be used
        cout<<"\twA   = "<<wAx->quadrature();
        cout<<"\t["<<wAx->min()<<", "<<wAx->max()<<"]"<<endl;
        cout<<"\twB   = "<<wBx->quadrature();
        cout<<"\t["<<wBx->min()<<", "<<wBx->max()<<"]"<<endl;
    }
    else{
        cout<<"\twA   = "<<wA->quadrature();
        cout<<"\t["<<wA->min()<<", "<<wA->max()<<"]"<<endl;
        cout<<"\twB   = "<<wB->quadrature();
        cout<<"\t["<<wB->min()<<", "<<wB->max()<<"]"<<endl;
        cout<<"\twC   = "<<wC->quadrature();
        cout<<"\t["<<wC->min()<<", "<<wC->max()<<"]"<<endl;
    }
    if(!is_compressible){
        cout<<"\tyita = "<<yita->quadrature();
        cout<<"\t["<<yita->min()<<", "<<yita->max()<<"]"<<endl;
    }

    cout.setf(ios::scientific, ios::floatfield);
    cout.precision(2);
    //cout<<"\tIncompressibility = "<<incomp()<<endl;
    cout<<"\tQ_qB = "<<qB->Qt();
    cout<<"\tQ_qAc = "<<qAc->Qt()<<endl;
    cout<<"\tQ_qC = "<<qC->Qt();
    cout<<"\tQ_qCc = "<<qCc->Qt()<<endl;
    cout<<"\tResidual Error    = "<<residual_error()<<endl;
    cout<<"\tphiA+phiB+phiC-1.0     = "<<(phiA->quadrature()+phiB->quadrature()+phiC->quadrature()-1.0) << endl;

    cout.unsetf(ios::floatfield);
    cout.precision(6);
}

void Model_AB::save(const string file){
    phiA->uc().save(file, phiA->Lx(), phiA->Ly(), phiA->Lz());
    save_field(file);
    save_density(file);
}

void Model_AB::save_field(const string file){
    if(_cfg.algo_scft_type() == AlgorithmSCFTType::ANDERSON){  //this part could not be used
        wAx->save(file);
        wBx->save(file);
    }
    else{
        wA->save(file);
        wB->save(file);
        wC->save(file);
    }
    if(!is_compressible)
        yita->save(file);
}

void Model_AB::save_density(const string file){
    phiA->save(file);
    phiB->save(file);
    phiC->save(file);
}

void Model_AB::save_q(const string file){
    qA->save(file);
    qAc->save(file);
    qB->save(file);
    qBc->save(file);
    qC->save(file);
    qCc->save(file);
}

void Model_AB::save_model(const string file){
    CMatFile mat;
    mat.matInit(file,"u");
    if(!mat.queryStatus()){
        mat.matPutScalar("f_brush", f_brush);
        mat.matPutScalar("fA", fA);
        mat.matPutScalar("fB", fB);
        mat.matPutScalar("aA", aA);
        mat.matPutScalar("aB", aB);
        mat.matPutScalar("aC", aC);
        mat.matPutScalar("chiNab", chiNab);
        mat.matPutScalar("chiNac", chiNac);
        mat.matPutScalar("chiNbc", chiNbc);
        mat.matPutScalar("dsA", dsA);
        mat.matPutScalar("dsB", dsB);
        mat.matPutScalar("dsC", dsC);
        mat.matPutScalar("sA", sA);
        mat.matPutScalar("sB", sB);
        mat.matPutScalar("sC", sC);
        if(_cfg.algo_scft_type() == AlgorithmSCFTType::ANDERSON){  //this part could not be used
            mat.matPutScalar("seedA", wAx->seed());
            mat.matPutScalar("seedB", wBx->seed());
        }
        else{
            mat.matPutScalar("seedA", wA->seed());
            mat.matPutScalar("seedB", wB->seed());
            mat.matPutScalar("seedC", wC->seed());
        }

        mat.matPutScalar("dim", _cfg.dim());
        mat.matPutScalar("Lx", phiA->Lx());
        mat.matPutScalar("Ly", phiA->Ly());
        mat.matPutScalar("Lz", phiA->Lz());
        mat.matPutScalar("lx", phiA->lx());
        mat.matPutScalar("ly", phiA->ly());
        mat.matPutScalar("lz", phiA->lz());
        mat.matPutString("crystal_system_type", phiA->uc().type());
        mat.matPutString("gridInitType",_cfg.get_grid_init_type_string());
        mat.matRelease();
    }
}

void Model_AB::display_parameters() const{
    cout<<endl;
    cout<<"********* Model_AB Parameter List **********"<<endl;
    cout<<"Compressibility: ";
    if(is_compressible)
        cout<<"Helfand compressible model."<<endl;
    else
        cout<<"Incompressible model."<<endl;
    cout<<"Confinement: "<<_cfg.get_confine_type_string()<<endl;
    cout<<"MDE algorithm: "<<_cfg.get_algo_mde_type_string()<<endl;
    cout<<"SCFT algorithm: "<<_cfg.get_algo_scft_type_string()<<endl;
    cout<<"Cell optimization algorithm: ";
    cout<<_cfg.get_algo_cell_optimization_type_string()<<endl;
    cout<<"Contour integration algorithm: ";
    cout<<_cfg.get_algo_contour_integration_type_string()<<endl;
    cout<<endl;

    cout<<"f_brush = "<<f_brush<<"\tfA = "<<fA<<"\tfB = "<<fB<<endl;
    cout<<"chiNab = "<<chiNab<<"\tchiNac = "<<chiNac<<"\tchiNbc = "<<chiNbc<<endl;
    cout<<"aA = "<<aA<<"\taB = "<<aB<<"\taC = "<<aC<<endl;
    cout<<"dsA = "<<dsA<<"\tdsB = "<<dsB<<"\tdsC = "<<dsC<<endl;
    cout<<"sA = "<<sA<<"\tsB = "<<sB<<"\tsC = "<<sC<<endl;
    if(_cfg.algo_scft_type() == AlgorithmSCFTType::ANDERSON)                //this part could not be used
        cout<<"seedA = "<<wAx->seed()<<"\tseedB = "<<wBx->seed()<<endl;
    else
        cout<<"seedA = "<<wA->seed()<<"\tseedB = "<<wB->seed()<<"\tseedC = "<<wC->seed()<<endl;
    cout<<endl;

    cout<<"dimension: "<<_cfg.dim()<<endl;
    cout<<"(Lx, Ly, Lz) = ";
    cout<<"("<<phiA->Lx()<<", "<<phiA->Ly()<<", "<<phiA->Lz()<<")"<<endl;
    cout<<"(a, b, c) = ";
    cout<<"("<<phiA->lx()<<", "<<phiA->ly()<<", "<<phiA->lz()<<")"<<endl;

    cout<<"*******************************************"<<endl;
    cout<<endl;
}

void Model_AB::release_memory(){
    if(_cfg.algo_scft_type() == AlgorithmSCFTType::ANDERSON){             
        delete wAx;
        delete wBx;
        delete wCx;
    }
    else{
        delete wA;
        delete wB;
        delete wC;
    }
    if(!is_compressible)
        delete yita;
    delete phiA;
    delete phiB;
    delete phiC;
    delete qA;
    delete qB;
    delete qC;
    delete qAc;
    delete qBc;
    delete qCc;
    if(_cfg.ctype() != ConfineType::NONE
       || (_cfg.algo_mde_type() == AlgorithmMDEType::ETDRK4
           && _cfg.etdrk4_M() > 0)){
        delete ppropupA;
        delete ppropupB;
        delete ppropupC;
    }
}

Model_AB::~Model_AB(){
    release_memory();
}

void Model_AB::init_random_field(){
    double low = 0.0;
    double high = 1.0;
    int seed = _cfg.seed();
    if(_cfg.algo_scft_type() == AlgorithmSCFTType::ANDERSON){
        uword n = _cfg.n_Anderson_mixing();
        wAx = new FieldAX("wA", _cfg, lamA, n, low, high, seed);
        wBx = new FieldAX("wB", _cfg, lamB, n, low, high, seed);
        wCx = new FieldAX("wC", _cfg, lamB, n, low, high, seed);
    }
    else{
        wA = new Field("wA", _cfg, lamA, low, high, seed);
        wB = new Field("wB", _cfg, lamB, low, high, seed);
        wC = new Field("wC", _cfg, lamB, low, high, seed);
    }
}

void Model_AB::init_constant_field(){
    double vA = 0.5;
    double vB = 0.5;
    double vC = 0.5;
    if(_cfg.algo_scft_type() == AlgorithmSCFTType::ANDERSON){
        uword n = _cfg.n_Anderson_mixing();
        wAx = new FieldAX("wA", _cfg, vA, lamA, n);
        wBx = new FieldAX("wB", _cfg, vB, lamB, n);
        wCx = new FieldAX("wC", _cfg, vC, lamC, n);
    }
    else{
        wA = new Field("wA", _cfg, vA, lamA);
        wB = new Field("wB", _cfg, vB, lamB);
        wC = new Field("wC", _cfg, vB, lamC);
    }
}

void Model_AB::init_file_field(){
    string file = _cfg.field_data_file();
    //cout<<"init_data_file: "<<file<<endl;
    if(_cfg.algo_scft_type() == AlgorithmSCFTType::ANDERSON){
        uword n = _cfg.n_Anderson_mixing();
        wAx = new FieldAX("wA", _cfg, file, lamA, n);
        wBx = new FieldAX("wB", _cfg, file, lamB, n);
        wCx = new FieldAX("wC", _cfg, file, lamC, n);
    }
    else{
        wA = new Field("wA", _cfg, file, lamA);
        wB = new Field("wB", _cfg, file, lamB);
        wC = new Field("wC", _cfg, file, lamC);
    }
}

void Model_AB::init_pattern_field(){    //this part could not be used for time
    double c = fA<fB?fA:fB;
    double v1 = 0;
    double v2 = 1;
    PhasePattern pt = _cfg.get_phase_pattern();
    if(_cfg.algo_scft_type() == AlgorithmSCFTType::ANDERSON){
        uword n = _cfg.n_Anderson_mixing();
        wAx = new FieldAX("wA", _cfg, lamA, n);
        wBx = new FieldAX("wB", _cfg, lamB, n);
        Helper::init_pattern((*wAx), pt, c, v1, v2);
        Helper::init_pattern((*wBx), pt, c, v2, v1);
    }
    else{
        wA = new Field("wA", _cfg, lamA);
        wB = new Field("wB", _cfg, lamB);
        Helper::init_pattern((*wA), pt, c, v1, v2);
        Helper::init_pattern((*wB), pt, c, v2, v1);
    }
}

void Model_AB::init_field(){
    switch(_cfg.get_grid_init_type()){
        case GridInitType::RANDOM_INIT:
            init_random_field();
            break;
        case GridInitType::CONSTANT_INIT:
            init_constant_field();
            break;
        case GridInitType::FILE_INIT:
            init_file_field();
            break;
        case GridInitType::PATTERN_INIT:
            init_pattern_field();
            break;
        default:
            cerr<<"Unkonwn or unsupported grid init type!"<<endl;
            exit(1);
    }
    if(!is_compressible)
        yita = new Yita("yita", _cfg, lamYita);  // No special initialization
}

void Model_AB::init_density(){
    AlgorithmContourType actype = _cfg.algo_contour_integration_type();
    if(actype == AlgorithmContourType::TRAPEZOIDAL){
        phiA = new Density("phiA", _cfg);
        phiB = new Density("phiB", _cfg);
        phiC = new Density("phiC", _cfg);
    }
    else if(actype == AlgorithmContourType::SIMPSON){
        if(sA%2 != 0)
            phiA = new Density("phiA", _cfg, new Simpson);
        else
            phiA = new Density("phiA", _cfg, new Quad4_Closed);
        if(sB%2 != 0)
            phiB = new Density("phiB", _cfg, new Simpson);
        else
            phiB = new Density("phiB", _cfg, new Quad4_Closed);
        if(sC%2 != 0)
            phiC = new Density("phiC", _cfg, new Simpson);
        else
            phiC = new Density("phiC", _cfg, new Quad4_Closed);
    }
    else{
        cerr<<"Contour integration algorithm: ";
        cerr<<_cfg.get_algo_contour_integration_type_string();
        cerr<<" is not available."<<endl;
        exit(1);
    }
}

void Model_AB::init_propagator(){
    UnitCell uc(_cfg);
    ConfineType confine_type = _cfg.ctype();
    uword dim = _cfg.dim();
    uword Lx = _cfg.Lx();
    uword Ly = _cfg.Ly();
    uword Lz = _cfg.Lz();
    Grid one(uc, Lx, Ly, Lz, 1.0);

    if(confine_type == ConfineType::NONE){
        if(_cfg.algo_mde_type() == AlgorithmMDEType::ETDRK4
           && _cfg.etdrk4_M() <= 0){
            // NOTE: ONLY Cox-Matthews scheme has been implemented.
            // Thus the input _cfg.etdrk4_scheme_type() is ignored.
            // Etdrk4_PBC is the default algo for Propagator.
            qA = new Propagator("qA", _cfg, sA, dsA, one);
            qB = new Propagator("qB", _cfg, sB, dsB);
            qAc = new Propagator("qAc", _cfg, sA, dsA);
            qBc = new Propagator("qBc", _cfg, sB, dsB, one);
            qC = new Propagator("qC", _cfg, sC, dsC, one);
            qCc = new Propagator("qCc", _cfg, sC, dsC, one);
        }
        else if(_cfg.algo_mde_type() == AlgorithmMDEType::ETDRK4
                && _cfg.etdrk4_M() > 0){
            // NOTE: ONLY Cox-Matthews scheme has been implemented.
            // Thus the input _cfg.etdrk4_scheme_type() is ignored.
            ppropupA = new Etdrk4_PBC(uc, Lx, Ly, Lz, dsA, _cfg.etdrk4_M());
            ppropupB = new Etdrk4_PBC(uc, Lx, Ly, Lz, dsB, _cfg.etdrk4_M());
            ppropupC = new Etdrk4_PBC(uc, Lx, Ly, Lz, dsC, _cfg.etdrk4_M());
            qA = new Propagator("qA", _cfg, sA, dsA, one, ppropupA);
            qB = new Propagator("qB", _cfg, sB, dsB, ppropupB);
            qAc = new Propagator("qAc", _cfg, sA, dsA, ppropupA);
            qBc = new Propagator("qBc", _cfg, sB, dsB, one, ppropupB);
            qC = new Propagator("qC", _cfg, sC, dsC, one, ppropupC);
            qCc = new Propagator("qCc", _cfg, sC, dsC, one, ppropupC);
        }
        else if(_cfg.algo_mde_type() == AlgorithmMDEType::OS2){
            ppropupA = new PseudoSpectral(uc, Lx, Ly, Lz, dsA);
            ppropupB = new PseudoSpectral(uc, Lx, Ly, Lz, dsB);
            ppropupC = new PseudoSpectral(uc, Lx, Ly, Lz, dsC);
            qA = new Propagator("qA", _cfg, sA, dsA, one, ppropupA);
            qB = new Propagator("qB", _cfg, sB, dsB, ppropupB);
            qAc = new Propagator("qAc", _cfg, sA, dsA, ppropupA);
            qBc = new Propagator("qBc", _cfg, sB, dsB, one, ppropupB);
            qC = new Propagator("qC", _cfg, sC, dsC, one, ppropupC);
            qCc = new Propagator("qCc", _cfg, sC, dsC, one, ppropupC);
        }
        else if(_cfg.algo_mde_type() == AlgorithmMDEType::RQM4){
            ppropupA = new RQM4(uc, Lx, Ly, Lz, dsA);
            ppropupB = new RQM4(uc, Lx, Ly, Lz, dsB);
            ppropupC = new RQM4(uc, Lx, Ly, Lz, dsC);
            qA = new Propagator("qA", _cfg, sA, dsA, one, ppropupA);
            qB = new Propagator("qB", _cfg, sB, dsB, ppropupB);
            qAc = new Propagator("qAc", _cfg, sA, dsA, ppropupA);
            qBc = new Propagator("qBc", _cfg, sB, dsB, one, ppropupB);
            qC = new Propagator("qC", _cfg, sC, dsC, one, ppropupC);
            qCc = new Propagator("qCc", _cfg, sC, dsC, one, ppropupC);
        }
        else{
            cerr<<"MDE algorithm: "<<_cfg.get_algo_mde_type_string();
            cerr<<" is not available."<<endl;
            exit(1);
        }
    }
    else if(confine_type == ConfineType::CUBE){
        vec lbcc = _cfg.BC_coefficients_left();
        vec rbcc = _cfg.BC_coefficients_right();
        Boundary lbcA(lbcc(0), lbcc(1), lbcc(2));
        Boundary rbcA(rbcc(0), rbcc(1), rbcc(2));
        Boundary lbcB(lbcc(0), -fA/fB*lbcc(1), lbcc(2));
        Boundary rbcB(rbcc(0), -fA/fB*rbcc(1), rbcc(2));
        Boundary lbcC(lbcc(0), lbcc(1), lbcc(2));
        Boundary rbcC(rbcc(0), rbcc(1), rbcc(2));

        // NOTE: ONLY Krogstad scheme has been implemented.
        // Thus the input _cfg.etdrk4_scheme_type() is ignored.
        if(_cfg.etdrk4_M() <= 0){
            ppropupA = new Etdrk4(uc, dim, Lx, Ly, Lz, dsA,
                                  confine_type, lbcA, rbcA);
            ppropupB = new Etdrk4(uc, dim, Lx, Ly, Lz, dsB,
                                  confine_type, lbcB, rbcB);
            ppropupC = new Etdrk4(uc, dim, Lx, Ly, Lz, dsC,
                                  confine_type, lbcC, rbcC);

        }
        else{
            ppropupA = new Etdrk4(uc, dim, Lx, Ly, Lz, dsA,
                                  confine_type, lbcA, rbcA,
                                  ETDRK4SCHEME::KROGSTAD, _cfg.etdrk4_M());
            ppropupB = new Etdrk4(uc, dim, Lx, Ly, Lz, dsB,
                                  confine_type, lbcB, rbcB,
                                  ETDRK4SCHEME::KROGSTAD, _cfg.etdrk4_M());
            ppropupC = new Etdrk4(uc, dim, Lx, Ly, Lz, dsC,
                                  confine_type, lbcC, rbcC,
                                  ETDRK4SCHEME::KROGSTAD, _cfg.etdrk4_M());
        }
        qA = new Propagator("qA", _cfg, sA, dsA, one, ppropupA);
        qB = new Propagator("qB", _cfg, sB, dsB, ppropupB);
        qAc = new Propagator("qAc", _cfg, sA, dsA, ppropupA);
        qBc = new Propagator("qBc", _cfg, sB, dsB, one, ppropupB);
        qC = new Propagator("qC", _cfg, sC, dsC, ppropupC);
        qCc = new Propagator("qCc", _cfg, sC, dsC, one, ppropupC);
    }
    else{
        cerr<<"Confinement: "<<_cfg.get_confine_type_string();
        cerr<<" is not available."<<endl;
        exit(1);
    }
}

