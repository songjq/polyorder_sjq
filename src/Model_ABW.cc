#include "Model_ABW.h"
#include "CMatFile.h"
#include "Helper.h"

using namespace arma;
using namespace std;

Model_ABW::Model_ABW(const string config_file):Model(config_file){
    init();
}

void Model_ABW::reset(const string& config_data){
    release_memory();
    bool rc = _cfg.set_config_data(config_data);
    if(!rc){
        cerr<<"Setting config data failed. Abort."<<endl;
        exit(1);
    }
    init();
}

void Model_ABW::init(){
    is_compressible = _cfg.is_compressible();

    vec vf = _cfg.f();
    fA = vf(0);
    fB = vf(1);
    vec vlrlam = _cfg.wall_affinity();
    llam = vlrlam(0);
    rlam = vlrlam(1);
    vec vchiN = _cfg.chiN();
    chiN = vchiN(0);
    vec va = _cfg.segment_length();
    aA = va(0);
    aB = va(1);
    vec vds = _cfg.ds();
    dsA = vds(0);
    dsB = vds(1);

    umat vMs = _cfg.Ms();
    sA = vMs(0);
    sB = vMs(1);
    dsA = fA/(sA-1);
    dsB = fB/(sB-1);
    vec lam = _cfg.lam();
    lamA = lam(0);
    lamB = lam(1);
    lamYita = lam(2);
    init_field();
    init_density();
    init_propagator();
}

void Model_ABW::update(){
    if(_cfg.algo_scft_type() == AlgorithmSCFTType::ANDERSON){
        qA->update(*wAx);
        qB->set_head( qA->get_tail() );
        qB->update(*wBx);

        qBc->update(*wBx);
        qAc->set_head( qBc->get_tail() );
        qAc->update(*wAx);
    }
    else{
        qA->update(*wA);
        qB->set_head( qA->get_tail() );
        qB->update(*wB);

        qBc->update(*wB);
        qAc->set_head( qBc->get_tail() );
        qAc->update(*wA);
    }

    double Q = qB->Qt();
    phiA->set_cc(1/Q*phi);
    phiA->update(*qA, *qAc);
    phiB->set_cc(1/Q*phi);
    phiB->update(*qB, *qBc);

    /**
     * It is essential to combine A and B blocks to prodcue
     * Anderson mixing coefficient C.
     * Updating by Anderson mixing for A and B separately fails.
     */
    if(is_compressible){
        Grid eta = lamYita * ((*phiA) + (*phiB) - 1.0);
        if(_cfg.algo_scft_type() == AlgorithmSCFTType::ANDERSON){
            wAx->pre_update(chiN * (*phiB) + eta);
            wBx->pre_update(chiN * (*phiA) + eta);
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
            wA->update( chiN * (*phiB) + eta );
            wB->update( chiN * (*phiA) + eta );
        }
    }
    else{
        if(_cfg.algo_scft_type() == AlgorithmSCFTType::ANDERSON){
            (*yita) = 0.5 * ((*wAx) + (*wBx) - chiN);
            //cout<<"wAx k = "<<wAx->n_iteration()<<endl;
            //cout<<"wAx cur_pos = "<<wAx->current_index()<<endl;
            //cout<<"wBx k = "<<wBx->n_iteration()<<endl;
            //cout<<"wBx cur_pos = "<<wBx->current_index()<<endl;
            wAx->pre_update(chiN * (*phiB) + (*yita));
            wBx->pre_update(chiN * (*phiA) + (*yita));
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
            yita->update( (*phiA) + (*phiB) + (*phiW) - 1.0 );
            wA->update( chiN * (*phiB) - (*wH) + (*yita) );
            wB->update( chiN * (*phiA) + (*wH) + (*yita) );
        }
    }
    blitz::Array<double, 3> phia_begin(phiA_begin->data());
	blitz::Array<double, 3> phia_end(phiA_end->data());
	blitz::Array<double, 3> phib_begin(phiB_begin->data());
	blitz::Array<double, 3> phib_end(phiB_end->data());
	blitz::Array<double, 4> qsA(qA->qs());
	blitz::Array<double, 4> qsAc(qAc->qs());
	blitz::Array<double, 4> qsB(qB->qs());
	blitz::Array<double, 4> qsBc(qBc->qs());
	blitz::Range all = blitz::Range::all();
	phia_begin(all, all, all) = qsA(0, all, all, all)*qsAc(sA-1, all, all, all)/Q*phi;
	phia_end(all, all, all) = qsA(sA-1, all, all, all)*qsAc(0, all, all, all)/Q*phi;
	phib_begin(all, all, all) = qsB(sB-1, all, all, all)*qsBc(0, all, all, all)/Q*phi;
	phib_end(all, all, all) = qsB(0, all, all, all)*qsBc(sB-1, all, all, all)/Q*phi;
}


double Model_ABW::Hw() const{
    double ret;
    if(is_compressible){
        if(_cfg.algo_scft_type() == AlgorithmSCFTType::ANDERSON){
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
        }
    }
    else{
        if(_cfg.algo_scft_type() == AlgorithmSCFTType::ANDERSON){
            Grid g = chiN * (*phiA) * (*phiB) -
                     (*wAx) * (*phiA) - (*wBx) * (*phiB);
            ret = g.quadrature()/phi;
        }
        else{
            Grid g = chiN * (*phiA) * (*phiB) - (*wH)*((*phiA)-(*phiB)) -
                     (*wA) * (*phiA) - (*wB) * (*phiB)/* + (*yita)*((*phiA)+(*phiB)+(*phiW)-1.0)*/;
            ret = g.quadrature()/phi;
        }
    }
    return ret;
}

double Model_ABW::Hs() const{
    return -log(qB->Qt());
}

double Model_ABW::H() const{
    return Hw() + Hs();
}

double Model_ABW::incomp() const{
    Grid g = (*phiA) + (*phiB) + (*phiW) - 1.0;
    return g.abs_quadrature();
}

double Model_ABW::residual_error() const{
    double res = 0.0;
    if(_cfg.algo_scft_type() == AlgorithmSCFTType::ANDERSON){
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
            g1 = chiN * (*phiB) - (*wH) + (*yita);
            g2 = g1 - (*wA);
            r = g2.abs_mean();
            x = wA->abs_mean();
            b = g1.abs_mean();
            res += r / (x+b);
            // wB
            g1 = chiN * (*phiA) + (*wH) + (*yita);
            g2 = g1 - (*wB);
            r = g2.abs_mean();
            x = wB->abs_mean();
            b = g1.abs_mean();
            res += r / (x+b);
            // Yita
            g1 = (*phiA) + (*phiB) + (*phiW);
            g2 = g1 - 1.0;
            r = g2.abs_mean();
            x = 1.0;
            b = g1.abs_mean();
            res += r / (x+b);
            res /= 3.0;
        }
        else{
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
        }
    }
    return res;
}

/*
double Model_ABW::density_error() const{
    double err=0.0;
    Grid g=(*_phiA)-(*_phiA0);
    err=g.abs_mean();
    g=(*_phiB)-(*_phiB0);
    err += g.abs_mean();
    return err/2;
}*/

void Model_ABW::display() const{
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

    if(_cfg.algo_scft_type() == AlgorithmSCFTType::ANDERSON){
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
    cout<<"\tResidual Error    = "<<residual_error()<<endl;
    cout<<"\tphiA+phiB+phiW-1.0 = " <<phiA->quadrature()+phiB->quadrature()+phiW->quadrature()-1.0<< endl;

    cout.unsetf(ios::floatfield);
    cout.precision(6);
}

void Model_ABW::save(const string file){
    phiA->uc().save(file, phiA->Lx(), phiA->Ly(), phiA->Lz());
    save_field(file);
    save_density(file);
}

void Model_ABW::save_field(const string file){
    if(_cfg.algo_scft_type() == AlgorithmSCFTType::ANDERSON){
        wAx->save(file);
        wBx->save(file);
    }
    else{
        wA->save(file);
        wB->save(file);
        wH->save(file);
    }
    if(!is_compressible)
        yita->save(file);
}

void Model_ABW::save_density(const string file){
    phiA->save(file);
    phiB->save(file);
    phiW->save(file);

    CMatFile mat;
    mat.matInit(file,"u");
    if(!mat.queryStatus()){
        mat.matPutScalar("QAB", qB->Qt());
        mat.matPutScalar("fp", phi);
        mat.matRelease();
    }
    
    phiA_begin->save(file);
    phiA_end->save(file);
    phiB_begin->save(file);
    phiB_end->save(file);
}

void Model_ABW::save_q(const string file){
    qA->save(file);
    qAc->save(file);
    qB->save(file);
    qBc->save(file);
}

void Model_ABW::save_model(const string file){
    CMatFile mat;
    mat.matInit(file,"u");
    if(!mat.queryStatus()){
        mat.matPutScalar("fA", fA);
        mat.matPutScalar("fB", fB);
        mat.matPutScalar("aA", aA);
        mat.matPutScalar("aB", aB);
        mat.matPutScalar("chiN", chiN);
        mat.matPutScalar("dsA", dsA);
        mat.matPutScalar("dsB", dsB);
        mat.matPutScalar("sA", sA);
        mat.matPutScalar("sB", sB);
        mat.matPutScalar("phi", phi);
        if(_cfg.algo_scft_type() == AlgorithmSCFTType::ANDERSON){
            mat.matPutScalar("seedA", wAx->seed());
            mat.matPutScalar("seedB", wBx->seed());
        }
        else{
            mat.matPutScalar("seedA", wA->seed());
            mat.matPutScalar("seedB", wB->seed());
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

void Model_ABW::display_parameters() const{
    cout<<endl;
    cout<<"********* Model_ABW Parameter List **********"<<endl;
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

    cout<<"fA = "<<fA<<"\tfB = "<<fB<<endl;
    cout<<"chiN = "<<chiN<<endl;
    cout<<"aA = "<<aA<<"\taB = "<<aB<<endl;
    cout<<"dsA = "<<dsA<<"\tdsB = "<<dsB<<endl;
    cout<<"sA = "<<sA<<"\tsB = "<<sB<<endl;
    if(_cfg.algo_scft_type() == AlgorithmSCFTType::ANDERSON)
        cout<<"seedA = "<<wAx->seed()<<"\tseedB = "<<wBx->seed()<<endl;
    else
        cout<<"seedA = "<<wA->seed()<<"\tseedB = "<<wB->seed()<<endl;
    cout<<endl;

    cout<<"dimension: "<<_cfg.dim()<<endl;
    cout<<"(Lx, Ly, Lz) = ";
    cout<<"("<<phiA->Lx()<<", "<<phiA->Ly()<<", "<<phiA->Lz()<<")"<<endl;
    cout<<"(a, b, c) = ";
    cout<<"("<<phiA->lx()<<", "<<phiA->ly()<<", "<<phiA->lz()<<")"<<endl;

    cout<<"*******************************************"<<endl;
    cout<<endl;
}

void Model_ABW::release_memory(){
    if(_cfg.algo_scft_type() == AlgorithmSCFTType::ANDERSON){
        delete wAx;
        delete wBx;
    }
    else{
        delete wA;
        delete wB;
        delete wH;
    }
    if(!is_compressible)
        delete yita;
    delete phiA;
    delete phiB;
    delete phiW;
    delete phiA_begin;
    delete phiA_end;
    delete phiB_begin;
    delete phiB_end;
    delete qA;
    delete qB;
    delete qAc;
    delete qBc;
    if(_cfg.ctype() != ConfineType::NONE
       || (_cfg.algo_mde_type() == AlgorithmMDEType::ETDRK4
           && _cfg.etdrk4_M() > 0)){
        delete ppropupA;
        delete ppropupB;
    }
}

Model_ABW::~Model_ABW(){
    release_memory();
}

void Model_ABW::init_random_field(){
    double low = 0.0;
    double high = 1.0;
    int seed = _cfg.seed();
    if(_cfg.algo_scft_type() == AlgorithmSCFTType::ANDERSON){
        uword n = _cfg.n_Anderson_mixing();
        wAx = new FieldAX("wA", _cfg, lamA, n, low, high, seed);
        wBx = new FieldAX("wB", _cfg, lamB, n, low, high, seed);
    }
    else{
        wA = new Field("wA", _cfg, lamA, low, high, seed);
        wB = new Field("wB", _cfg, lamB, low, high, seed);
    }
}

void Model_ABW::init_constant_field(){
    double vA = 0.5;
    double vB = 0.5;
    if(_cfg.algo_scft_type() == AlgorithmSCFTType::ANDERSON){
        uword n = _cfg.n_Anderson_mixing();
        wAx = new FieldAX("wA", _cfg, vA, lamA, n);
        wBx = new FieldAX("wB", _cfg, vB, lamB, n);
    }
    else{
        wA = new Field("wA", _cfg, vA, lamA);
        wB = new Field("wB", _cfg, vB, lamB);
    }
}

void Model_ABW::init_file_field(){
    string file = _cfg.field_data_file();
    //cout<<"init_data_file: "<<file<<endl;
    if(_cfg.algo_scft_type() == AlgorithmSCFTType::ANDERSON){
        uword n = _cfg.n_Anderson_mixing();
        wAx = new FieldAX("wA", _cfg, file, lamA, n);
        wBx = new FieldAX("wB", _cfg, file, lamB, n);
    }
    else{
        wA = new Field("wA", _cfg, file, lamA);
        wB = new Field("wB", _cfg, file, lamB);
    }
}

void Model_ABW::init_pattern_field(){
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

void Model_ABW::init_field(){
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

void Model_ABW::init_density(){
	phiA_begin = new Density("phiA_begin", _cfg);
    phiA_end = new Density("phiA_end", _cfg);
    phiB_begin = new Density("phiB_begin", _cfg);
    phiB_end = new Density("phiB_end", _cfg);

    AlgorithmContourType actype = _cfg.algo_contour_integration_type();
    if(actype == AlgorithmContourType::TRAPEZOIDAL){
        phiA = new Density("phiA", _cfg);
        phiB = new Density("phiB", _cfg);
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
        //phiW = new Density("phiW", _cfg, new Simpson);
    }
    else{
        cerr<<"Contour integration algorithm: ";
        cerr<<_cfg.get_algo_contour_integration_type_string();
        cerr<<" is not available."<<endl;
        exit(1);
    }

/*define wall density phiW, wall potential H and polymer volume fraction phi*/
    blitz::Range all = blitz::Range::all();
    uword Lx = _cfg.Lx();
    uword Ly = _cfg.Ly();
    uword Lz = _cfg.Lz();
    double La = _cfg.a();
    double Lb = _cfg.b();
    double Lc = _cfg.c();
    int L_depletion;
    int L_padding;
    wh.resize(Lx, Ly, Lz);
    phiw.resize(Lx, Ly, Lz);
    deplen = _cfg.depletion_length(); 

    switch (_cfg.dim()) {
        case 1 : 
        {
            blitz::Array<double, 1> x(Lx);
            blitz::Array<double, 1> array1(Lx);
            blitz::Array<double, 1> array2(Lx);
            blitz::Array<double, 1> EXP_array(Lx);
            blitz::Array<double, 3> phia(phiA->data());
            blitz::Array<double, 3> phib(phiB->data());

             
            blitz::firstIndex j;

            if(_cfg.ctype() == ConfineType::CUBE) {
                x = cos(kPI*j/(Lx-1));
                x = (1-x)/2*La;
                L_depletion = (int)((Lx-1)/kPI*acos(1-2*deplen/La));
                deplen = (1-cos(kPI*L_depletion/(Lx-1)))/2*La;
            }
            else {
                x = 1.0 * La * j / (Lx-1);
                L_depletion = (int)((Lx-1) * deplen / La);
                deplen = 1.0 * La * L_depletion / (Lx-1);
            }
            array1 = 0.0;
            array2 = 0.0;
            switch (_cfg.mask()) {
                case MaskType::EXP:
                {
                    for(int i=0; i<L_depletion; ++i) {
                        EXP_array(i) = exp(4*deplen*x(i)/(deplen*deplen-x(i)*x(i)));
                        array1(i) = 1.0-pow2(EXP_array(i)-1)/pow2(EXP_array(i)+1);
                        array2(i) = llam * array1(i);
                    }
                    blitz::Array<double, 1> tmparray1(array1(Range(0, L_depletion-1)));
                    array1(Range(Lx-L_depletion, Lx-1)) = tmparray1.reverse(blitz::firstDim); 
                    array2(Range(Lx-L_depletion, Lx-1)) = rlam*array1(Range(Lx-L_depletion, Lx-1));
                    break;
                }
                case MaskType::COS:
                {
                    for(int i=0; i<L_depletion; ++i) {
                        array1(i) = (1.0+cos(kPI*x(i)/deplen))/2.0;
                        array2(i) = llam * array1(i);
                    }
                    blitz::Array<double, 1> tmparray1(array1(Range(0, L_depletion-1)));
                    array1(Range(Lx-L_depletion, Lx-1)) = tmparray1.reverse(blitz::firstDim);
                    array2(Range(Lx-L_depletion, Lx-1)) = rlam * array1(Range(Lx-L_depletion, Lx-1));
                    break;
                }
                case MaskType::TANH:
                {
                    array1 = (1.0-tanh(5.0/deplen*(x-0.25)))/2; //phiw = (1-tanh(m/t(x-T/2)))/2 t:deplen;T/2:padding length
                    array2 = llam * array1;
                    blitz::Array<double, 1> tmparray1(array1(Range(0, Lx/2-1)));
                    array1(Range(Lx/2, Lx-1)) = tmparray1.reverse(blitz::firstDim);
                    array2(Range(Lx/2, Lx-1)) = rlam * array1(Range(Lx/2, Lx-1));
                    break;
                }
                default:
                    cerr<<"Unkonwn or unsupported mask type!"<<endl;
                    exit(1);
            }
            for(int i=0; i<Lx; ++i)
                for(int j=0; j<Ly; ++j)
                    for(int k=0; k<Lz; ++k) {        
                        phiw(i, j, k) = array1(i);
                        wh(i, j, k) = array2(i);
            }
            break;
        } 
        case 2:
        {
            blitz::Array<double, 1> y(Ly);
            blitz::Array<double, 1> array1(Ly);
            blitz::Array<double, 1> array2(Ly);
            blitz::Array<double, 1> EXP_array(Ly);
            blitz::Array<double, 3> phia(phiA->data());
            blitz::Array<double, 3> phib(phiB->data());

             
            blitz::firstIndex j;

            if(_cfg.ctype() == ConfineType::CUBE) {
                y = cos(kPI*j/(Ly-1));
                y = (1-y)/2*Lb;
                L_depletion = (int)((Ly-1)/kPI*acos(1-2*deplen/Lb));
                deplen = (1-cos(kPI*L_depletion/(Ly-1)))/2*Lb;
            }
            else {
                y = 1.0 * Lb * j / (Ly-1);
                L_depletion = (int)((Ly-1) * deplen / Lb);
                deplen = 1.0 * Lb * L_depletion / (Ly-1);
            }
            array1 = 0.0;
            array2 = 0.0;
            switch (_cfg.mask()) {
                case MaskType::EXP:  //see D. Meng and Q. Wang 2007 JCP
                {
                    for(int i=0; i<L_depletion; ++i) {
                        EXP_array(i) = exp(4*deplen*y(i)/(deplen*deplen-y(i)*y(i)));
                        array1(i) = 1.0-pow2(EXP_array(i)-1)/pow2(EXP_array(i)+1);
                        array2(i) = llam * array1(i);
                    }
                    blitz::Array<double, 1> tmparray1(array1(Range(0, L_depletion-1)));
                    array1(Range(Ly-L_depletion, Ly-1)) = tmparray1.reverse(blitz::firstDim); 
                    array2(Range(Ly-L_depletion, Ly-1)) = rlam*array1(Range(Ly-L_depletion, Ly-1));
                    break;
                }
                case MaskType::COS:
                {
                    for(int i=0; i<L_depletion; ++i) {
                        array1(i) = (1.0+cos(kPI*y(i)/deplen))/2.0;
                        array2(i) = llam * array1(i);
                    }
                    blitz::Array<double, 1> tmparray1(array1(Range(0, L_depletion-1)));
                    array1(Range(Ly-L_depletion, Ly-1)) = tmparray1.reverse(blitz::firstDim);
                    array2(Range(Ly-L_depletion, Ly-1)) = rlam * array1(Range(Ly-L_depletion, Ly-1));
                    break;
                }
                case MaskType::TANH:
                {
                    array1 = (1.0-tanh(5.0/deplen*(y-0.25)))/2; //phiw = (1-tanh(m/t(x-T/2)))/2 t:deplen;T/2:padding length
                    array2 = llam * array1;
                    blitz::Array<double, 1> tmparray1(array1(Range(0, Ly/2-1)));
                    array1(Range(Ly/2, Ly-1)) = tmparray1.reverse(blitz::firstDim);
                    array2(Range(Ly/2, Ly-1)) = rlam * array1(Range(Ly/2, Ly-1));
                    break;
                }
                default:
                    cerr<<"Unkonwn or unsupported mask type!"<<endl;
                    exit(1);
            }
            for(int i=0; i<Lx; ++i)
                for(int j=0; j<Ly; ++j)
                    for(int k=0; k<Lz; ++k) {        
                        phiw(i, j, k) = array1(j);
                        wh(i, j, k) = array2(j);
            }
            break;   
        }
        case 3:
        {
            blitz::Array<double, 1> z(Lz);
            blitz::Array<double, 1> array1(Lz);
            blitz::Array<double, 1> array2(Lz);
            blitz::Array<double, 1> EXP_array(Lz);
            blitz::Array<double, 3> phia(phiA->data());
            blitz::Array<double, 3> phib(phiB->data());

             
            blitz::firstIndex j;

            if(_cfg.ctype() == ConfineType::CUBE) {
                z = cos(kPI*j/(Lz-1));
                z = (1-z)/2*Lc;
                L_depletion = (int)((Lz-1)/kPI*acos(1-2*deplen/Lc));
                deplen = (1-cos(kPI*L_depletion/(Lz-1)))/2*Lc;
            }
            else {
                z = 1.0 * Lb * j / Lz;
                L_depletion = (int)((Lz-1) * deplen / Lc);
                deplen = 1.0 * Lc * L_depletion / (Lz-1);
            }
            array1 = 0.0;
            array2 = 0.0;
            switch (_cfg.mask()) {
                case MaskType::EXP:
                {
                    for(int i=0; i<L_depletion; ++i) {
                        EXP_array(i) = exp(4*deplen*z(i)/(deplen*deplen-z(i)*z(i)));
                        array1(i) = 1.0-pow2(EXP_array(i)-1)/pow2(EXP_array(i)+1);
                        array2(i) = llam * array1(i);
                    }
                    blitz::Array<double, 1> tmparray1(array1(Range(0, L_depletion-1)));
                    array1(Range(Lz-L_depletion, Lz-1)) = tmparray1.reverse(blitz::firstDim); 
                    array2(Range(Lz-L_depletion, Lz-1)) = rlam*array1(Range(Lz-L_depletion, Lz-1));
                    break;
                }
                case MaskType::COS:
                {
                    for(int i=0; i<L_depletion; ++i) {
                        array1(i) = (1.0+cos(kPI*z(i)/deplen))/2.0;
                        array2(i) = llam * array1(i);
                    }
                    blitz::Array<double, 1> tmparray1(array1(Range(0, L_depletion-1)));
                    array1(Range(Lz-L_depletion, Lz-1)) = tmparray1.reverse(blitz::firstDim);
                    array2(Range(Lz-L_depletion, Lz-1)) = rlam * array1(Range(Lz-L_depletion, Lz-1));
                    break;
                }
                case MaskType::TANH:
                {
                    array1 = (1.0-tanh(5.0/deplen*(z-0.25)))/2; //phiw = (1-tanh(m/t(x-T/2)))/2 t:deplen;T/2:padding length
                    array2 = llam * array1;
                    blitz::Array<double, 1> tmparray1(array1(Range(0, Lz/2-1)));
                    array1(Range(Lz/2, Lz-1)) = tmparray1.reverse(blitz::firstDim);
                    array2(Range(Lz/2, Lz-1)) = rlam * array1(Range(Lz/2, Lz-1));
                    break;
                }
                default:
                    cerr<<"Unkonwn or unsupported mask type!"<<endl;
                    exit(1);
            }
            for(int i=0; i<Lx; ++i)
                for(int j=0; j<Ly; ++j)
                    for(int k=0; k<Lz; ++k) {        
                        phiw(i, j, k) = array1(k);
                        wh(i, j, k) = array2(k);
            }
            break;   
        }
        default:
            break;
    }          
    
    phiW = new Density("phiW", _cfg, phiw);
    wH = new Field("wH", _cfg, wh, lamA);
    if(_cfg.ctype() == ConfineType::CUBE) {
        blitz::firstIndex i;
        blitz::secondIndex j;
        blitz::thirdIndex k;
        switch (_cfg.dim()) {
        case 1: 
            {
                blitz::Array<double, 1> data1(phiw(all, 0, 0));
                arma::vec v(data1.data(), Lx);
                Cheb cheb(Lx);
                phi = 1.0-0.5 * cheb.quadrature_clencurt(v); //phi=1-\int_0^{L_z}phiw(all,0,0)dz/L_z   Clenshaw-Curtis quadrature is used here
            break;
            }
        case 2:
            {
            	blitz::Array<double, 2> data2(phiw(all, all, 0));
            	blitz::Array<double, 1> data1(Ly);
            	data1 = blitz::mean(data2(j,i),j);
            	arma::vec v(data1.data(), Ly);
            	Cheb cheb(Ly);
            	phi = 1.0 - 0.5 * cheb.quadrature_clencurt(v);
            	break;
            }
        case 3:
            {
                blitz::Array<double, 3> data3(phiw(all, all, all));
                blitz::Array<double, 2> data2(Lx, Lz);
                blitz::Array<double, 1> data1(Lz);
                data2 = blitz::mean(data3(i,k,j),k);
                data1 = blitz::mean(data2(j,i),j);
                arma::vec v(data1.data(), Lz);
                Cheb cheb(Lz);
                phi =1.0 - 0.5 * cheb.quadrature_clencurt(v);
                break;
            }
        default:
            break;
        }
    }
    //else phi = 1.0 - phiW->quadrature(); 
    else phi = 1.0 - blitz::mean(phiw);
    
}

void Model_ABW::init_propagator(){
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
        }
        else if(_cfg.algo_mde_type() == AlgorithmMDEType::ETDRK4
                && _cfg.etdrk4_M() > 0){
            // NOTE: ONLY Cox-Matthews scheme has been implemented.
            // Thus the input _cfg.etdrk4_scheme_type() is ignored.
            ppropupA = new Etdrk4_PBC(uc, Lx, Ly, Lz, dsA, _cfg.etdrk4_M());
            ppropupB = new Etdrk4_PBC(uc, Lx, Ly, Lz, dsB, _cfg.etdrk4_M());
            qA = new Propagator("qA", _cfg, sA, dsA, one, ppropupA);
            qB = new Propagator("qB", _cfg, sB, dsB, ppropupB);
            qAc = new Propagator("qAc", _cfg, sA, dsA, ppropupA);
            qBc = new Propagator("qBc", _cfg, sB, dsB, one, ppropupB);
        }
        else if(_cfg.algo_mde_type() == AlgorithmMDEType::OS2){
            ppropupA = new PseudoSpectral(uc, Lx, Ly, Lz, dsA);
            ppropupB = new PseudoSpectral(uc, Lx, Ly, Lz, dsB);
            qA = new Propagator("qA", _cfg, sA, dsA, one, ppropupA);
            qB = new Propagator("qB", _cfg, sB, dsB, ppropupB);
            qAc = new Propagator("qAc", _cfg, sA, dsA, ppropupA);
            qBc = new Propagator("qBc", _cfg, sB, dsB, one, ppropupB);
        }
        else if(_cfg.algo_mde_type() == AlgorithmMDEType::RQM4){
            ppropupA = new RQM4(uc, Lx, Ly, Lz, dsA);
            ppropupB = new RQM4(uc, Lx, Ly, Lz, dsB);
            qA = new Propagator("qA", _cfg, sA, dsA, one, ppropupA);
            qB = new Propagator("qB", _cfg, sB, dsB, ppropupB);
            qAc = new Propagator("qAc", _cfg, sA, dsA, ppropupA);
            qBc = new Propagator("qBc", _cfg, sB, dsB, one, ppropupB);
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

        // NOTE: ONLY Krogstad scheme has been implemented.
        // Thus the input _cfg.etdrk4_scheme_type() is ignored.
        if(_cfg.etdrk4_M() <= 0){
            ppropupA = new Etdrk4(uc, dim, Lx, Ly, Lz, dsA,
                                  confine_type, lbcA, rbcA);
            ppropupB = new Etdrk4(uc, dim, Lx, Ly, Lz, dsB,
                                  confine_type, lbcB, rbcB);
        }
        else{
            ppropupA = new Etdrk4(uc, dim, Lx, Ly, Lz, dsA,
                                  confine_type, lbcA, rbcA,
                                  ETDRK4SCHEME::KROGSTAD, _cfg.etdrk4_M());
            ppropupB = new Etdrk4(uc, dim, Lx, Ly, Lz, dsB,
                                  confine_type, lbcB, rbcB,
                                  ETDRK4SCHEME::KROGSTAD, _cfg.etdrk4_M());
        }
        qA = new Propagator("qA", _cfg, sA, dsA, one, ppropupA);
        qB = new Propagator("qB", _cfg, sB, dsB, ppropupB);
        qAc = new Propagator("qAc", _cfg, sA, dsA, ppropupA);
        qBc = new Propagator("qBc", _cfg, sB, dsB, one, ppropupB);
    }
    else{
        cerr<<"Confinement: "<<_cfg.get_confine_type_string();
        cerr<<" is not available."<<endl;
        exit(1);
    }
}

