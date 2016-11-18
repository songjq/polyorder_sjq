#include "Model_AB.h"
#include "CMatFile.h"
#include "Helper.h"

using namespace arma;
using namespace std;

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

    cout.setf(ios::fixed, ios::floatfield);
    cout.precision(8);
    cout << "dsA = " << dsA << endl;
    cout << "dsB = " << dsB << endl;
 
    vec lam = _cfg.lam();
    lamA = lam(0);
    lamB = lam(1);
    lamYita = lam(2);

    init_field();
    init_density();
    init_propagator();
}

void Model_AB::update(){
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
}

double Model_AB::Hw() const{
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
                     (*wA) * (*phiA) - (*wB) * (*phiB) + (*yita)*((*phiA)+(*phiB)+(*phiW)-1.0);
            ret = g.quadrature()/phi;
        }
    }
    return ret;
}

double Model_AB::Hs() const{
    return -log(qB->Qt());
}

double Model_AB::H() const{
    return Hw() + Hs();
}

double Model_AB::incomp() const{
    Grid g = (*phiA) + (*phiB) + (*phiW) - 1.0;
    return g.abs_quadrature();
}

double Model_AB::residual_error() const{
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

void Model_AB::save(const string file){
    phiA->uc().save(file, phiA->Lx(), phiA->Ly(), phiA->Lz());
    save_field(file);
    save_density(file);
}

void Model_AB::save_field(const string file){
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

void Model_AB::save_density(const string file){
    phiA->save(file);
    phiB->save(file);
    phiW->save(file);
}

void Model_AB::save_q(const string file){
    qA->save(file);
    qAc->save(file);
    qB->save(file);
    qBc->save(file);
}

void Model_AB::save_model(const string file){
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

void Model_AB::release_memory(){
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
    }
    else{
        wA = new Field("wA", _cfg, lamA, low, high, seed);
        wB = new Field("wB", _cfg, lamB, low, high, seed);
    }
}

void Model_AB::init_constant_field(){
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

void Model_AB::init_file_field(){
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

void Model_AB::init_pattern_field(){
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
        //phiW = new Density("phiW", _cfg);
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
    uword Lx = _cfg.Lx();
    uword Ly = _cfg.Ly();
    uword Lz = _cfg.Lz();
    double La = _cfg.a();
    int L_depletion;
    int L_padding;
    wh.resize(Lx, Ly, Lz);
    phiw.resize(Lx, Ly, Lz);
    blitz::Array<double, 1> x(Lx);
    blitz::Array<double, 1> array1(Lx);
    blitz::Array<double, 1> array2(Lx);
    blitz::Array<double, 1> EXP_array(Lx);

    deplen = _cfg.depletion_length(); //transition layer
    cout << "depletion_length = " << deplen << endl;
    double padlen = _cfg.padding_length();  //padding length
    cout << "padding_length = " << padlen << endl;
    L_depletion = (int)(Lx * deplen / La);
    deplen = 1.0 * La * L_depletion / Lx;
    L_padding = (int)(Lx * padlen / La);
    padlen = 1.0 * La * L_padding / Lx;
    blitz::firstIndex j;
    x = 1.0 * La * j / Lx;

    for(int i=0; i<Lx; ++i) {
        if(i<L_padding) {
        	array1(i) = 1.0;
            array2(i) = llam;
        }
        else if(i<L_padding+L_depletion) {
            EXP_array(i) = exp(4*deplen*(x(i)-padlen)/(deplen*deplen-(x(i)-padlen)*(x(i)-padlen)));
            array1(i) = 1.0-pow2(EXP_array(i)-1)/pow2(EXP_array(i)+1);
            array2(i) = llam*(1.0-pow2(EXP_array(i)-1)/pow2(EXP_array(i)+1));   
        }
        else if(i<Lx-L_padding-L_depletion) {
            array1(i) = 0.0;
            array2(i) = 0.0;
        }
        else if(i<Lx-L_padding){
            EXP_array(i) = exp(4.0*deplen*(La-(x(i)+padlen))/(deplen*deplen-pow2(La-(x(i)+padlen))));
            array1(i) = 1.0-pow2(EXP_array(i)-1)/pow2(EXP_array(i)+1);
            array2(i) = rlam*(1.0-pow2(EXP_array(i)-1)/pow2(EXP_array(i)+1));  
        }
        else {
        	array1(i) = 1.0;
            array2(i) = rlam; 
        }
        /*if(i<L_depletion) {
            array1(i) = (1.0+cos(kPI*x(i)/deplen))/2.0;
            array2(i) = llam * (1.0+cos(kPI*x(i)/deplen))/2.0;
        }
        else if(i<Lx-L_depletion) {
            array1(i) = 0.0;
            array2(i) = 0.0;
        }
        else {
            array1(i) = (1+cos(kPI*(La*(1.0-1.0/Lx)-x(i))/deplen))/2.0;
            array2(i) = rlam * (1+cos(kPI*(La*(1.0-1.0/Lx)-x(i))/deplen))/2.0;
        }*/

        /*if(i<L_padding) {
        	array1(i) = 1.0;
            array2(i) = llam;
        }
        else if(i<L_padding+L_depletion) {
            array1(i) = (1.0+cos(kPI*(x(i)-padlen)/deplen))/2.0;
            array2(i) = llam * (1.0+cos(kPI*(x(i)-padlen)/deplen))/2.0;
        }
        else if(i<Lx-L_padding-L_depletion) {
            array1(i) = 0.0;
            array2(i) = 0.0;
        }
        else if(i<Lx-L_padding){
            array1(i) = (1+cos(kPI*(La*(1.0-1.0/Lx)-x(i)-padlen)/deplen))/2.0;
            array2(i) = rlam * (1+cos(kPI*(La*(1.0-1.0/Lx)-x(i)-padlen)/deplen))/2.0;
        }
        else {
        	array1(i) = 1.0;
            array2(i) = rlam;
        }*/
        if(array1(i)!=array1(i)) array1(i) = 0.0; //to remove NAN type value
        if(array2(i)!=array2(i)) array2(i) = 0.0;
    }
    
    for(int i=0; i<Lx; ++i)
        for(int j=0; j<Ly; ++j)
            for(int k=0; k<Lz; ++k) {        
                phiw(i, j, k) = array1(i);
                wh(i, j, k) = array2(i);
        }
    phiW = new Density("phiW", _cfg, phiw);
    //cout << "phiW = " << *phiW << endl;
    wH = new Field("wH", _cfg, wh, lamA);
    phi = 1.0 - phiW->quadrature();
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

