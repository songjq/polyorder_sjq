#include "Model_A_B.h"
#include "CMatFile.h"
#include "Helper.h"

using namespace arma;
using namespace std;
using namespace blitz;

Model_A_B::Model_A_B(const string config_file):Model(config_file){
    init();
}

void Model_A_B::reset(const string& config_data){
    release_memory();
    bool rc = _cfg.set_config_data(config_data);
    if(!rc){
        cerr<<"Setting config data failed. Abort."<<endl;
        exit(1);
    }
    init();
}

void Model_A_B::init(){
	blitz::Range all = blitz::Range::all();
    is_compressible = _cfg.is_compressible();

    vec vf = _cfg.f();
    fA = vf(0);
    fC = vf(1);
    fC = 1.0 - fA;
    vec vchiN = _cfg.chiN();
    chiN = vchiN(0);
    vec va = _cfg.segment_length();
    aA = va(0);
    aC = va(1);

    vec vds = _cfg.ds();
    dsA = vds(0);
    dsC = vds(1);

    umat vMs = _cfg.Ms();
    sA = vMs(0);
    sC = vMs(1);
    dsA = _cfg.length_ratio()/(sA-1); //length_ratio is length ration of free homopymer A over grafted homopolymer C
    dsC = 1.0/(sC-1);

    vec lam = _cfg.lam();
    lamA = lam(0);
    lamC = lam(1);
    lamYita = lam(2);
    
    switch (_cfg.dim()) {
        case 1 :
            sigma = fC * _cfg.a();  // sigma is \sigma*N/\rho_0 actually
            break;
        case 2 :
            sigma = fC * _cfg.b();
            break;
        case 3 :
            sigma = fC * _cfg.c();
            break;
        default :
            cout << "Please input correct dimension !" << endl;
            break;
    }
    
    init_delta();
    init_field();
    init_density();
    init_propagator();
}

void Model_A_B::update(){
    blitz::Range all = blitz::Range::all();
    qA->update(*wA);
    qAc->update(*wA);
    qCc->update(*wC);
    blitz::Array<double, 4> qc(qC->qs());
    blitz::Array<double, 4> qcc(qCc->qs());

    blitz::firstIndex i;
    blitz::secondIndex j;
    blitz::thirdIndex k;
    switch(_cfg.dim()) {
    	case 1:
    	{
    		blitz::Array<double, 1> data1(delta(all, 0, 0)*qcc(sC-1, all, 0, 0));
    		arma::vec v(data1.data(), _cfg.Lx());
        	Cheb cheb(_cfg.Lx());
            QC = 0.5 * cheb.quadrature_clencurt(v); //Notice:QC here is already normalized in .

    		for(int i=0; i<_cfg.Lx(); ++i) 
    			qc(0, i, all, all) = sigma * delta(i, all, all) / (QC*_cfg.a()); 
        	break;
        }
        case 2:
        {
            blitz::Array<double, 2> data2(delta(all, all, 0) * qcc(sC-1, all, all, 0));
        	blitz::Array<double, 1> data1(_cfg.Ly());
        	data1 = blitz::mean(data2(j,i),j);
        	arma::vec v(data1.data(), _cfg.Ly());
        	Cheb cheb(_cfg.Ly());
            QC = 0.5 * cheb.quadrature_clencurt(v);

        	for(int j=0; j<_cfg.Ly(); ++j)
        		qc(0, all, j, all) = sigma * delta(all, j, all) / (QC*_cfg.b()); //QC must be multified by L_b only but not L_a*L_b
        	break;
        }
        case 3:
        {
        	blitz::Array<double, 3> data3(delta(all, all, all) * qcc(sC-1,all,all,all));
            blitz::Array<double, 2> data2(_cfg.Lx(), _cfg.Lz());
            blitz::Array<double, 1> data1(_cfg.Lz());
            data2 = blitz::mean(data3(i,k,j),k);
            data1 = blitz::mean(data2(j,i),j);
            arma::vec v(data1.data(), _cfg.Lz());
        	Cheb cheb(_cfg.Lz());
            QC = 0.5 * cheb.quadrature_clencurt(v);

        	for(int k=0; k<_cfg.Lz(); ++k)
        		qc(0, all, all, k) = sigma * delta(all, all, k) / (QC*_cfg.c()); //QC must be multified by L_c only but not L_a*L_b*L_c
        	break;
        }
        default:
        	break;
    }

    qC->update(*wC);
    double QA = qA->Qt();
    phiA->set_cc(fA/QA/_cfg.length_ratio());   
    phiA->update(*qA, *qAc);
    phiC->set_cc(1.0);
    phiC->update(*qC, *qCc);

    yita->update( (*phiA) + (*phiC) - 1.0 );
    wA->update( chiN * (*phiC) + (*yita) );
    wC->update( chiN * (*phiA) + (*yita) );  
/***************************calculate specific segment**********************************/
    blitz::Array<double, 3> phia_end(phiA_end->data());
    blitz::Array<double, 3> phia_middle(phiA_middle->data());
    blitz::Array<double, 4> qsA(qA->qs());
    blitz::Array<double, 4> qsAc(qAc->qs());
    phia_end(all, all, all) = fA/QA/_cfg.length_ratio()*qsA(0, all, all, all)*qsAc(sA-1, all, all, all);
    phia_middle(all, all, all) = fA/QA/_cfg.length_ratio()*qsA((sA-1)/2, all, all, all)*qsAc((sA-1)/2, all, all, all); 

    blitz::Array<double, 3> phic_begin(phiC_begin->data());
    blitz::Array<double, 3> phic_end(phiC_end->data());
    blitz::Array<double, 3> phic_middle(phiC_middle->data());
    blitz::Array<double, 4> qsC(qC->qs());
    blitz::Array<double, 4> qsCc(qCc->qs());        
    phic_begin(all, all, all) = qsC(0, all, all, all)*qsCc(sC-1, all, all, all);
    phic_end(all, all, all) = qsC(sC-1, all, all, all)*qsCc(0, all, all, all);
    phic_middle(all, all, all) = qsC((sC-1)/2, all, all, all)*qsCc((sC-1)/2, all, all, all);
}

double Model_A_B::Hw() const{
    double ret;
    Grid g = chiN * (*phiA) * (*phiC) + (*yita)*((*phiA)+(*phiC)-1.0) -
             (*wA) * (*phiA) - (*wC) * (*phiC);
    ret = g.quadrature();
    return ret;
}

double Model_A_B::Hs() const{
	double Hs1 = -fA/_cfg.length_ratio()*log(qA->Qt())-fC*log(QC);
    double Hs2 = (phiC->quadrature())*(log(phiC->quadrature())-1.0);
    double Hs3 = (phiA->quadrature())/_cfg.length_ratio()*log(phiA->quadrature());
    return Hs1 + Hs2 + Hs3;
}

double Model_A_B::H() const{
    return Hw() + Hs();
}

double Model_A_B::incomp() const{
    Grid g = (*phiA) + (*phiC) - 1.0;
    return g.abs_quadrature();
}

double Model_A_B::residual_error() const{
    double res = 0.0;
    /* For Field ONLY */
    double r, x, b;
    Grid g1, g2;
    // wA
    g1 = chiN * (*phiC) + (*yita);
    g2 = g1 - (*wA);
    r = g2.abs_mean();
    x = wA->abs_mean();
    b = g1.abs_mean();
    res += r / (x+b);
    // wC
    g1 = chiN * (*phiA) + (*yita);
    g2 = g1 - (*wC);
    r = g2.abs_mean();
    x = wC->abs_mean();
    b = g1.abs_mean();
    res += r / (x+b);
    // Yita
    g1 = (*phiA) + (*phiC);
    g2 = g1 - 1.0;
    r = g2.abs_mean();
    x = 1.0;
    b = g1.abs_mean();
    res += r / (x+b);
    res /= 3.0;

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

void Model_A_B::display() const{
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
    cout<<"\tphiC = "<<phiC->quadrature();
    cout<<"\t["<<phiC->min()<<", "<<phiC->max()<<"]"<<endl;
    cout<<"\twA   = "<<wA->quadrature();
    cout<<"\t["<<wA->min()<<", "<<wA->max()<<"]"<<endl;
    cout<<"\twC   = "<<wC->quadrature();
    cout<<"\t["<<wC->min()<<", "<<wC->max()<<"]"<<endl;
    cout<<"\tyita = "<<yita->quadrature();
    cout<<"\t["<<yita->min()<<", "<<yita->max()<<"]"<<endl;
    cout.setf(ios::scientific, ios::floatfield);
    cout.precision(2);
    //cout<<"\tIncompressibility = "<<incomp()<<endl;
    blitz::Array<double, 4> qcc(qCc->qs());
    //cout<<"\tQC = "<<sum(qcc(sC-1, 0, all, all));
    cout<<"\tQA = "<<qAc->Qt()<<"\tQC = "<<QC<<endl;
    cout<<"\tResidual Error    = "<<residual_error()<<endl;
    cout<<"\tphiA+phiC-1.0     = "<<(phiA->quadrature()+phiC->quadrature()-1.0) << endl;

    cout.unsetf(ios::floatfield);
    cout.precision(6);
}

void Model_A_B::save(const string file){
    phiA->uc().save(file, phiA->Lx(), phiA->Ly(), phiA->Lz());
    save_field(file);
    save_density(file);
}

void Model_A_B::save_field(const string file){
    wA->save(file);
    wC->save(file);
    yita->save(file);
}

void Model_A_B::save_density(const string file){
    phiA->save(file);
    phiC->save(file);
    phiA_end->save(file);
    phiA_middle->save(file);
    phiC_begin->save(file);
    phiC_end->save(file);
    phiC_middle->save(file);
}

void Model_A_B::save_q(const string file){
    qA->save(file);
    qAc->save(file);
    qC->save(file);
    qCc->save(file);
}

void Model_A_B::save_model(const string file){
    CMatFile mat;
    mat.matInit(file,"u");
    if(!mat.queryStatus()){
        mat.matPutScalar("fA", fA);
        mat.matPutScalar("fC", fC);
        mat.matPutScalar("aA", aA);
        mat.matPutScalar("aC", aC);
        mat.matPutScalar("chiN", chiN);
        mat.matPutScalar("dsA", dsA);
        mat.matPutScalar("dsC", dsC);
        mat.matPutScalar("sA", sA);
        mat.matPutScalar("sC", sC);
        mat.matPutScalar("seedA", wA->seed());
        mat.matPutScalar("seedC", wC->seed());

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

void Model_A_B::display_parameters() const{
    cout<<endl;
    cout<<"********* Model Parameter List **********"<<endl;
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

    cout<<"fA = "<<fA<<"\tfC = "<<fC<<endl;
    cout<<"chiN = "<<chiN<<endl;
    cout<<"aA = "<<aA<<"\taC = "<<aC<<endl;
    cout<<"dsA = "<<dsA<<"\tdsC = "<<dsC<<endl;
    cout<<"sA = "<<sA<<"\tsC = "<<sC<<endl;
    if(_cfg.algo_scft_type() == AlgorithmSCFTType::ANDERSON)
        cout<<"seedA = "<<wAx->seed()<<"\tseedC = "<<wCx->seed()<<endl;
    else
        cout<<"seedA = "<<wA->seed()<<"\tseedC = "<<wC->seed()<<endl;
    cout<<endl;

    cout<<"dimension: "<<_cfg.dim()<<endl;
    cout<<"(Lx, Ly, Lz) = ";
    cout<<"("<<phiA->Lx()<<", "<<phiA->Ly()<<", "<<phiA->Lz()<<")"<<endl;
    cout<<"(a, b, c) = ";
    cout<<"("<<phiA->lx()<<", "<<phiA->ly()<<", "<<phiA->lz()<<")"<<endl;

    cout<<"*******************************************"<<endl;
    cout<<endl;
}

void Model_A_B::release_memory(){
    if(_cfg.algo_scft_type() == AlgorithmSCFTType::ANDERSON){
        delete wAx;
        delete wCx;
    }
    else{
        delete wA;
        delete wC;
    }
    if(!is_compressible)
        delete yita;
    delete phiA;
    delete phiC;
    delete phiA_end;
    delete phiA_middle;
    delete phiC_begin;
    delete phiC_end;
    delete phiC_middle;
    delete qA;
    delete qC;
    delete qAc;
    delete qCc;
    if(_cfg.ctype() != ConfineType::NONE
       || (_cfg.algo_mde_type() == AlgorithmMDEType::ETDRK4
           && _cfg.etdrk4_M() > 0)){
        delete ppropupA;
        delete ppropupC;
    }
}

Model_A_B::~Model_A_B(){
    release_memory();
}

void Model_A_B::init_delta(){

/*****************calculate delta function**********************/
    blitz::Range all = blitz::Range::all();
	double alpha = 0.01;  
    uword Lx = _cfg.Lx();
    uword Ly = _cfg.Ly();
    uword Lz = _cfg.Lz();
    blitz::firstIndex i;

    switch(_cfg.dim()) {
    	case 1:
    	{                              // brackets is necessary for initialization in switch case
    		blitz::Array<double, 1> x(Lx);
    		blitz::Array<double, 1> arr(Lx);
    		x = cos(kPI*i/Lx);
    		x = 0.5 * (1-x) * (_cfg.a());
    		delta.resize(Lx, Ly, Lz);
    		arr = 2.0/sqrt(2.0*kPI*alpha) * exp(-pow2(x)/2/alpha); 
    		for(int i=0; i<Lx; ++i)   delta(i, all, all) = arr(i);

    	    break;
    	}
    	case 2:
    	{                              
    		blitz::Array<double, 1> y(Ly);
    		blitz::Array<double, 1> arr(Ly);
    		y = cos(kPI*i/Ly);        //notice the index is i not j
    		y = 0.5 * (1-y) * (_cfg.b());
    		delta.resize(Lx, Ly, Lz);
            arr = 2.0/sqrt(2.0*kPI*alpha) * exp(-pow2(y)/2/alpha); 
    		for(int j=0; j<Ly; ++j)   delta(all, j, all) = arr(j);
    	    break;
    	}
    	case 3:
    	{                              
    		blitz::Array<double, 1> z(Lz);
    		blitz::Array<double, 1> arr(Lz);
    		z = cos(kPI*i/Lz);        //notice the index is i not j
    		z = 0.5 * (1-z) * (_cfg.c());
    		delta.resize(Lx, Ly, Lz);
            arr = 2.0/sqrt(2.0*kPI*alpha) * exp(-pow2(z)/2/alpha);  
    		for(int k=0; k<Lz; ++k)   delta(all, all, k) = arr(k);
    	    break;
    	}
    	default:
    	    break;
    }
    
    CMatFile mat;
    mat.matInit("delta.mat","w");
    mwSize dim_array[3]={Lx, Ly, Lz};
    blitz::Array<double, 3> data(Lx, Ly, Lz, fortranArray);
    data = delta;
    mat.matPut("delta",data.data(),data.size()*sizeof(double),3,dim_array,mxDOUBLE_CLASS,mxREAL);
    mat.matRelease();
}

void Model_A_B::init_random_field(){
    double low = 0.0;
    double high = 1.0;
    int seed = _cfg.seed();
    if(_cfg.algo_scft_type() == AlgorithmSCFTType::ANDERSON){
        uword n = _cfg.n_Anderson_mixing();
        wAx = new FieldAX("wA", _cfg, lamA, n, low, high, seed);
        wCx = new FieldAX("wB", _cfg, lamC, n, low, high, seed);
    }
    else{
        wA = new Field("wA", _cfg, lamA, low, high, seed);
        wC = new Field("wC", _cfg, lamC, low, high, seed);
    }
}

void Model_A_B::init_constant_field(){
    double vA = 0.5;
    double vC = 0.5;
    if(_cfg.algo_scft_type() == AlgorithmSCFTType::ANDERSON){
        uword n = _cfg.n_Anderson_mixing();
        wAx = new FieldAX("wA", _cfg, vA, lamA, n);
        wCx = new FieldAX("wC", _cfg, vC, lamC, n);
    }
    else{
        wA = new Field("wA", _cfg, vA, lamA);
        wC = new Field("wC", _cfg, vC, lamC);
    }
}

void Model_A_B::init_file_field(){
    string file = _cfg.field_data_file();
    //cout<<"init_data_file: "<<file<<endl;
    if(_cfg.algo_scft_type() == AlgorithmSCFTType::ANDERSON){
        uword n = _cfg.n_Anderson_mixing();
        wAx = new FieldAX("wA", _cfg, file, lamA, n);
        wCx = new FieldAX("wC", _cfg, file, lamC, n);
    }
    else{
        wA = new Field("wA", _cfg, file, lamA);
        wC = new Field("wC", _cfg, file, lamC);
    }
}

void Model_A_B::init_pattern_field(){
    double c = fA<fC?fA:fC;
    double v1 = 0;
    double v2 = 1;
    PhasePattern pt = _cfg.get_phase_pattern();
    if(_cfg.algo_scft_type() == AlgorithmSCFTType::ANDERSON){
        uword n = _cfg.n_Anderson_mixing();
        wAx = new FieldAX("wA", _cfg, lamA, n);
        wCx = new FieldAX("wC", _cfg, lamC, n);
        Helper::init_pattern((*wAx), pt, c, v1, v2);
        Helper::init_pattern((*wCx), pt, c, v2, v1);
    }
    else{
        wA = new Field("wA", _cfg, lamA);
        wC = new Field("wC", _cfg, lamC);
        Helper::init_pattern((*wA), pt, c, v1, v2);
        Helper::init_pattern((*wC), pt, c, v2, v1);
    }
}

void Model_A_B::init_field(){
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

void Model_A_B::init_density(){
    phiA_end = new Density("phiA_end", _cfg);
    phiA_middle = new Density("phiA_middle", _cfg);
    phiC_begin = new Density("phiC_begin", _cfg);
    phiC_end = new Density("phiC_end", _cfg);
    phiC_middle = new Density("phiC_middle", _cfg);

    AlgorithmContourType actype = _cfg.algo_contour_integration_type();
    if(actype == AlgorithmContourType::TRAPEZOIDAL){
        phiA = new Density("phiA", _cfg);
        phiC = new Density("phiC", _cfg);
    }
    else if(actype == AlgorithmContourType::SIMPSON){
        if(sA%2 != 0)
            phiA = new Density("phiA", _cfg, new Simpson);
        else
            phiA = new Density("phiA", _cfg, new Quad4_Open);
        if(sC%2 != 0)
            phiC = new Density("phiC", _cfg, new Simpson);
        else
            phiC = new Density("phiC", _cfg, new Quad4_Open);
    }
    else{
        cerr<<"Contour integration algorithm: ";
        cerr<<_cfg.get_algo_contour_integration_type_string();
        cerr<<" is not available."<<endl;
        exit(1);
    }
}

void Model_A_B::init_propagator(){
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
            qC = new Propagator("qC", _cfg, sC, dsC, one);
            qAc = new Propagator("qAc", _cfg, sA, dsA, one);
            qCc = new Propagator("qCc", _cfg, sC, dsC, one);
        }
        else if(_cfg.algo_mde_type() == AlgorithmMDEType::ETDRK4
                && _cfg.etdrk4_M() > 0){
            // NOTE: ONLY Cox-Matthews scheme has been implemented.
            // Thus the input _cfg.etdrk4_scheme_type() is ignored.
            ppropupA = new Etdrk4_PBC(uc, Lx, Ly, Lz, dsA, _cfg.etdrk4_M());
            ppropupC = new Etdrk4_PBC(uc, Lx, Ly, Lz, dsC, _cfg.etdrk4_M());
            qA = new Propagator("qA", _cfg, sA, dsA, one, ppropupA);
            qC = new Propagator("qC", _cfg, sC, dsC, one, ppropupC);
            qAc = new Propagator("qAc", _cfg, sA, dsA, one, ppropupA);
            qCc = new Propagator("qCc", _cfg, sC, dsC, one, ppropupC);
        }
        else if(_cfg.algo_mde_type() == AlgorithmMDEType::OS2){
            ppropupA = new PseudoSpectral(uc, Lx, Ly, Lz, dsA);
            ppropupC = new PseudoSpectral(uc, Lx, Ly, Lz, dsC);
            qA = new Propagator("qA", _cfg, sA, dsA, one, ppropupA);
            qC = new Propagator("qC", _cfg, sC, dsC, one, ppropupC);
            qAc = new Propagator("qAc", _cfg, sA, dsA, one, ppropupA);
            qCc = new Propagator("qCc", _cfg, sC, dsC, one, ppropupC);
        }
        else if(_cfg.algo_mde_type() == AlgorithmMDEType::RQM4){
            ppropupA = new RQM4(uc, Lx, Ly, Lz, dsA);
            ppropupC = new RQM4(uc, Lx, Ly, Lz, dsC);
            qA = new Propagator("qA", _cfg, sA, dsA, one, ppropupA);
            qC = new Propagator("qC", _cfg, sC, dsC, one, ppropupC);
            qAc = new Propagator("qAc", _cfg, sA, dsA, one, ppropupA);
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
        Boundary lbcC(lbcc(0), lbcc(1), lbcc(2));
        Boundary rbcC(rbcc(0), rbcc(1), rbcc(2));
        //Boundary lbcB(lbcc(0), lbcc(1), lbcc(2));
        //Boundary rbcB(rbcc(0), rbcc(1), rbcc(2));

        // NOTE: ONLY Krogstad scheme has been implemented.
        // Thus the input _cfg.etdrk4_scheme_type() is ignored.
        if(_cfg.etdrk4_M() <= 0){
            ppropupA = new Etdrk4(uc, dim, Lx, Ly, Lz, dsA,
                                  confine_type, lbcA, rbcA);
            ppropupC = new Etdrk4(uc, dim, Lx, Ly, Lz, dsC,
                                  confine_type, lbcC, rbcC);
        }
        else{
            ppropupA = new Etdrk4(uc, dim, Lx, Ly, Lz, dsA,
                                  confine_type, lbcA, rbcA,
                                  ETDRK4SCHEME::KROGSTAD, _cfg.etdrk4_M());
            ppropupC = new Etdrk4(uc, dim, Lx, Ly, Lz, dsC,
                                  confine_type, lbcC, rbcC,
                                  ETDRK4SCHEME::KROGSTAD, _cfg.etdrk4_M());
        }
        qA = new Propagator("qA", _cfg, sA, dsA, one, ppropupA);
        qC = new Propagator("qC", _cfg, sC, dsC, one, ppropupC);
        qAc = new Propagator("qAc", _cfg, sA, dsA, one, ppropupA);
        qCc = new Propagator("qCc", _cfg, sC, dsC, one, ppropupC);
    }
    else{
        cerr<<"Confinement: "<<_cfg.get_confine_type_string();
        cerr<<" is not available."<<endl;
        exit(1);
    }
}

