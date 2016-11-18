#include "Model_AS.h"
#include "CMatFile.h"
#include "Helper.h"

using namespace arma;
using namespace std;

Model_AS::Model_AS(const string config_file):Model(config_file) {
    init();
}

void Model_AS::reset(const string& config_data){
    release_memory();
    bool rc = _cfg.set_config_data(config_data);
    if(!rc){
        cerr<<"Setting config data failed. Abort."<<endl;
        exit(1);
    }
    init();
}

void Model_AS::init(){
    is_compressible = _cfg.is_compressible();
    vec vv = _cfg.chiN();
    v = vv(0);
    vec va = _cfg.segment_length();
    aA = va(0);
    vec vds = _cfg.ds();
    dsA = vds(0);
    umat vMs = _cfg.Ms();
    sA = vMs(0);
    dsA = 1.0 / (sA-1);  
                                              
    vec lam = _cfg.lam();
    lamA = lam(0);
    graft_density = _cfg.graft_density();

    init_field();
    init_density();
    init_propagator();
}

void Model_AS::update(){
    blitz::Range all = blitz::Range::all();

    qA->update(*wA);

    int ix;
    double x0;
    blitz::Array<double, 4> q(qA->qs());
    blitz::Array<double, 4> qc(qAc->qs());
    blitz::Array<double, 1> x(_cfg.Lx());
    blitz::Array<double, 1> arr(_cfg.Lx());
    blitz::firstIndex i;
    x = cos(cPI*i/_cfg.Lx());
    x = 0.5 * (x+1) * (_cfg.a());
    delta.resize(_cfg.Lx(), _cfg.Ly(), _cfg.Lz());
    x0 = _cfg.a();
    arr = 1.0/sqrt(2.0*cPI*alpha) * exp(-(x-x0)*(x-x0)/2/alpha);  // x0=0:grafted at left surface,x0=_cfg.a():grafted at right surface
    qc = 0.0;
    for(int i=0; i<_cfg.Lx(); ++i) {
        delta(i, all, all) = arr(i);
        qc(0, i, all, all) = graft_density * delta(i, all, all) / q(sA-1, 0, all, all);
    }

    qAc->update(*wA);

    double Q = sum(q(sA-1, 0, all, all));  //q is a 4D array
    phiA->set_cc(1.0/C);
    phiA->update(*qA, *qAc);
    
    wA->update( v*C*(*phiA) );

    CMatFile mat;
    mat.matInit("delta.mat","w");
    mwSize dim_array[3]={_cfg.Lx(), _cfg.Ly(), _cfg.Lz()};
    blitz::Array<double, 3> data(_cfg.Lx(), _cfg.Ly(), _cfg.Lz(), fortranArray);
    data = delta;
    mat.matPut("delta",data.data(),data.size()*sizeof(double),3,dim_array,mxDOUBLE_CLASS,mxREAL);
    mat.matRelease();
}

double Model_AS::Hw() const{
    double ret;
    Grid g = 0.5 * v * C * C * (*phiA) * (*phiA) - C * (*wA) * (*phiA);
         ret = g.quadrature();
    return ret;
}

double Model_AS::Hs() const{
    return -C * log(qA->Qt()); 
}

double Model_AS::H() const{
    return Hw() + Hs();
}

double Model_AS::incomp() const{}

double Model_AS::residual_error() const{
    double res = 0.0;
    double r, x, b;
    Grid g1, g2;
    g1 = v * C * (*phiA);
    g2 = g1 - (*wA);
    r = g2.abs_quadrature();
    x = wA->abs_quadrature();
    b = g1.abs_quadrature();
    res += r / (x+b);

    return res;
}

void Model_AS::display() const{
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

    cout<<"\twA   = "<<wA->quadrature();
    cout<<"\t["<<wA->min()<<", "<<wA->max()<<"]"<<endl;

    cout.setf(ios::scientific, ios::floatfield);
    cout.precision(2);
    //cout<<"\tIncompressibility = "<<incomp()<<endl;
    cout<<"\tQ_qA = "<<qA->Qt();
    cout<<"\tQ_qAc = "<<qAc->Qt()<<endl;
    cout<<"\tResidual Error    = "<<residual_error()<<endl;

    cout.unsetf(ios::floatfield);
    cout.precision(6);
}

void Model_AS::save(const string file){
    phiA->uc().save(file, phiA->Lx(), phiA->Ly(), phiA->Lz());
    save_field(file);
    save_density(file);
}

void Model_AS::save_field(const string file){
    wA->save(file);
}

void Model_AS::save_density(const string file){
    phiA->save(file);
}

void Model_AS::save_q(const string file){
    qA->save(file);
    qAc->save(file);
}

void Model_AS::save_model(const string file){
    CMatFile mat;
    mat.matInit(file,"u");
    if(!mat.queryStatus()){
        mat.matPutScalar("C", C);
        mat.matPutScalar("aA", aA);
        mat.matPutScalar("v", v);
        mat.matPutScalar("dsA", dsA);
        mat.matPutScalar("sA", sA);
        mat.matPutScalar("seedA", wA->seed());
    }
        mat.matPutScalar("dim", _cfg.dim());
        mat.matPutScalar("Lx", phiA->Lx());
        mat.matPutScalar("Ly", phiA->Ly());
        mat.matPutScalar("Lz", phiA->Lz());
        //mat.matPutScalar("lx", phiA->lx());
        //mat.matPutScalar("ly", phiA->ly());
        //mat.matPutScalar("lz", phiA->lz());
        mat.matPutString("crystal_system_type", phiA->uc().type());
        mat.matPutString("gridInitType",_cfg.get_grid_init_type_string());
        
        /*mwSize dim_array[3]={_cfg.Lx(), _cfg.Ly(), _cfg.Lz()};
        blitz::Array<double, 3> data(_cfg.Lx(), _cfg.Ly(), _cfg.Lz(), fortranArray);
        data = delta;
        mat.matPut("delta",data.data(),data.size()*sizeof(double),3,dim_array,mxDOUBLE_CLASS,mxREAL);*/
        mat.matRelease();
}

void Model_AS::display_parameters() const{
    cout<<endl;
    cout<<"********* Model_AS Parameter List **********"<<endl;
    cout<<"Compressibility: ";
    /*if(is_compressible)
        cout<<"Helfand compressible model."<<endl;
    else
        cout<<"Incompressible model."<<endl;*/
    cout<<"Confinement: "<<_cfg.get_confine_type_string()<<endl;
    cout<<"MDE algorithm: "<<_cfg.get_algo_mde_type_string()<<endl;
    cout<<"SCFT algorithm: "<<_cfg.get_algo_scft_type_string()<<endl;
    cout<<"Cell optimization algorithm: ";
    cout<<_cfg.get_algo_cell_optimization_type_string()<<endl;
    cout<<"Contour integration algorithm: ";
    cout<<_cfg.get_algo_contour_integration_type_string()<<endl;
    cout<<endl;

    cout<<"aA = "<<aA<<endl;
    cout<<"dsA = "<<dsA<<endl;
    cout<<"sA = "<<sA<<endl;
    cout<<"seedA = "<<wA->seed()<<endl;
    cout<<endl;

    cout<<"dimension: "<<_cfg.dim()<<endl;
    cout<<"(Lx, Ly, Lz) = ";
    cout<<"("<<phiA->Lx()<<", "<<phiA->Ly()<<", "<<phiA->Lz()<<")"<<endl;
    cout<<"(a, b, c) = ";
    cout<<"("<<phiA->lx()<<", "<<phiA->ly()<<", "<<phiA->lz()<<")"<<endl;

    cout<<"*******************************************"<<endl;
    cout<<endl;
}

void Model_AS::release_memory(){
    delete wA;
    delete phiA;
    delete qA;
    delete qAc;
    if(_cfg.ctype() != ConfineType::NONE
       || (_cfg.algo_mde_type() == AlgorithmMDEType::ETDRK4
           && _cfg.etdrk4_M() > 0)){
        delete ppropupA;
    }
}

Model_AS::~Model_AS(){
    release_memory();
}

void Model_AS::init_random_field(){
    double low = 0.0;
    double high = 1.0;
    int seed = _cfg.seed();
    wA = new Field("wA", _cfg, lamA, low, high, seed);
}

void Model_AS::init_constant_field(){
    double vA = 0.5;
    wA = new Field("wA", _cfg, vA, lamA);
}

void Model_AS::init_file_field(){
    string file = _cfg.field_data_file();
    wA = new Field("wA", _cfg, file, lamA);
}

/*void Model_AS::init_pattern_field(){
    double c = fA<fB?fA:fB;
    double v1 = 0;
    double v2 = 1;
    PhasePattern pt = _cfg.get_phase_pattern();
    wA = new Field("wA", _cfg, lamA);
    Helper::init_pattern((*wA), pt, c, v1, v2);
    Helper::init_pattern((*wB), pt, c, v2, v1);
}*/

void Model_AS::init_field(){
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
        /*case GridInitType::PATTERN_INIT:
            init_pattern_field();
            break;*/
        default:
            cerr<<"Unkonwn or unsupported grid init type!"<<endl;
            exit(1);
    }
}

void Model_AS::init_density(){
    AlgorithmContourType actype = _cfg.algo_contour_integration_type();
    if(actype == AlgorithmContourType::TRAPEZOIDAL){
        phiA = new Density("phiA", _cfg);
    }
    else if(actype == AlgorithmContourType::SIMPSON){
        if(sA%2 != 0)
            phiA = new Density("phiA", _cfg, new Simpson);
        else
            phiA = new Density("phiA", _cfg, new Quad4_Closed);
    }
    else{
        cerr<<"Contour integration algorithm: ";
        cerr<<_cfg.get_algo_contour_integration_type_string();
        cerr<<" is not available."<<endl;
        exit(1);
    }
}

void Model_AS::init_propagator(){
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
            qAc = new Propagator("qAc", _cfg, sA, dsA, one);
        }
        else if(_cfg.algo_mde_type() == AlgorithmMDEType::ETDRK4
                && _cfg.etdrk4_M() > 0){
            // NOTE: ONLY Cox-Matthews scheme has been implemented.
            // Thus the input _cfg.etdrk4_scheme_type() is ignored.
            ppropupA = new Etdrk4_PBC(uc, Lx, Ly, Lz, dsA, _cfg.etdrk4_M());
            qA = new Propagator("qA", _cfg, sA, dsA, one, ppropupA);
            qAc = new Propagator("qAc", _cfg, sA, dsA, one, ppropupA);
        }
        else if(_cfg.algo_mde_type() == AlgorithmMDEType::OS2){
            ppropupA = new PseudoSpectral(uc, Lx, Ly, Lz, dsA);
            qA = new Propagator("qA", _cfg, sA, dsA, one, ppropupA);
            qAc = new Propagator("qAc", _cfg, sA, dsA, one, ppropupA);
        }
        else if(_cfg.algo_mde_type() == AlgorithmMDEType::RQM4){
            ppropupA = new RQM4(uc, Lx, Ly, Lz, dsA);
            qA = new Propagator("qA", _cfg, sA, dsA, one, ppropupA);
            qAc = new Propagator("qAc", _cfg, sA, dsA, one, ppropupA);
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
        // NOTE: ONLY Krogstad scheme has been implemented.
        // Thus the input _cfg.etdrk4_scheme_type() is ignored.
        if(_cfg.etdrk4_M() <= 0){
            ppropupA = new Etdrk4(uc, dim, Lx, Ly, Lz, dsA,
                                  confine_type, lbcA, rbcA);
        }
        else{
            ppropupA = new Etdrk4(uc, dim, Lx, Ly, Lz, dsA,
                                  confine_type, lbcA, rbcA,
                                  ETDRK4SCHEME::KROGSTAD, _cfg.etdrk4_M());
        }
        qA = new Propagator("qA", _cfg, sA, dsA, one, ppropupA);
        qAc = new Propagator("qAc", _cfg, sA, dsA, one, ppropupA);
    }
    else{
        cerr<<"Confinement: "<<_cfg.get_confine_type_string();
        cerr<<" is not available."<<endl;
        exit(1);
    }
}

