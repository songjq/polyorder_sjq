#include "Model_AB_C.h"
#include "CMatFile.h"
#include "Helper.h"

using namespace arma;
using namespace std;
using namespace blitz;

Model_AB_C::Model_AB_C(const string config_file):Model(config_file){
    init();
}

void Model_AB_C::reset(const string& config_data){
    release_memory();
    bool rc = _cfg.set_config_data(config_data);
    if(!rc){
        cerr<<"Setting config data failed. Abort."<<endl;
        exit(1);
    }
    init();
}

void Model_AB_C::init(){
	blitz::Range all = blitz::Range::all();
    is_compressible = _cfg.is_compressible();
    eps = _cfg.get_double("Model","eps");
    LamA = _cfg.get_double("Model","LamA");
    LamB = _cfg.get_double("Model","LamB");
    LamC = _cfg.get_double("Model","LamC");
    cout << "LamA = " << LamA << endl;
    cout << "LamB = " << LamB << endl;
    cout << "LamC = " << LamC << endl;

    vec vf = _cfg.f();
    fA = vf(0);
    fB = vf(1);
    fC = vf(2);
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

    fA = 1.0 * (sA-1) / (sA+sB-2);
    fB = 1.0 - fA;   
    dsA = fA/(sA-1); 
    dsB = fB/(sB-1); 
    dsC = 1.0*eps/(sC-1);

    vec lam = _cfg.lam();
    lamA = lam(0);
    lamB = lam(1);
    lamC = lam(2);
    lamYita = lam(3);
    
    switch (_cfg.dim()) {
        case 1 :
            sigma = fC * _cfg.a() / eps;  // sigma is  \sigma*N/\rho_0  actually
            break;
        case 2 :
            sigma = fC * _cfg.b() / eps;
            break;
        case 3 :
            sigma = fC * _cfg.c() / eps;
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

void Model_AB_C::update(){
    blitz::Range all = blitz::Range::all();
    qA->update(*wA);
    qB->set_head( qA->get_tail() );
    qB->update(*wB);
    qBc->update(*wB);
    qAc->set_head( qBc->get_tail() );
    qAc->update(*wA);
    qCc->update(*wC);
    blitz::Array<double, 4> qc(qC->qs());
    blitz::Array<double, 4> qcc(qCc->qs());
    int Grids = _cfg.Lx() * _cfg.Ly() * _cfg.Lz();

    blitz::firstIndex i;
    blitz::secondIndex j;
    blitz::thirdIndex k;
    switch(_cfg.dim()) {
        case 1:
        {
            blitz::Array<double, 1> data1(delta(all, 0, 0)*qcc(sC-1, all, 0, 0));
            arma::vec v(data1.data(), _cfg.Lx());
            Cheb cheb(_cfg.Lx());
            QC = 0.5 * cheb.quadrature_clencurt(v); //Notice:QC here is already normalized.

            for(int i=0; i<_cfg.Lx(); ++i) 
                qc(0, i, all, all) = sigma * delta(i, all, all) / (QC*_cfg.a()); //delta only deplict half dirac function, should times 2.0
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
                qc(0, all, j, all) = sigma * delta(all, j, all) / (QC*_cfg.b());
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
                qc(0, all, all, k) = sigma * delta(all, all, k) / (QC*_cfg.c());
            break;
        }
        default:
            break;
    }

    qC->update(*wC);
    QAB = qB->Qt();
    phiA->set_cc((1.0-fC)/QAB); //QAB should be ignored for better convergence
    //phiA->set_cc((1.0-fC));   
    phiA->update(*qA, *qAc);
    phiB->set_cc((1.0-fC)/QAB);
    //phiB->set_cc((1.0-fC));   
    phiB->update(*qB, *qBc);
    phiC->set_cc(1.0);
    phiC->update(*qC, *qCc);

    if(is_compressible) {
        wA->update( chiNac * (*phiC) + chiNab * (*phiB) + LamA * (*wH) + lamYita*((*phiA)+(*phiB)+(*phiC)-1.0) );
        wB->update( chiNab * (*phiA) + chiNbc * (*phiC) + LamB * (*wH) + lamYita*((*phiA)+(*phiB)+(*phiC)-1.0) );
        wC->update( chiNac * (*phiA) + chiNbc * (*phiB) + LamC * (*wH) + lamYita*((*phiA)+(*phiB)+(*phiC)-1.0) ); 
    } 
    else {
        yita->update( (*phiA) + (*phiB) + (*phiC) - 1.0 );
        wA->update( chiNac * (*phiC) + chiNab * (*phiB) + LamA * (*wH) + (*yita) );
        wB->update( chiNab * (*phiA) + chiNbc * (*phiC) + LamB * (*wH) + (*yita) );
        wC->update( chiNac * (*phiA) + chiNbc * (*phiB) + LamC * (*wH) + (*yita) );
    }
     
/**********************segment density****************************************/  
    blitz::Array<double, 3> phiab_end(phiAB_end->data());
    blitz::Array<double, 3> phiab_middle(phiAB_middle->data());
    blitz::Array<double, 4> qsA(qA->qs());
    blitz::Array<double, 4> qsAc(qAc->qs());
    phiab_end(all, all, all) = (1-fC)/QAB*qsA(0, all, all, all)*qsAc(sA-1, all, all, all);
    phiab_middle(all, all, all) = (1-fC)/QAB*qsA(sA-1, all, all, all)*qsAc(0, all, all, all); 

    blitz::Array<double, 3> phic_begin(phiC_begin->data());
    blitz::Array<double, 3> phic_end(phiC_end->data());
    blitz::Array<double, 3> phic_middle(phiC_middle->data());
    blitz::Array<double, 4> qsC(qC->qs());
    blitz::Array<double, 4> qsCc(qCc->qs());        
    phic_begin(all, all, all) = qsC(0, all, all, all)*qsCc(sC-1, all, all, all);
    phic_end(all, all, all) = qsC(sC-1, all, all, all)*qsCc(0, all, all, all);
    phic_middle(all, all, all) = qsC((sC-1)/2, all, all, all)*qsCc((sC-1)/2, all, all, all);
}

double Model_AB_C::Hw() const{
    double ret;
    if(is_compressible){
        Grid g = chiNac * (*phiA) * (*phiC) + chiNab * (*phiA) * (*phiB) + chiNbc * (*phiB) * (*phiC)
                 + lamYita/2.0 * ((*phiA) + (*phiB) + (*phiC) - 1.0) * ((*phiA) + (*phiB) + (*phiC) - 1.0)
                 + LamA * (*wH) * (*phiA) + LamB * (*wH) * (*phiB) + LamC * (*wH) * (*phiC)
                 - (*wA) * (*phiA) - (*wB) * (*phiB) - (*wC) * (*phiC);
        ret = g.quadrature();
    }
    else {
        Grid g = chiNac * (*phiA) * (*phiC) + chiNab * (*phiA) * (*phiB) + chiNbc * (*phiB) * (*phiC)
                 + LamA * (*wH) * (*phiA) + LamB * (*wH) * (*phiB) + LamC * (*wH) * (*phiC)
                 - (*wA) * (*phiA) - (*wB) * (*phiB) - (*wC) * (*phiC) +(*yita) * ((*phiA)+(*phiB)+(*phiC)-1.0) ;
        ret = g.quadrature();
    }

    return ret;
}

double Model_AB_C::Hs() const{
    return -(1.0-fC)*log(QAB) - 1.0*fC/eps*log(QC);
}

double Model_AB_C::H() const{
    return Hw() + Hs();             //Free energy in grand canonical ensemble
}

double Model_AB_C::incomp() const{
    Grid g = (*phiA) + (*phiB) + (*phiC) - 1.0;
    return g.abs_quadrature();
}

double Model_AB_C::residual_error() const{
    double res = 0.0;
    /* For Field ONLY */
    double r, x, b;
    Grid g1, g2;

    if(is_compressible) {
        // wA
        g1 = chiNac * (*phiC) + chiNab * (*phiB) + LamA * (*wH) + lamYita*((*phiA)+(*phiB)+(*phiC)-1.0);
        g2 = g1 - (*wA);
        r = g2.abs_mean();
        x = wA->abs_mean();
        b = g1.abs_mean();
        res += r / (x+b);
        // wB
        g1 = chiNab * (*phiA) + chiNbc * (*phiC) + LamB * (*wH) + lamYita*((*phiA)+(*phiB)+(*phiC)-1.0);
        g2 = g1 - (*wB);
        r = g2.abs_mean();
        x = wB->abs_mean();
        b = g1.abs_mean();
        res += r / (x+b);
        // wC
        g1 = chiNac * (*phiA) + chiNbc * (*phiB) + LamC * (*wH) + lamYita*((*phiA)+(*phiB)+(*phiC)-1.0);
        g2 = g1 - (*wC);
        r = g2.abs_mean();
        x = wC->abs_mean();
        b = g1.abs_mean();
        res += r / (x+b);
    
        res /= 3.0;
    }
    else {
        // wA
        g1 = chiNac * (*phiC) + chiNab * (*phiB) + LamA * (*wH) + (*yita);
        g2 = g1 - (*wA);
        r = g2.abs_mean();
        x = wA->abs_mean();
        b = g1.abs_mean();
        res += r / (x+b);
        // wB
        g1 = chiNab * (*phiA) + chiNbc * (*phiC) + LamB * (*wH) + (*yita);
        g2 = g1 - (*wB);
        r = g2.abs_mean();
        x = wB->abs_mean();
        b = g1.abs_mean();
        res += r / (x+b);
        // wC
        g1 = chiNac * (*phiA) + chiNbc * (*phiB) + LamC * (*wH) + (*yita);
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
    
    return res;
}


void Model_AB_C::display() const{
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
    cout<<"\twA   = "<<wA->quadrature();
    cout<<"\t["<<wA->min()<<", "<<wA->max()<<"]"<<endl;
    cout<<"\twB   = "<<wB->quadrature();
    cout<<"\t["<<wB->min()<<", "<<wB->max()<<"]"<<endl;
    cout<<"\twC   = "<<wC->quadrature();
    cout<<"\t["<<wC->min()<<", "<<wC->max()<<"]"<<endl;
    if(!is_compressible) {
        cout<<"\tyita = "<<yita->quadrature();
        cout<<"\t["<<yita->min()<<", "<<yita->max()<<"]"<<endl;
    }
    
    cout.setf(ios::scientific, ios::floatfield);
    cout.precision(2);
    blitz::Array<double, 4> qcc(qCc->qs());
    cout<<"\tQAB = "<<qB->Qt()<<"\tQC = "<<QC<<endl;
    cout<<"\tResidual Error    = "<<residual_error()<<endl;
    cout<<"\tphiA+phiB+phiC-1.0     = "<<(phiA->quadrature()+phiB->quadrature()+phiC->quadrature()-1.0) << endl;

    cout.unsetf(ios::floatfield);
    cout.precision(6);
}

void Model_AB_C::save(const string file){
    phiA->uc().save(file, phiA->Lx(), phiA->Ly(), phiA->Lz());
    save_field(file);
    save_density(file);
}

void Model_AB_C::save_field(const string file){
    wA->save(file);
    wB->save(file);
    wC->save(file);
    wH->save(file); //LamABC *wH is the true short-range potential
    if(!is_compressible)
        yita->save(file);
}

void Model_AB_C::save_density(const string file){
    phiA->save(file);
    phiB->save(file);
    phiC->save(file);

    phiC_begin->save(file);
    phiC_middle->save(file);
    phiC_end->save(file);
    phiAB_middle->save(file);
    phiAB_end->save(file);
    
    CMatFile mat;
    mat.matInit(file,"u");
    if(!mat.queryStatus()){
        mat.matPutScalar("QAB", QAB);
        mat.matPutScalar("QC", QC);
        mat.matPutScalar("f_brush", phiC->quadrature());
        mat.matPutScalar("f_A", phiA->quadrature());
        mat.matPutScalar("f_B", phiB->quadrature());
        mat.matRelease();
    }
}

void Model_AB_C::save_q(const string file){
    qA->save(file);
    qAc->save(file);
    qB->save(file);
    qBc->save(file);
    qC->save(file);
    qCc->save(file);
}

void Model_AB_C::save_model(const string file){
    CMatFile mat;
    mat.matInit(file,"u");
    if(!mat.queryStatus()){
        mat.matPutScalar("fA", fA);
        mat.matPutScalar("fB", fB);
        mat.matPutScalar("fC", fC);
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
        mat.matPutScalar("seedA", wA->seed());
        mat.matPutScalar("seedB", wB->seed());
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

void Model_AB_C::display_parameters() const{
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

    cout<<"fA = "<<fA<<"\tfB = "<<fB<<"\tfC = "<<fC<<endl;
    cout<<"chiNab = "<<chiNab<<"\tchiNac = "<<chiNac<<"\tchiNbc = "<<chiNbc<<endl;
    cout<<"aA = "<<aA<<"\taB = "<<aB<<"\taC = "<<aC<<endl;
    cout<<"dsA = "<<dsA<<"\tdsB = "<<dsB<<"\tdsC = "<<dsC<<endl;
    cout<<"sA = "<<sA<<"\tsB = "<<sB<<"\tsB = "<<sC<<endl;
    if(_cfg.algo_scft_type() == AlgorithmSCFTType::ANDERSON)
        cout<<"seedA = "<<wAx->seed()<<"\tseedC = "<<wCx->seed()<<endl;
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

void Model_AB_C::release_memory(){
    if(_cfg.algo_scft_type() == AlgorithmSCFTType::ANDERSON){
        delete wAx;
        delete wBx;
        delete wCx;
    }
    else{
        delete wA;
        delete wB;
        delete wC;
        delete wH;
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

Model_AB_C::~Model_AB_C(){
    release_memory();
}

void Model_AB_C::init_delta(){

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
            x = cos(kPI*i/(Lx-1));
            x = 0.5 * (1-x) * (_cfg.a());
            delta.resize(Lx, Ly, Lz);
            arr = 1.0/sqrt(2.0*kPI*alpha) * exp(-pow2(x)/2/alpha);// + x*x*exp(-_cfg.a()*_cfg.a()/2/alpha)/(2*alpha*sqrt(2.0*kPI*alpha));
            /*****************brush grafted onto both flats************************/
            //arr = arr / 2.0;         
            //blitz::Array<double, 1> tmparr(arr(Range(0, int(Lx/2-1))));
            //arr(Range(Lx-1-int(Lx/2-1), Lx-1)) = tmparr.reverse(blitz::firstDim);
            /**********************************************************************/
            for(int i=0; i<Lx; ++i)   delta(i, all, all) = arr(i);

            break;
        }
        case 2:
        {                              
            blitz::Array<double, 1> y(Ly);
            blitz::Array<double, 1> arr(Ly);
            y = cos(kPI*i/(Ly-1));        //notice the index is i not j
            y = 0.5 * (1-y) * (_cfg.b());
            delta.resize(Lx, Ly, Lz);
            arr = 1.0/sqrt(2.0*kPI*alpha) * exp(-pow2(y)/2/alpha);// + y*y*exp(-_cfg.b()*_cfg.b()/2/alpha)/(2*alpha*sqrt(2.0*kPI*alpha)); 
            //arr = arr / 2.0;         
            //blitz::Array<double, 1> tmparr(arr(Range(0, int(Ly/2-1))));
            //arr(Range(Ly-1-int(Ly/2-1), Ly-1)) = tmparr.reverse(blitz::firstDim);
            for(int j=0; j<Ly; ++j)   delta(all, j, all) = arr(j);
            break;
        }
        case 3:
        {                              
            blitz::Array<double, 1> z(Lz);
            blitz::Array<double, 1> arr(Lz);
            z = cos(kPI*i/(Lz-1));        //notice the index is i not j
            z = 0.5 * (1-z) * (_cfg.c());
            delta.resize(Lx, Ly, Lz);
            arr = 1.0/sqrt(2.0*kPI*alpha) * exp(-pow2(z)/2/alpha)/2;// + z*z*exp(-_cfg.c()*_cfg.c()/2/alpha)/(2*alpha*sqrt(2.0*kPI*alpha));  
            //arr = arr / 2.0;         
            //blitz::Array<double, 1> tmparr(arr(Range(0, int(Lz/2-1))));
            //arr(Range(Lz-1-int(Lz/2-1), Lz-1)) = tmparr.reverse(blitz::firstDim);
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

void Model_AB_C::init_random_field(){
    double low = 0.0;
    double high = 1.0;
    int seed = _cfg.seed();
    if(_cfg.algo_scft_type() == AlgorithmSCFTType::ANDERSON){
        uword n = _cfg.n_Anderson_mixing();
        wAx = new FieldAX("wA", _cfg, lamA, n, low, high, seed);
        wBx = new FieldAX("wB", _cfg, lamB, n, low, high, seed);
        wCx = new FieldAX("wC", _cfg, lamC, n, low, high, seed);
    }
    else{
        wA = new Field("wA", _cfg, lamA, low, high, seed);
        wB = new Field("wB", _cfg, lamB, low, high, seed);
        wC = new Field("wC", _cfg, lamC, low, high, seed);
    }
}

void Model_AB_C::init_constant_field(){
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

void Model_AB_C::init_file_field(){
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

void Model_AB_C::init_pattern_field(){
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

void Model_AB_C::init_field(){
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
    /***************define short-range potential of wall******************/
    blitz::Range all = blitz::Range::all();
    uword Lx = _cfg.Lx();
    uword Ly = _cfg.Ly();
    uword Lz = _cfg.Lz();
    double ds = 0.3;
    blitz::firstIndex i;
    blitz::Array<double, 3> wh(Lx, Ly, Lz);

    switch (_cfg.dim()) {
        case 1 :
        {
            blitz::Array<double, 1> x(Lx);
            blitz::Array<double, 1> array(Lx);
            x = cos(kPI*i/(Lx-1));
            x = 0.5 * (1-x) * (_cfg.a());
            array = 1.0/ds*exp(-x*x/(2*ds*ds));
            for(int i=0; i<Lx; ++i)   
                wh(i, all, all) = array(i);
            break;
        }
        case 2 :
        {
            blitz::Array<double, 1> y(Ly);
            blitz::Array<double, 1> array(Ly);
            y = cos(kPI*i/(Ly-1));
            y = 0.5 * (1-y) * (_cfg.b());
            array = 1.0/ds*exp(-y*y/(2*ds*ds));
            for(int j=0; j<Ly; ++j)
                wh(all, j, all) = array(j);
            break;
        }
        case 3 :
        {
            blitz::Array<double, 1> z(Lz);
            blitz::Array<double, 1> array(Lz);
            z = cos(kPI*i/(Lz-1));
            z = 0.5 * (1-z) * (_cfg.c());
            array = 1.0/ds*exp(-z*z/(2*ds*ds));
            for(int k=0; k<Lz; ++k)
                wh(all, all, k) = array(k);
            break;
        }
        default :
            cout << "Please input correct dimension !" << endl;
            break;
    }
    wH = new Field("wH", _cfg, wh, lamA);
    /*********************************************************************/
}

void Model_AB_C::init_density(){
    phiAB_end = new Density("phiAB_end", _cfg);
    phiAB_middle = new Density("phiAB_middle", _cfg);
    phiC_begin = new Density("phiC_begin", _cfg);
    phiC_end = new Density("phiC_end", _cfg);
    phiC_middle = new Density("phiC_middle", _cfg);

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

void Model_AB_C::init_propagator(){
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
            qC = new Propagator("qC", _cfg, sC, dsC);
            qAc = new Propagator("qAc", _cfg, sA, dsA);
            qBc = new Propagator("qBc", _cfg, sB, dsB, one);
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
            qB = new Propagator("qB", _cfg, sA, dsB, ppropupB);
            qC = new Propagator("qC", _cfg, sC, dsC, ppropupC);
            qAc = new Propagator("qAc", _cfg, sA, dsA, ppropupA);
            qBc = new Propagator("qBc", _cfg, sA, dsB, one, ppropupB);
            qCc = new Propagator("qCc", _cfg, sC, dsC, one, ppropupC);
        }
        else if(_cfg.algo_mde_type() == AlgorithmMDEType::OS2){
            ppropupA = new PseudoSpectral(uc, Lx, Ly, Lz, dsA);
            ppropupB = new PseudoSpectral(uc, Lx, Ly, Lz, dsB);
            ppropupC = new PseudoSpectral(uc, Lx, Ly, Lz, dsC);
            qA = new Propagator("qA", _cfg, sA, dsA, one, ppropupA);
            qB = new Propagator("qB", _cfg, sA, dsB, ppropupB);
            qC = new Propagator("qC", _cfg, sC, dsC, ppropupC);
            qAc = new Propagator("qAc", _cfg, sA, dsA, ppropupA);
            qBc = new Propagator("qBc", _cfg, sA, dsB, one, ppropupB);
            qCc = new Propagator("qCc", _cfg, sC, dsC, one, ppropupC);
        }
        else if(_cfg.algo_mde_type() == AlgorithmMDEType::RQM4){
            ppropupA = new RQM4(uc, Lx, Ly, Lz, dsA);
            ppropupB = new RQM4(uc, Lx, Ly, Lz, dsB);
            ppropupC = new RQM4(uc, Lx, Ly, Lz, dsC);
            qA = new Propagator("qA", _cfg, sA, dsA, one, ppropupA);
            qB = new Propagator("qB", _cfg, sA, dsB, ppropupB);
            qC = new Propagator("qC", _cfg, sC, dsC, ppropupC);
            qAc = new Propagator("qAc", _cfg, sA, dsA, ppropupA);
            qBc = new Propagator("qBc", _cfg, sB, dsB, one, ppropupB);
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
        //Boundary lbcB(lbcc(0), lbcc(1), lbcc(2));
        //Boundary rbcB(rbcc(0), rbcc(1), rbcc(2));

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
        qC = new Propagator("qC", _cfg, sC, dsC, ppropupC);
        qAc = new Propagator("qAc", _cfg, sA, dsA, ppropupA);
        qBc = new Propagator("qBc", _cfg, sB, dsB, one, ppropupB);
        qCc = new Propagator("qCc", _cfg, sC, dsC, one, ppropupC);
    }
    else{
        cerr<<"Confinement: "<<_cfg.get_confine_type_string();
        cerr<<" is not available."<<endl;
        exit(1);
    }
}

