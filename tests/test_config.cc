#include <iostream>
#include "armadillo"

#include "common.h"
#include "Config.h"

using namespace std;
using namespace arma;

void test_get_list_integer(const string &config){
    Config cfg(config.c_str());
    string section("Model");
    string key("chiN");
    uvec v = cfg.get_list_integer(section, key);
    v.print("v =");
}

void test_get_list_double(const string &config){
    Config cfg(config.c_str());
    string section("Model");
    string key("f");
    vec v = cfg.get_list_double(section, key);
    v.print("v =");
}

void test_all(const string &config){
    cout<<"Begin Config test..."<<endl;
    Config cfg(config.c_str());
    // Will perform internal check during construction.
    cout<<"Successfully loaded."<<!cfg.is_empty()<<endl;

    cout<<"/***** Version Section *****/"<<endl;
    cout<<"Version: "<<cfg.version()<<endl;
    cout<<endl;

    cout<<"/***** IO Section *****/"<<endl;
    cout<<"base_dir: "<<cfg.base_dir()<<endl;
    cout<<"data_file: "<<cfg.data_file()<<endl;
    cout<<"param_file: "<<cfg.param_file()<<endl;
    cout<<"q_file: "<<cfg.q_file()<<endl;
    cout<<"is_display: "<<cfg.is_display()<<endl;
    cout<<"is_save_data: "<<cfg.is_save_data()<<endl;
    cout<<"is_save_q: "<<cfg.is_save_q()<<endl;
    cout<<"display_interval: "<<cfg.display_interval()<<endl;
    cout<<"record_interval: "<<cfg.record_interval()<<endl;
    cout<<"save_interval: "<<cfg.save_interval()<<endl;
    cout<<endl;

    cout<<"/***** Model Section *****/"<<endl;
    cout<<"Model: "<<cfg.get_model_type_string()<<endl;
    cout<<"n_chain: "<<cfg.n_chain()<<endl;
    uvec nb = cfg.n_block();
    nb.print("n_block = ");
    cout<<"n_component: "<<cfg.n_component()<<endl;
    vec bv = cfg.segment_length();
    bv.print("segment length list: ");
    vec f = cfg.f();
    f.print("f = ");
    vec chiN = cfg.chiN();
    chiN.print("chiN =");
    cout<<"is_compressible: "<<cfg.is_compressible()<<endl;
    cout<<"graft_density: "<<cfg.graft_density()<<endl;
    cout<<"excluded_volume: "<<cfg.excluded_volume()<<endl;
    cout<<endl;

    cout<<"/***** UnitCell Section *****/"<<endl;
    cout<<"crystal system type: "<<cfg.get_crystal_system_type_string()<<endl;
    cout<<"a = "<<cfg.a()<<endl;
    cout<<"b = "<<cfg.b()<<endl;
    cout<<"c = "<<cfg.c()<<endl;
    /*
    cfg.a(cfg.a() + 0.1);
    cfg.b(cfg.b() + 0.1);
    cfg.c(cfg.c() + 0.1);
    cout<<"a = "<<cfg.a()<<endl;
    cout<<"b = "<<cfg.b()<<endl;
    cout<<"c = "<<cfg.c()<<endl;
    */
    cout<<"alpha = "<<cfg.alpha()<<endl;
    cout<<"beta = "<<cfg.beta()<<endl;
    cout<<"gamma = "<<cfg.gamma()<<endl;
    cout<<endl;

    cout<<"/***** Grid Section *****/"<<endl;
    cout<<"dimension: "<<cfg.dim()<<endl;
    cout<<"Lx: "<<cfg.Lx()<<endl;
    cout<<"Ly: "<<cfg.Ly()<<endl;
    cout<<"Lz: "<<cfg.Lz()<<endl;
    cout<<"Confinement: "<<cfg.get_confine_type_string()<<endl;
    cout<<"grid_type_x: "<<cfg.get_gtypex_string()<<endl;
    cout<<"grid_type_y: "<<cfg.get_gtypey_string()<<endl;
    cout<<"grid_type_z: "<<cfg.get_gtypez_string()<<endl;
    vec lbv = cfg.BC_coefficients_left();
    lbv.print("left BC:");
    vec rbv = cfg.BC_coefficients_right();
    rbv.print("right BC: ");
    cout<<"grid_init_type: "<<cfg.get_grid_init_type_string()<<endl;
    cout<<"random seed: "<<cfg.seed()<<endl;
    cout<<"field_data: "<<cfg.field_data_file()<<endl;
    cout<<endl;

    cout<<"/***** Algorithm_MDE Section *****/"<<endl;
    cout<<"algorithm: "<<cfg.get_algo_mde_type_string()<<endl;
    vec ds = cfg.ds();
    ds.print("ds = ");
    uvec Ms = cfg.Ms();
    Ms.print("Ms = ");
    cout<<"ETDRK4 scheme: "<<cfg.get_etdrk4_scheme_type_string()<<endl;
    cout<<"ETDRK4 M: "<<cfg.etdrk4_M()<<endl;
    cout<<endl;

    cout<<"/***** Algorithm_SCFT Section *****/"<<endl;
    cout<<"algorithm: "<<cfg.get_algo_scft_type_string()<<endl;
    vec lam = cfg.lam();
    lam.print("lam = ");
    cout<<"min_iter: "<<cfg.min_iter()<<endl;
    cout<<"max_iter: "<<cfg.max_iter()<<endl;
    cout<<"thresh_H: "<<cfg.thresh_H()<<endl;
    cout<<"thresh_residual: "<<cfg.thresh_residual()<<endl;
    cout<<"thresh_incomp: "<<cfg.thresh_incomp()<<endl;
    cout<<"n_Anderson_mixing: "<<cfg.n_Anderson_mixing()<<endl;
    cout<<endl;

    cout<<"/***** Algorithm_Cell_Optimization Section *****/"<<endl;
    cout<<"algorithm: "<<cfg.get_algo_cell_optimization_type_string()<<endl;
    cout<<"tol_cell: "<<cfg.tol_cell()<<endl;
    cout<<"max_iter_cell: "<<cfg.max_iter_cell()<<endl;
    vec min = cfg.batch_cell_min();
    min.print("batch_cell_min = ");
    vec max = cfg.batch_cell_max();
    max.print("batch_cell_max = ");
    vec step = cfg.batch_cell_step();
    step.print("batch_cell_step = ");
    /*
    for(double a=min(0); a<=max(0); a+=step(0))
        for(double b=min(1); b<=max(1); b+=step(1))
            for(double c=min(2); c<=max(2); c+=step(2))
                cout<<"(a,b,c) = "<<"("<<a<<", "<<b<<", "<<c<<")"<<endl;
    */
    cout<<endl;

    cout<<"/***** Algorithm_Contour_Integration Section *****/"<<endl;
    cout<<"algorithm: "<<cfg.get_algo_contour_integration_type_string()<<endl;
    cout<<endl;

    /*
    cfg.save("test_config.ini");
    cfg.clear();
    cout<<"Successfully cleared? "<<cfg.is_empty()<<endl;
    cout<<endl;
    */
}

int main(int argc, char* argv[]){
    string config = "param.ini";
    if(argc > 1){
        config = argv[1];
    }
    test_all(config);

    //test_get_list_integer(config);
    //test_get_list_double(config);
    return 0;
}
