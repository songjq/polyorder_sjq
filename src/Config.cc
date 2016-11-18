#include "Config.h"
#include "common.h"

#include <string>
#include <algorithm>  // for std::transform used in string comparison
#include <iostream>
#include <sstream>

#include "SimpleIni.h"
#include "armadillo"

using namespace std;
using namespace arma;

const string Config::VERSION_SECTION = "Version";
const string Config::IO_SECTION = "IO";
const string Config::MODEL_SECTION = "Model";
const string Config::UNITCELL_SECTION = "UnitCell";
const string Config::GRID_SECTION = "Grid";
const string Config::ALGO_MDE_SECTION = "Algorithm_MDE";
const string Config::ALGO_SCFT_SECTION = "Algorithm_SCFT";
const string Config::ALGO_CELL_SECTION = "Algorithm_Cell_Optimization";
const string Config::ALGO_CONTOUR_SECTION = "Algorithm_Contour_Integration";
const string Config::ALGO_CHARGE_SECTION = "Algorithm_Charge_PBE";

Config::Config(const char *config_file){
    _ini.LoadFile(config_file);
    if(!check()){
        cout<<endl;
        cerr<<"Config internal check FAILED. Please modify your configuration file!";
        cout<<endl;
        exit(1);
    }
}

Config::Config(const string config_data){
    _ini.LoadData(config_data);
    if(!check()){
        cout<<endl;
        cerr<<"Config internal check FAILED. Please modify your configuration file!";
        cout<<endl;
        exit(1);
    }
}

bool Config::get_config_data(string& config_data) const{
    SI_Error rc = _ini.Save(config_data);
    return !(rc < 0);
}

bool Config::set_config_data(const string& config_data){
    clear();
    SI_Error rc = _ini.LoadData(config_data);
    if(!check()){
        cout<<endl;
        cerr<<"Config internal check FAILED. Please modify your configuration file!";
        cout<<endl;
        exit(1);
    }
}

bool Config::save(const string config_file){
    SI_Error rc = _ini.SaveFile(config_file.c_str());
    return !(rc < 0);
}

bool Config::reload_from_file(const string config_file){
    clear();
    SI_Error rc = _ini.LoadFile(config_file.c_str());
    return !(rc < 0);
}

bool Config::is_empty() const{
    bool flag = _ini.IsEmpty();
    if(flag)
        cerr<<"Configuration object is empty!"<<endl;
    return flag;
}

void Config::clear(){
    _ini.Reset();
}

bool Config::check() const{
    bool flag = true;
    bool ret;
    ret = is_empty();
    if(ret) flag = false;
    ret = check_n_block();
    if(!ret) flag = false;
    ret = check_segment_length();
    if(!ret) flag = false;
    ret = check_f();
    if(!ret) flag = false;
    ret = check_chiN();
    if(!ret) flag = false;
    ret = check_BC_left();
    if(!ret) flag = false;
    ret = check_BC_right();
    if(!ret) flag = false;
    ret = check_dim();
    if(!ret) flag = false;
    ret = check_Lx();
    if(!ret) flag = false;
    ret = check_Ly();
    if(!ret) flag = false;
    ret = check_Lz();
    if(!ret) flag = false;
    ret = check_grid_type();
    if(!ret) flag = false;
    ret = check_grid_init_type();
    if(!ret) flag = false;
    ret = check_field_data();
    if(!ret) flag = false;
    ret = check_unitcell_a();
    if(!ret) flag = false;
    ret = check_unitcell_b();
    if(!ret) flag = false;
    ret = check_unitcell_c();
    if(!ret) flag = false;
    ret = check_crystal_system_type();
    if(!ret) flag = false;
    ret = check_lam();
    if(!ret) flag = false;
    ret = check_ds();
    if(!ret) flag = false;
    ret = check_Ms();
    if(!ret) flag = false;
    ret = check_anderson_mixing();
    if(!ret) flag = false;
    ret = check_batch_cell_min();
    if(!ret) flag = false;
    ret = check_batch_cell_max();
    if(!ret) flag = false;
    ret = check_batch_cell_step();
    if(!ret) flag = false;
    return flag;
}

/***********************************************************************
 *                                                                     *
 *                                                                     *
 *        Member functions for Version Section                         *
 *                                                                     *
 *                                                                     *
 ***********************************************************************/

/* Version number
 * Format: <major>.<minor>, mapping to a float number.
 * Requires: <minor> = 0, 1, 2, ..., 9
 * Compare versions, larger one is the latest one.
 * For example:
 *          9.2 > 9.1 > 8.0 > 7.9
 */
double Config::version() const{
    return get_double(VERSION_SECTION, "version");
}

/***********************************************************************
 *                                                                     *
 *                                                                     *
 *        Member functions for IO Section                              *
 *                                                                     *
 *                                                                     *
 ***********************************************************************/

string Config::base_dir() const{
    return get_string(IO_SECTION, "base_dir");
}

string Config::data_file() const{
    return get_string(IO_SECTION, "data_file");
}

string Config::param_file() const{
    return get_string(IO_SECTION, "param_file");
}

string Config::q_file() const{
    return get_string(IO_SECTION, "q_file");
}

bool Config::is_display() const{
    return get_bool(IO_SECTION, "is_display");
}

bool Config::is_save_data() const{
    return get_bool(IO_SECTION, "is_save_data");
}

bool Config::is_save_q() const{
    return get_bool(IO_SECTION, "is_save_q");
}

uword Config::display_interval() const{
    return get_integer(IO_SECTION, "display_interval");
}

uword Config::record_interval() const{
    return get_integer(IO_SECTION, "record_interval");
}

uword Config::save_interval() const{
    return get_integer(IO_SECTION, "save_interval");
}

/***********************************************************************
 *                                                                     *
 *                                                                     *
 *        Member functions for Model Section                           *
 *                                                                     *
 *                                                                     *
 ***********************************************************************/

ModelType Config::model() const{
    string t = get_string(MODEL_SECTION, "model");
    transform(t.begin(), t.end(), t.begin(), ::toupper);

    if(t == "AB" || t == "DIBLOCK" || t == "DIBLOCK COPOLYMER") 
        return ModelType::AB;
    else if(t == "ABW" || t == "AB_W") 
        return ModelType::ABW;
    /*else if(t == "AgC" || t == "A_gC") 
        return ModelType::A_gC;*/
    else if(t == "A_B" || t == "A+B") 
        return ModelType::A_B;
    else if(t == "AS") 
        return ModelType::AS;
    else if(t == "A_S" || t == "A+S") 
        return ModelType::A_S;
    else if(t == "AB_S" || t == "AB+S") 
        {return ModelType::AB_S;cout << "configcc_232" << endl;}
    else if(t == "g_AB_S" || t == "gAB+S") 
        {return ModelType::g_AB_S;cout << "configcc_234" << endl;}
    else if(t == "AB_C" || t == "AB+C") 
        {return ModelType::AB_C;cout << "configcc_236" << endl;}
    else if(t == "ABC" || t == "TRIBLOCK" || t == "TRIBLOCK COPOLYMER") 
        return ModelType::ABC;
    else if(t == "ABm" || t == "A-Bm" || t == "MIKTOARM")
        return ModelType::ABm;
    else if(t == "BmABm" || t == "Bm-A-Bm")
        return ModelType::BmABm;
    else if(t == "BABm" || t == "B-A-Bm")
        return ModelType::BABm;
    else if(t == "STAR" || t == "Am")
        return ModelType::STAR;
    /*else
        return ModelType::AB;*/
}

string Config::get_model_type_string() const{
    string type;
    switch(model()){
        case ModelType::AB:
            type = "AB";
            break;
        case ModelType::ABW:
            type = "ABW";
            break;
        /*case ModelType::A_gC:
            type = "A_gC";
            break;*/
        case ModelType::A_B:
            type = "A+B";
            break;
        case ModelType::AS:
            type = "AS";
            break;
        case ModelType::A_S:
            type = "A+S";
            break;
        case ModelType::AB_S:
            type = "AB+S";
            break;
        case ModelType::g_AB_S:
            type = "AB+S";
            break;
        case ModelType::AB_C:
            type = "AB+C";
            break;
        case ModelType::ABC:
            type = "ABC";
            break;
        case ModelType::ABm:
            type = "ABm";
            break;
        case ModelType::BmABm:
            type = "Bm-A-Bm";
            break;
        case ModelType::BABm:
            type = "B-A-Bm";
            break;
        case ModelType::STAR:
            type = "Star";
            break;
    }
    return type;
}

MaskType Config::mask() const{
    string t = get_string(MODEL_SECTION, "mask");
    transform(t.begin(), t.end(), t.begin(), ::toupper);

    if(t == "COS")
        return MaskType::COS;
    else if(t == "EXP")
        return MaskType::EXP;
    else if(t == "TANH")
        return MaskType::TANH;
    else
        return MaskType::EXP;
}

string Config::get_mask_type_string() const{
    string type;
    switch(mask()){
        case MaskType::COS:
            type = "COS";
            break;
        case MaskType::EXP:
            type = "EXP";
            break;
        case MaskType::TANH:
            type = "TANH";
            break;
        }

    return type;
}

EnsembleType Config::ensemble() const{
    string t = get_string(MODEL_SECTION, "ensemble");
    //cout << "t is " << t << endl;
    transform(t.begin(), t.end(), t.begin(), ::toupper);

    if(t == "canonical")
    {
        cout << "configcc_337" << endl;
        return EnsembleType::canonical;
    }
    else if(t == "grandCanonical")
    {
        cout << "configcc_342" << endl;
        return EnsembleType::grandCanonical;

    }
    else
    {
        //cout << "configcc_348" << endl;
        return EnsembleType::grandCanonical;
    }
}

string Config::get_ensemble_type_string() const{
    string type;
    switch(ensemble()){
        case EnsembleType::canonical:
            type = "canonical";
            break;
        case EnsembleType::grandCanonical:
            type = "grandCanonical";
            break;
        }

    return type;
}

/*
 * Number of chains.
 * Example:
 *      AB + BC + C + M
 *      n_chains = 4
 */
uword Config::n_chain() const{
    return get_integer(MODEL_SECTION, "n_chain");
}

/* A list contains number of blocks for each chain.
 * For homopolymer and small molecules, n_block = 1.
 * We refer blocks, homopolymers, and small molecules to components.
 */
uvec Config::n_block() const{
    return get_list_integer(MODEL_SECTION, "n_block");
}

bool Config::check_n_block() const{
    bool flag = true;
    uvec v = n_block();
    if(v.n_elem != n_chain()){
        cerr<<"n_block and n_chain not match!"<<endl;
        flag = false;
    }
    return flag;
}

/* number of components.
 *      n_component = sum_{i=1}{n_specie} n_block(i)
 */
uword Config::n_component() const{
    uvec nb = n_block();
    return sum(nb);
} 	

/* The physical segment length of each component.
 * Taking AB + C + M as an example,
 * where AB is an A-B diblock copolymer),
 * C is a homopolymer,
 * M is a small molecules.
 * Then, the returning list is
 *      [bA, bB, bC, bM]
 */
vec Config::segment_length() const{
    return get_list_double(MODEL_SECTION, "a");
}

bool Config::check_segment_length() const{
    bool flag = true;
    vec v = segment_length();
    if(v.n_elem != n_component()){
        cerr<<"segment length and n_component not match!"<<endl;
        flag = false;
    }
    return flag;
}

/* The number fraction of each component.
 * Note the meaning of f depend on the choice of base.
 * For example, for an AB + C system, the list of f is
 *      [0.2, 0.8, 0.6]
 * which may mean
 *      f_A = N_A / N_AB
 *      f_B = N_B / N_AB
 *      f_C = N_C / N_AB
 * where N_AB = N_A + N_B.
 * Or, equivalently, f may write as
 *      [0.125, 0.5, 0.375]
 * which means
 *      f_A = N_A / N_ABC
 *      f_B = N_B / N_ABC
 *      f_C = N_C / N_ABC
 * where N_ABC = N_A + N_B + N_C.
 */
vec Config::f() const{
    return get_list_double(MODEL_SECTION, "f");
}

vec Config::wall_affinity() const{
    return get_list_double(MODEL_SECTION, "wall_affinity");
}

bool Config::check_f() const{
    bool flag = true;
    vec v = f();
    if(v.n_elem != n_component()){
        cerr<<"elements in f and n_component not match!"<<endl;
        flag = false;
    }
    return flag;
}

/* The chi*N of each pair of components.
 * For n components, there are at most
 *      n*(n-1)/2
 * chiNs, where n = sum_{i=1}{n_species} n_block(i).
 * These chiNs are listed in the following way illustrated by an example.
 *       AB + C + M
 *   [chiN_AB, chiN_AC, chiN_AM, chiN_BC, chiN_BM, chiN_CM]
 */
vec Config::chiN() const{
    return get_list_double(MODEL_SECTION, "chiN");
}

double Config::depletion_length() const{
    return get_double(MODEL_SECTION, "depletion_length");
}

double Config::length_ratio() const{
    return get_double(MODEL_SECTION, "length_ratio");
}

bool Config::check_chiN() const{
    bool flag = true;
    vec v = chiN();
    uword n = n_component();
    uword n_chiN = n * (n-1) / 2;
    if(v.n_elem != n_chiN){
        cerr<<"elements in chiN and n_component not match!"<<endl;
        flag = false;
    }
    return flag;
}

bool Config::is_compressible() const{
    return get_bool(MODEL_SECTION, "is_compressible");
}

double Config::graft_density() const{
    return get_double(MODEL_SECTION, "graft_density");
}

double Config::graft_area() const{
    return get_double(MODEL_SECTION, "graft_area");
}

double Config::excluded_volume() const{
    return get_double(MODEL_SECTION, "excluded_volume");
}

/***********************************************************************
 *                                                                     *
 *                                                                     *
 *        Member functions for UnitCell Section                        *
 *                                                                     *
 *                                                                     *
 ***********************************************************************/

CrystalSystemType Config::get_crystal_system_type() const{
    string t = get_string(UNITCELL_SECTION, "CrystalSystemType");
    transform(t.begin(), t.end(), t.begin(), ::toupper);

    if(t == "LAM" || t == "LAMELLAR" || t == "LAMELLA"
        || t == "LAMELLAE")
        return CrystalSystemType::LAMELLAR;
    else if(t == "SQUARE" || t == "SQ")
        return CrystalSystemType::SQUARE;
    else if(t == "RECT" || t == "RECTANGLE" || t == "RECTANGULAR")
        return CrystalSystemType::RECTANGULAR;
    else if(t == "HEX" || t == "HEXAGONAL" || t == "HEXGON")
        return CrystalSystemType::HEXAGONAL;
    else if(t == "OBLIQUE" || t == "OB")
        return CrystalSystemType::OBLIQUE;
    else if(t == "CUBIC" || t == "CUBE")
        return CrystalSystemType::CUBIC;
    else if(t == "TETRAGONAL" || t == "TETRAGON")
        return CrystalSystemType::TETRAGONAL;
    else if(t == "ORTHORHOMBIC")
        return CrystalSystemType::ORTHORHOMBIC;
    else if(t == "TRIGONAL")
        return CrystalSystemType::TRIGONAL;
    else if(t == "MONOCLINIC")
        return CrystalSystemType::MONOCLINIC;
    else if(t == "TRICLINIC")
        return CrystalSystemType::TRICLINIC;
    else
        return CrystalSystemType::UNKNOWN;
}

string Config::get_crystal_system_type_string(CrystalSystemType cstype){
    switch(cstype){
        case CrystalSystemType::LAMELLAR:
            return "LAM";
        case CrystalSystemType::SQUARE:
            return "Square";
        case CrystalSystemType::RECTANGULAR:
            return "Rectangular";
        case CrystalSystemType::HEXAGONAL:
            return "HEX";
        case CrystalSystemType::OBLIQUE:
            return "Oblique";
        case CrystalSystemType::CUBIC:
            return "Cubic";
        case CrystalSystemType::TETRAGONAL:
            return "Tetragonal";
        case CrystalSystemType::ORTHORHOMBIC:
            return "Orthorhombic";
        case CrystalSystemType::TRIGONAL:
            return "Trigonal";
        case CrystalSystemType::MONOCLINIC:
            return "Monoclinic";
        case CrystalSystemType::TRICLINIC:
            return "Triclinic";
        case CrystalSystemType::UNKNOWN:
            return "Unknown";
    }
}

string Config::get_crystal_system_type_string() const{
    CrystalSystemType cstype = get_crystal_system_type();
    return get_crystal_system_type_string(cstype);
}

bool Config::check_crystal_system_type() const{
    bool flag = true;
    CrystalSystemType cstype = get_crystal_system_type();
    if(cstype == CrystalSystemType::UNKNOWN){
        cerr<<"UnitCell.CrystalSystemType should be set correctly!"<<endl;
        flag = false;
    }
    uword d = dim();
    if(d == 1 && cstype != CrystalSystemType::LAMELLAR){
        cerr<<"only LAMELLAR is allowed in UnitCell.CrystalSystemType ";
        cerr<<"for 1D."<<endl;
        flag = false;
    }
    if(d == 2 && (cstype != CrystalSystemType::SQUARE
                  && cstype != CrystalSystemType::RECTANGULAR
                  && cstype != CrystalSystemType::HEXAGONAL
                  && cstype != CrystalSystemType::OBLIQUE)){
        cerr<<"only SQUARE, RECTANGULAR, HEXAGONAL, OBLIQUE ";
        cerr<<"are allowed in UnitCell.CrystalSystemType for 2D."<<endl;
        flag = false;
    }
    if(d == 3 && (cstype != CrystalSystemType::HEXAGONAL
                  && cstype != CrystalSystemType::CUBIC
                  && cstype != CrystalSystemType::TETRAGONAL
                  && cstype != CrystalSystemType::ORTHORHOMBIC
                  && cstype != CrystalSystemType::TRIGONAL
                  && cstype != CrystalSystemType::MONOCLINIC
                  && cstype != CrystalSystemType::TRIGONAL
                  && cstype != CrystalSystemType::MONOCLINIC
                  && cstype != CrystalSystemType::TRICLINIC)){
        cerr<<"only HEXAGONAL, CUBIC, TETRAGONAL, ORTHORHOMBIC, ";
        cerr<<"TRIGONAL, MONOCLINIC, TRIGONAL, MONOCLINIC, TRICLINIC ";
        cerr<<"are allowed in UnitCell.CrystalSystemType for 3D."<<endl;
        flag = false;
    }
    return flag;
}

double Config::a() const{
    return get_double(UNITCELL_SECTION, "a");
}

bool Config::check_unitcell_a() const{
    bool flag = true;
    double val = a();
    if(val <= 0){
        cerr<<"the lenght of unitcell vector a should > 0!"<<endl;
        flag = false;
    }
    return flag;
}

double Config::b() const{
    return get_double(UNITCELL_SECTION, "b");
}

bool Config::check_unitcell_b() const{
    bool flag = true;
    uword d = dim();
    double val = b();
    if(val <= 0 && d > 1){
        cerr<<"the lenght of unitcell vector b should > 0 for 2D & 3D!"<<endl;
        flag = false;
    }
    if(val > 0 && d < 2){
        cerr<<"the lenght of unitcell vector b should have no value!"<<endl;
        flag = false;
    }
    return flag;
}

double Config::c() const{
    return get_double(UNITCELL_SECTION, "c");
}

bool Config::check_unitcell_c() const{
    bool flag = true;
    uword d = dim();
    double val = c();
    if(val <= 0 && d > 2){
        cerr<<"the lenght of unitcell vector c should > 0 for 3D!"<<endl;
        flag = false;
    }
    if(val > 0 && d < 3){
        cerr<<"the lenght of unitcell vector c should have no value!"<<endl;
        flag = false;
    }
    return flag;
}

double Config::a(double val){
    bool ret = set_double(UNITCELL_SECTION, "a", val);
    if(!ret){
        cerr<<"Error: set UnitCell::a failed."<<endl;
    }
    return (double)ret;
}

double Config::b(double val){
    bool ret = set_double(UNITCELL_SECTION, "b", val);
    if(!ret){
        cerr<<"Error: set UnitCell::b failed."<<endl;
    }
    return (double)ret;
}

double Config::c(double val){
    bool ret = set_double(UNITCELL_SECTION, "c", val);
    if(!ret){
        cerr<<"Error: set UnitCell::c failed."<<endl;
    }
    return (double)ret;
}

/* Convert degree to radian internally */
double Config::alpha() const{
    return (PI / 180.0) * get_double(UNITCELL_SECTION, "alpha");
}

/* Convert degree to radian internally */
double Config::beta() const{
    return (PI / 180.0) * get_double(UNITCELL_SECTION, "beta");
}

/* Convert degree to radian internally */
double Config::gamma() const{
    return (PI / 180.0) * get_double(UNITCELL_SECTION, "gamma");
}

/***********************************************************************
 *                                                                     *
 *                                                                     *
 *        Member functions for Grid Section                            *
 *                                                                     *
 *                                                                     *
 ***********************************************************************/

uword Config::dim() const{
    return get_integer(GRID_SECTION, "dimension");
}

bool Config::check_dim() const{
    bool flag = true;
    uword d = dim();
    if(d == 0 || d > 3){
        cerr<<"Space dimension must be 1, 2, or 3."<<endl;
        flag = false;
    }
    return flag;
}

uword Config::Lx() const{
    return get_integer(GRID_SECTION, "Lx");
}

bool Config::check_Lx() const{
    bool flag = true;
    uword L = Lx();
    if(L < 1){
        cerr<<"Lx must be larger than 1."<<endl;
        flag = false;
    }
    return flag;
}

uword Config::Ly() const{
    return get_integer(GRID_SECTION, "Ly");
}

bool Config::check_Ly() const{
    bool flag = true;
    uword d = dim();
    uword L = Ly();
    if(L != 1 && d == 1){
        cerr<<"Ly must be 1 for 1D."<<endl;
        flag = false;
    }
    if(L < 1 && d > 1){
        cerr<<"Ly must be larger than 1 for 2D & 3D."<<endl;
        flag = false;
    }
    return flag;
}

uword Config::Lz() const{
    return get_integer(GRID_SECTION, "Lz");
}

bool Config::check_Lz() const{
    bool flag = true;
    uword d = dim();
    uword L = Lz();
    if(L != 1 && d < 3){
        cerr<<"Lz must be 1 for 2D & 3D."<<endl;
        flag = false;
    }
    if(L < 1 && d > 2){
        cerr<<"Lz must be larger than 1 for 3D."<<endl;
        flag = false;
    }
    return flag;
}

ConfineType Config::ctype() const{
    string t = get_string(GRID_SECTION, "confine_geometry");
    transform(t.begin(), t.end(), t.begin(), ::toupper);

    if(t == "LINE" || t == "SLIT" || t == "SLAB" || t == "CUBE")
        return ConfineType::CUBE;
    else if(t == "DISK")
        return ConfineType::DISK;
    else if(t == "CYLINDER")
        return ConfineType::CYLINDER;
    else if(t == "SPHERE")
        return ConfineType::SPHERE;
    else
        return ConfineType::NONE;
}

string Config::get_confine_type_string() const{
    string type;
    switch(ctype()){
        case ConfineType::CUBE:
            type = "Cube";
            break;
        case ConfineType::DISK:
            type = "Disk";
            break;
        case ConfineType::CYLINDER:
            type = "Cylinder";
            break;
        case ConfineType::SPHERE:
            type = "Sphere";
            break;
        case ConfineType::NONE:
            type = "None";
            break;
    }
    return type;
}

GridType Config::get_grid_type(const string key) const{
    string t = get_string(GRID_SECTION, key);
    transform(t.begin(), t.end(), t.begin(), ::toupper);

    if(t == "R" || t == "REGULAR" || t == "NORMAL")
        return GridType::REGULAR;
    else if(t == "C" || t == "CGL" || t == "CHEBYSHEV" ||
            t == "CHEBYSHEV-GAUSS-LOBATTO")
        return GridType::CHEBYSHEV_GAUSS_LOBATTO;
    else
        return GridType::REGULAR;
}

string Config::get_grid_type_string(GridType gtype){
    switch(gtype){
        case GridType::REGULAR:
            return "Regular";
        case GridType::CHEBYSHEV_GAUSS_LOBATTO:
            return "Chebyshev-Gauss-Lobatto";
    }
}

void Config::set_grid_type(const string key, const GridType gtype){
    string t;
    switch(gtype){
        case GridType::REGULAR:
            t = "Regular";
            break;
        case GridType::CHEBYSHEV_GAUSS_LOBATTO:
            t = "Chebyshev_Gauss_Lobatto";
            break;
        default:
            t = "Regular";
    }
    set_string(GRID_SECTION, key, t);
}

GridType Config::gtypex() const{
    return get_grid_type("grid_type_x");
}

string Config::get_gtypex_string() const{
    return get_grid_type_string(gtypex());
}

GridType Config::gtypey() const{
    return get_grid_type("grid_type_y");
}

string Config::get_gtypey_string() const{
    return get_grid_type_string(gtypey());
}

GridType Config::gtypez() const{
    return get_grid_type("grid_type_z");
}

string Config::get_gtypez_string() const{
    return get_grid_type_string(gtypez());
}

bool Config::check_grid_type() const{
    bool flag = true;
    if(ctype() == ConfineType::NONE && gtypex() != GridType::REGULAR){
        cerr<<"ONLY Regular grid in x is allowed for no confinement!"<<endl;
        flag = false;
    }
    if(ctype() == ConfineType::NONE && gtypey() != GridType::REGULAR){
        cerr<<"ONLY Regular grid in y is allowed for no confinement!"<<endl;
        flag = false;
    }
    if(ctype() == ConfineType::NONE && gtypez() != GridType::REGULAR){
        cerr<<"ONLY Regular grid in z is allowed for no confinement!"<<endl;
        flag = false;
    }
    return flag;
}

void Config::set_gtypex(const GridType gtype){
    set_grid_type("grid_type_x", gtype);
}

void Config::set_gtypey(const GridType gtype){
    set_grid_type("grid_type_y", gtype);
}

void Config::set_gtypez(const GridType gtype){
    set_grid_type("grid_type_z", gtype);
}

/*
 * A three-element vector describes the boundary condition:
 *      alpha * du/dx + beta * u = gamma
 * the vector is [alpha, beta, gamma]
 * See the definition of DBC, NBC, RBC, and PBC in cheb++ package.
 */
vec Config::BC_coefficients_left() const{
    return get_list_double(GRID_SECTION, "BC_coefficients_left");
}

bool Config::check_BC_left() const{
    bool flag = true;
    vec v = BC_coefficients_left();
    if(v.n_elem != 3 && ctype() != ConfineType::NONE){
        cerr<<"Number of left boundary coefficients must be 3!"<<endl;
        flag = false;
    }
    return flag;
}

vec Config::BC_coefficients_right() const{
    return get_list_double(GRID_SECTION, "BC_coefficients_right");
}

bool Config::check_BC_right() const{
    bool flag = true;
    vec v = BC_coefficients_right();
    if(v.n_elem != 3 && ctype() != ConfineType::NONE){
        cerr<<"Number of right boundary coefficients must be 3!"<<endl;
        flag = false;
    }
    return flag;
}

vec Config::BC_coefficients_left_brush() const{
    return get_list_double(GRID_SECTION, "BC_coefficients_left_brush");
}

bool Config::check_BC_left_brush() const{
    bool flag = true;
    vec v = BC_coefficients_left_brush();
    if(v.n_elem != 3 && ctype() != ConfineType::NONE){
        cerr<<"Number of left boundary coefficients for brush must be 3!"<<endl;
        flag = false;
    }
    return flag;
}

vec Config::BC_coefficients_right_brush() const{
    return get_list_double(GRID_SECTION, "BC_coefficients_right_brush");
}

bool Config::check_BC_right_brush() const{
    bool flag = true;
    vec v = BC_coefficients_right_brush();
    if(v.n_elem != 3 && ctype() != ConfineType::NONE){
        cerr<<"Number of right boundary coefficients for brush must be 3!"<<endl;
        flag = false;
    }
    return flag;
}

GridInitType Config::get_grid_init_type() const{
    string t = get_string(GRID_SECTION, "gridInitType");
    transform(t.begin(), t.end(), t.begin(), ::toupper);

    if(t == "RANDOM" || t == "RAN")
        return GridInitType::RANDOM_INIT;
    else if(t == "CONST" || t == "CONSTANT")
        return GridInitType::CONSTANT_INIT;
    else if(t == "DATA")
        return GridInitType::DATA_INIT;
    else if(t == "FILE")
        return GridInitType::FILE_INIT;
    else if(t == "PATTERN")
        return GridInitType::PATTERN_INIT;
    else
        return GridInitType::RANDOM_INIT;
}

string Config::get_grid_init_type_string(GridInitType gitype){
    string type;
    switch(gitype){
        case GridInitType::RANDOM_INIT:
            type = "Random";
            break;
        case GridInitType::FILE_INIT:
            type = "File";
            break;
        case GridInitType::DATA_INIT:
            type = "Data";
            break;
        case GridInitType::CONSTANT_INIT:
            type = "Constant";
            break;
        case GridInitType::PATTERN_INIT:
            type = "Pattern";
            break;
    }
    return type;
}

string Config::get_grid_init_type_string() const{
    GridInitType gitype = get_grid_init_type();
    return get_grid_init_type_string(gitype);
}

bool Config::check_grid_init_type() const{
    bool flag = true;
    GridInitType gitype = get_grid_init_type();
    AlgorithmSCFTType astype = algo_scft_type();
    if(astype == AlgorithmSCFTType::ANDERSON
        && gitype != GridInitType::FILE_INIT){
        cerr<<"Anderson mixing requires field file initialization."<<endl;
        flag = false;
    }
    return flag;
}

void Config::set_grid_init_type(const GridInitType git){
    string gitype_str = get_grid_init_type_string(git);
    set_string(GRID_SECTION, "gridInitType", gitype_str);
}

uword Config::seed() const{
    return get_integer(GRID_SECTION, "random_seed");
}

string Config::field_data_file() const{
    return get_string(GRID_SECTION, "field_data");
}

void Config::set_field_data_file(const string file){
    set_string(GRID_SECTION, "field_data", file);
}

bool Config::check_field_data() const{
    bool flag = true;
    GridInitType gitype = get_grid_init_type();
    string file = field_data_file();
    if(gitype == GridInitType::FILE_INIT && file.empty()){
        cerr<<"field_data file is required for file initialization."<<endl;
        flag = false;
    }
    return flag;
}

/*
 * This function is not used currently, use File init instead.
 */
PhasePattern Config::get_phase_pattern() const{
    string p = get_string(GRID_SECTION, "phase_pattern");

    for(int pattern=LAM1_PATTERN; pattern<NUM_PHASE_PATTERN; pattern++)
        if(p.c_str() == PhasePatterns[pattern])
            return static_cast<PhasePattern>(pattern);

    switch(int d = dim()){
        case 1:
            return LAM2_PATTERN;
        case 2:
            return LAM1_PATTERN;
        case 3:
            return GYROID_PATTERN;
        default:
            return GYROID_PATTERN;
    }
}

void Config::set_phase_pattern(const PhasePattern pt){
    set_string(GRID_SECTION, "phase_pattern", PhasePatterns[pt]);
}

/***********************************************************************
 *                                                                     *
 *                                                                     *
 *        Member functions for Algorithm_MDE Section                   *
 *                                                                     *
 *                                                                     *
 ***********************************************************************/

/*
 * Algorithms for solving modified diffusion equations.
 * Default is the ETDRK4 method.
 */
AlgorithmMDEType Config::algo_mde_type() const{
    string t = get_string(ALGO_MDE_SECTION, "algorithm");
    transform(t.begin(), t.end(), t.begin(), ::toupper);

    if(t == "PSEUDOSPECTRAL" || t == "OS2" || t == "OS")
        return AlgorithmMDEType::OS2;
    else if(t == "RQM4" || t == "OS4")
        return AlgorithmMDEType::RQM4;
    else if(t == "ETDRK4" || t == "ETDRK")
        return AlgorithmMDEType::ETDRK4;
    else
        return AlgorithmMDEType::ETDRK4;
}

string Config::get_algo_mde_type_string() const{
    string type;
    switch(algo_mde_type()){
        case AlgorithmMDEType::OS2:
            type = "Psudospectral OS2";
            break;
        case AlgorithmMDEType::RQM4:
            type = "RQM4";
            break;
        case AlgorithmMDEType::ETDRK4:
            type = "ETDRK4";
            break;
    }
    return type;
}

/* Size of contour step for each component.
 *      AB + C + M
 *   [ds_A, ds_B, ds_C, 1.0]
 * For small molecules, ds = 1.0
 */
vec Config::ds() const{
    return get_list_double(ALGO_MDE_SECTION, "ds");
}

bool Config::check_ds() const{
    bool flag = true;
    vec v = ds();
    if(v.n_elem != n_component()){
        cerr<<"elements in ds and n_component not match!"<<endl;
        flag = false;
    }
    return flag;
}

/* Number of contour steps for each component.
 *      AB + C + M
 *   [Ms_A, Ms_B, Ms_C, 1]
 * For small molecules, Ms = 1.
 *      ds_A = f_A / Ms_A
 *      ds_B = f_B / Ms_B
 *      ds_C = f_C / Ms_C
 * Use either ds() or Ms().
 */
uvec Config::Ms() const{
    return get_list_integer(ALGO_MDE_SECTION, "Ms");
}

bool Config::check_Ms() const{
    bool flag = true;
    uvec v = Ms();
    if(v.n_elem != n_component()){
        cerr<<"elements in Ms and n_component not match!"<<endl;
        flag = false;
    }
    return flag;
}

/*
 * Schemes for the ETDRK4 method.
 * Default is Cox-Matthews.
 */
ETDRK4SCHEME Config::etdrk4_scheme_type() const{
    string t = get_string(ALGO_MDE_SECTION, "etdrk4_scheme");
    transform(t.begin(), t.end(), t.begin(), ::toupper);

    if(t == "COX" || t == "COX-MATTHEWS")
        return ETDRK4SCHEME::COX;
    else if(t == "KROGSTAD")
        return ETDRK4SCHEME::KROGSTAD;
    else
        return ETDRK4SCHEME::COX;
}

string Config::get_etdrk4_scheme_type_string() const{
    string type;
    switch(etdrk4_scheme_type()){
        case ETDRK4SCHEME::COX:
            type = "Cox-Matthews";
            break;
        case ETDRK4SCHEME::KROGSTAD:
            type = "Krogstad";
            break;
    }
    return type;
}

/*
 * Number of discrete points along contour for ETDRK4 coefficients compuation.
 */
uword Config::etdrk4_M() const{
    return get_integer(ALGO_MDE_SECTION, "etdrk4_M");
}

/***********************************************************************
 *                                                                     *
 *                                                                     *
 *        Member functions for Algorithm_SCFT Section                  *
 *                                                                     *
 *                                                                     *
 ***********************************************************************/

/*
 * Algorithms for updating SCFT equations.
 * Default is the EM scheme.
 */
AlgorithmSCFTType Config::algo_scft_type() const{
    string t = get_string(ALGO_SCFT_SECTION, "algorithm");
    transform(t.begin(), t.end(), t.begin(), ::toupper);

    if(t == "EM" || t == "EULER")
        return AlgorithmSCFTType::EM;
    else if(t == "ANDERSON" ||
            t == "ANDERSON_MIXING" ||
            t == "ANDERSON MIXING")
        return AlgorithmSCFTType::ANDERSON;
    else if(t == "SIS")
        return AlgorithmSCFTType::SIS;
    else if(t == "ETD")
        return AlgorithmSCFTType::ETD;
    else
        return AlgorithmSCFTType::EM;
}

string Config::get_algo_scft_type_string() const{
    string type;
    switch(algo_scft_type()){
        case AlgorithmSCFTType::EM:
            type = "EM";
            break;
        case AlgorithmSCFTType::ANDERSON:
            type = "Anderson mixing";
            break;
        case AlgorithmSCFTType::SIS:
            type = "SIS";
            break;
        case AlgorithmSCFTType::ETD:
            type = "ETD";
            break;
    }
    return type;
}

/* Relaxation parameters for each component.
 *      AB + C + M
 *   [lamA, lamB, lamC, lamM]
 */
vec Config::lam() const{
    return get_list_double(ALGO_SCFT_SECTION, "lam");
}

bool Config::check_lam() const{
    bool flag = true;
    vec v = lam();

    // For both incompressible and compressible model,
    // there is an extra lam term.
    // For incompressible model, it is the relaxation coefficient.
    // For compressible model, it is the compressibility parameter.
    if(v.n_elem != n_component() + 1){
        cerr<<"elements in lam != n_component+1."<<endl;
        flag = false;
    }

    return flag;
}

uword Config::min_iter() const{
    return get_integer(ALGO_SCFT_SECTION, "min_iter");
}

uword Config::max_iter() const{
    return get_integer(ALGO_SCFT_SECTION, "max_iter");
}

double Config::thresh_H() const{
    return get_double(ALGO_SCFT_SECTION, "thresh_H");
}

double Config::thresh_residual() const{
    return get_double(ALGO_SCFT_SECTION, "thresh_residual");
}

double Config::thresh_incomp() const{
    return get_double(ALGO_SCFT_SECTION, "thresh_incomp");
}

uword Config::n_Anderson_mixing() const{
    return get_integer(ALGO_SCFT_SECTION, "n_Anderson_mixing");
}

bool Config::check_anderson_mixing() const{
    bool flag = true;
    uword val = n_Anderson_mixing();
    AlgorithmSCFTType astype = algo_scft_type();

    if(astype == AlgorithmSCFTType::ANDERSON && val < 1){
        cerr<<"n_Anderson_mixing must > 0."<<endl;
        flag = false;
    }
    if(astype == AlgorithmSCFTType::ANDERSON
        && get_grid_init_type() != GridInitType::FILE_INIT){
        cerr<<"n_Anderson_mixing needs input file to initialize. ";
        cerr<<"Change gridInitType to File."<<endl;
        flag = false;
    }
    return flag;
}

/***********************************************************************
 *                                                                     *
 *                                                                     *
 *        Member functions for Algorithm_Cell_Optimization Section     *
 *                                                                     *
 *                                                                     *
 ***********************************************************************/

/*
 * Algorithms for optimization cell size.
 * Default is no optimization.
 */
AlgorithmCellType Config::algo_cell_optimization_type() const{
    string t = get_string(ALGO_CELL_SECTION, "algorithm");
    transform(t.begin(), t.end(), t.begin(), ::toupper);

    if(t == "NO" || t == "NONE" || t == "SINGLE")
        return AlgorithmCellType::SINGLE;
    else if(t == "MANUAL" || t == "BATCH")
        return AlgorithmCellType::MANUAL;
    else if(t == "BRENT" || t == "BRENT METHOD" || t == "NEWTON")
        return AlgorithmCellType::BRENT;
    else if(t == "VARIABLE" || t == "VARIABLE CELL" ||
            t == "VARIABLE CELL METHOD" )
        return AlgorithmCellType::VARIABLE;
    else
        return AlgorithmCellType::SINGLE;
}

string Config::get_algo_cell_optimization_type_string() const{
    string type;
    switch(algo_cell_optimization_type()){
        case AlgorithmCellType::SINGLE:
            type = "None";
            break;
        case AlgorithmCellType::MANUAL:
            type = "Manual";
            break;
        case AlgorithmCellType::BRENT:
            type = "Brent method";
            break;
        case AlgorithmCellType::VARIABLE:
            type = "Variable cell method";
            break;
    }
    return type;
}

double Config::tol_cell() const{
    return get_double(ALGO_CELL_SECTION, "tol_cell");
}

arma::uword Config::max_iter_cell() const{
    return get_integer(ALGO_CELL_SECTION, "max_iter_cell");
}

vec Config::batch_cell_min() const{
    return get_list_double(ALGO_CELL_SECTION, "batch_cell_min");
}

bool Config::check_batch_cell_min() const{
    bool flag = true;
    vec v = batch_cell_min();

    if(algo_cell_optimization_type() == AlgorithmCellType::SINGLE)
        return true;

    if(v.n_elem < dim()){
        cerr<<"there must be "<<dim();
        cerr<<" number of elements in batch_cell_min for ";
        cerr<<dim()<<"D simulations."<<endl;
        flag = false;
    }
    else if(v(0)<=0){
        cerr<<"The first element must > 0 in batch_cell_min"<<endl;
        flag = false;
    }
    else if(v(1)<=0 && dim()>1){
        cerr<<"The second element must > 0 in batch_cell_min for 2D & 3D.";
        cerr<<endl;
        flag = false;
    }
    else if(v(2)<=0 && dim()>2){
        cerr<<"The third element must > 0 in batch_cell_min for 3D."<<endl;
        flag = false;
    }
    return flag;
}

vec Config::batch_cell_max() const{
    return get_list_double(ALGO_CELL_SECTION, "batch_cell_max");
}

bool Config::check_batch_cell_max() const{
    bool flag = true;
    vec v = batch_cell_max();
    vec vmin = batch_cell_min();

    if(algo_cell_optimization_type() == AlgorithmCellType::SINGLE
        || algo_cell_optimization_type() == AlgorithmCellType::BRENT)
        return true;

    if(v.n_elem < dim()){
        cerr<<"there must be "<<dim();
        cerr<<" number of elements in batch_cell_max for ";
        cerr<<dim()<<"D simulations."<<endl;
        flag = false;
    }
    else if(v(0)<=vmin(0)){
        cerr<<"The first element in batch_cell_max must > ";
        cerr<<"the first element in batch_cell_min."<<endl;
        flag = false;
    }
    else if(v(1)<=vmin(1) && dim()>1){
        cerr<<"The second element in batch_cell_max must > ";
        cerr<<"the second element in batch_cell_min for 2D & 3D."<<endl;
        flag = false;
    }
    else if(v(2)<=vmin(2) && dim()>2){
        cerr<<"The third element in batch_cell_max must > ";
        cerr<<"the third element in batch_cell_min for 3D."<<endl;
        flag = false;
    }
    return flag;
}

vec Config::batch_cell_step() const{
    return get_list_double(ALGO_CELL_SECTION, "batch_cell_step");
}

bool Config::check_batch_cell_step() const{
    bool flag = true;
    vec v = batch_cell_step();

    if(algo_cell_optimization_type() == AlgorithmCellType::SINGLE)
        return true;

    if(v.n_elem < dim()){
        cerr<<"there must be "<<dim();
        cerr<<" number of elements in batch_cell_step for ";
        cerr<<dim()<<"D simulations."<<endl;
        flag = false;
    }
    else if(v(0)<=0){
        cerr<<"The first element must > 0 in batch_cell_step"<<endl;
        flag = false;
    }
    else if(v(1)<=0 && dim()>1){
        cerr<<"The second element must > 0 in batch_cell_step for 2D & 3D.";
        cerr<<endl;
        flag = false;
    }
    else if(v(2)<=0 && dim()>2){
        cerr<<"The third element must > 0 in batch_cell_step for 3D."<<endl;
        flag = false;
    }
    return flag;
}

/***********************************************************************
 *                                                                     *
 *                                                                     *
 *        Member functions for Algorithm_Contour_Integration Section   *
 *                                                                     *
 *                                                                     *
 ***********************************************************************/

/*
 * Algorithms for performing contour integration.
 * Currently only one method is supported, i.e. SIMPSON.
 * For case where Simpson's rule cannot be used, use other 4th order
 * quadrature methods in Quadrature4.h/.cc instead.
 */
AlgorithmContourType Config::algo_contour_integration_type() const{
    string t = get_string(ALGO_CONTOUR_SECTION, "algorithm");
    transform(t.begin(), t.end(), t.begin(), ::toupper);

    if(t == "SIMPSON")
        return AlgorithmContourType::SIMPSON;
    else if(t == "TRAPEZOIDAL" || t == "TRAP" || t == "TRAPE" || t == "TRAPZ")
        return AlgorithmContourType::TRAPEZOIDAL;
    else
        return AlgorithmContourType::SIMPSON;
}

string Config::get_algo_contour_integration_type_string() const{
    string type;
    switch(algo_contour_integration_type()){
        case AlgorithmContourType::SIMPSON:
            type = "Simpson";
            break;
        case AlgorithmContourType::TRAPEZOIDAL:
            type = "Trapezoidal";
            break;
    }
    return type;
}

/***********************************************************************
 *                                                                     *
 *                                                                     *
 *        Member functions for raw I/O                                 *
 *                                                                     *
 *                                                                     *
 ***********************************************************************/

bool Config::get_bool(const string section, const string key) const{
    return _ini.GetBoolValue(section.c_str(), key.c_str());
}

int Config::get_integer(const string section, const string key) const{
    return _ini.GetLongValue(section.c_str(), key.c_str());
}

double Config::get_double(const string section, const string key) const{
    return _ini.GetDoubleValue(section.c_str(), key.c_str());
}

string Config::get_string(string const section, const string key) const{
    return _ini.GetValue(section.c_str(), key.c_str());
}

/**
 * Parse Python list of integers, such as
 *      [1, 2, 3, 4, 5, 6, 7, 8]
 * The resulted list is stored in an Armadillo ivec.
 */
uvec Config::get_list_integer(const string section, const string key) const{
    string instr = get_string(section, key);
    istringstream iss(instr);
    uword n = 0;  // number of elements
    uword ival;  // value of each element
    uvec v;

    char c;
    // Seek to the first '['
    while((iss >> c) && (c != '['));
    // Test whether iss is empty
    if(!iss)
        return v;

    // Pre-read to determine the number of elements
    while(iss >> ival){
        n += 1;
        iss >> c;
        if(c != ',')
            break;
    }

    v.set_size(n);
    uword i = 0;
    iss.seekg(0, iss.beg);  // Move to the begin of the string.
    while((iss >> c) && (c != '['));
    while(iss >> ival){
        v(i) = ival;
        i += 1;
        iss >> c;
        if(c != ',')
            break;
    }
    return v;
}

/**
 * Parse Python list of doubles, such as
 *      [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8]
 * The resulted list is stored in an Armadillo vec.
 */
vec Config::get_list_double(const string section, const string key) const{
    string instr = get_string(section, key);
    istringstream iss(instr);
    uword n = 0;  // number of elements
    double val;  // value of each element
    vec v;

    char c;
    // Seek to the first '['
    while((iss >> c) && (c != '['));
    // Test whether iss is empty
    if(!iss)
        return v;

    // Pre-read to determine the number of elements
    while(iss >> val){
        n += 1;
        iss >> c;
        if(c != ',')
            break;
    }

    v.set_size(n);
    uword i = 0;
    iss.seekg(0, iss.beg);  // Move to the begin of the string.
    while((iss >> c) && (c != '['));
    while(iss >> val){
        v(i) = val;
        i += 1;
        iss >> c;
        if(c != ',')
            break;
    }
    return v;
}

bool Config::set_list_integer(const string section, const string key,
                              const ivec list_int){
    /* to be implemented */
}

bool Config::set_list_double(const string section, const string key,
                             const vec list_double){
    /* to be implemented */
}

bool Config::set_bool(const string section, const string key, const bool val){
    SI_Error rc = _ini.SetBoolValue(section.c_str(), key.c_str(), val);
    return !(rc < 0);
}

bool Config::set_integer(const string section, const string key,
                         const int val){
    SI_Error rc = _ini.SetLongValue(section.c_str(), key.c_str(), val);
    return !(rc < 0);
}

bool Config::set_double(const string section, const string key,
                        const double val){
    SI_Error rc = _ini.SetDoubleValue(section.c_str(), key.c_str(), val);
    return !(rc < 0);
}

bool Config::set_string(string const section, const string key,
                        const string val){
    SI_Error rc = _ini.SetValue(section.c_str(), key.c_str(), val.c_str());
    return !(rc < 0);
}


