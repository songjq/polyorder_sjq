/**
 * Config.h
 * Created at 2011.6.8
 *
 * Config is a light-weight wrapper of SimpleIni to handle
 * the configuration file in qSCFT project.
 *
 * HISTORY:
 * 2014.6.11
 *   1. Split Config.h to Config.h & Config.cc.
 *   2. Now can parse Python list
 * 2012.4.2
 *   1. From now on, the history of this file is tracked by Mercurial.
 *   2. Package name: polyorder.
 * 2011.6.11
 *   1. Add methods: get_bool,set_bool,set_integer,set_double,set_string
 * 2011.6.8
 *   1. original version
 *
 * Copyright (C) 2012-2014 Yi-Xin Liu <lyx@fudan.edu.cn>
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

#ifndef polyorder_config_h
#define polyorder_config_h

#include "common.h"

#include "SimpleIni.h"
#include "armadillo"

#include <string>

using std::string;

class Config{
public:
    Config(){}
    Config(const char *config_file);
    Config(const string config_data);

    bool get_config_data(string& config_data) const;
    bool set_config_data(const string& config_data);
    bool save(const string config_file);
    bool reload_from_file(const string config_file);
    void clear();
    bool is_empty() const;

    // check the consistency of the configurations
    bool check() const;
    bool check_n_block() const;
    bool check_segment_length() const;
    bool check_f() const;
    bool check_chiN() const;
    bool check_BC_left() const;
    bool check_BC_right() const;
    bool check_BC_left_brush() const;
    bool check_BC_right_brush() const;
    bool check_dim() const;
    bool check_Lx() const;
    bool check_Ly() const;
    bool check_Lz() const;
    bool check_grid_type() const;
    bool check_grid_init_type() const;
    bool check_field_data() const;
    bool check_unitcell_a() const;
    bool check_unitcell_b() const;
    bool check_unitcell_c() const;
    bool check_crystal_system_type() const;
    bool check_lam() const;
    bool check_ds() const;
    bool check_Ms() const;
    bool check_anderson_mixing() const;
    bool check_batch_cell_min() const;
    bool check_batch_cell_max() const;
    bool check_batch_cell_step() const;

    /** member functions for Version section **/
    double version() const;

    /** member functions for IO section **/
    string base_dir() const;
    string data_file() const;
    string param_file() const;
    string q_file() const;
    bool is_display() const;
    bool is_save_data() const;
    bool is_save_q() const;
    arma::uword display_interval() const;
    arma::uword record_interval() const;
    arma::uword save_interval() const;

    /** member functions for Model section **/
    ModelType model() const;
    MaskType mask() const;
    EnsembleType ensemble() const;
    string get_model_type_string() const;
    string get_mask_type_string() const;
    string get_ensemble_type_string() const;
    // if true, using Helfand compressible model.
    arma::uword n_chain() const;  // number of chains
    /* A list contains number of blocks for each species.
     * For homopolymer and small molecules, n_block = 1.
     * We refer blocks, homopolymers, and small molecules to components.
     */
    arma::uvec n_block() const;
    /* number of components.
     *      n_component = sum_{i=1}{n_specie} n_block(i)
     */
    arma::uword n_component() const;
    /* The physical segment length of each component.
     * Taking AB + C + M as an example,
     * where AB is an A-B diblock copolymer),
     * C is a homopolymer,
     * M is a small molecules.
     * Then, the returning list is
     *      [bA, bB, bC, bM]
     */
    arma::vec segment_length() const;
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
    arma::vec f() const;
    arma::vec wall_affinity() const;
    /* The chi*N of each pair of components.
     * For n components, there are at most
     *      n*(n-1)/2
     * chiNs, where n = sum_{i=1}{n_species} n_block(i).
     * These chiNs are listed in the following way illustrated by an example.
     *       AB + C + M
     *   [chiN_AB, chiN_AC, chiN_AM, chiN_BC, chiN_BM, chiN_CM]
     */
    arma::vec chiN() const;
    double depletion_length() const;
    double length_ratio() const;
    bool is_compressible() const;
    double graft_density() const;
    double graft_area() const;   //added by songjq which is important to the stability of program
    double excluded_volume() const;

    /** member functions for UnitCell section **/
    CrystalSystemType get_crystal_system_type() const;
    static string get_crystal_system_type_string(CrystalSystemType cstype);
    string get_crystal_system_type_string() const;
    double a() const;
    double b() const;
    double c() const;
    double a(double val);
    double b(double val);
    double c(double val);
    // alpha, beta, gamma in radian unit
    double alpha() const;
    double beta() const;
    double gamma() const;

    /** member functions for Grid section **/
    arma::uword dim() const;
    arma::uword Lx() const;
    arma::uword Ly() const;
    arma::uword Lz() const;
    ConfineType ctype() const;
    string get_confine_type_string() const;
    GridType gtypex() const;
    GridType gtypey() const;
    GridType gtypez() const;
    static string get_grid_type_string(GridType gtype);
    string get_gtypex_string() const;
    string get_gtypey_string() const;
    string get_gtypez_string() const;
    void set_gtypex(const GridType gtype);
    void set_gtypey(const GridType gtype);
    void set_gtypez(const GridType gtype);
    /*
     * A three-element vector describes the boundary condition:
     *      alpha * du/dx + beta * u = gamma
     * the vector is [alpha, beta, gamma]
     * See the definition of DBC, NBC, RBC, and PBC in cheb++ package.
     */
    arma::vec BC_coefficients_left() const;
    arma::vec BC_coefficients_right() const;
    arma::vec BC_coefficients_left_brush() const;
    arma::vec BC_coefficients_right_brush() const;
    GridInitType get_grid_init_type() const;
    static string get_grid_init_type_string(GridInitType gitype);
    string get_grid_init_type_string() const;
    void set_grid_init_type(const GridInitType git);
    arma::uword seed() const;
    string field_data_file() const;
    void set_field_data_file(const string file);
    void set_phase_pattern(const PhasePattern pt);
    PhasePattern get_phase_pattern() const;

    /** Member functions for Algorithm_MDE Section **/
    AlgorithmMDEType algo_mde_type() const;
    string get_algo_mde_type_string() const;
    /* Size of contour step for each component.
     *      AB + C + M
     *   [ds_A, ds_B, ds_C, 1.0]
     * For small molecules, ds = 1.0
     */
    arma::vec ds() const;
    /* Number of contour steps for each component.
     *      AB + C + M
     *   [Ms_A, Ms_B, Ms_C, 1]
     * For small molecules, Ms = 1.
     *      ds_A = f_A / Ms_A
     *      ds_B = f_B / Ms_B
     *      ds_C = f_C / Ms_C
     * Use either ds() or Ms().
     */
    arma::uvec Ms() const;
    ETDRK4SCHEME etdrk4_scheme_type() const;
    string get_etdrk4_scheme_type_string() const;
    arma::uword etdrk4_M() const;

    /** Member functions for Algorithm_SCFT Section **/
    AlgorithmSCFTType algo_scft_type() const;
    string get_algo_scft_type_string() const;
    /* Relaxation parameters for each component.
     *      AB + C + M
     *   [lamA, lamB, lamC, lamM]
     */
    arma::vec lam() const;
    arma::uword min_iter() const;
    arma::uword max_iter() const;
    double thresh_H() const;
    double thresh_residual() const;
    double thresh_incomp() const;
    arma::uword n_Anderson_mixing() const;

    /** Member functions for Algorithm_Cell_Optimization Section **/
    AlgorithmCellType algo_cell_optimization_type() const;
    string get_algo_cell_optimization_type_string() const;
    double tol_cell() const;
    arma::uword max_iter_cell() const;
    arma::vec batch_cell_min() const;
    arma::vec batch_cell_max() const;
    arma::vec batch_cell_step() const;

    /** Member functions for Algorithm_Contour_Integration Section **/
    AlgorithmContourType algo_contour_integration_type() const;
    string get_algo_contour_integration_type_string() const;

    /** member function for raw IO **/
    bool get_bool(const string section, const string key) const;
    int get_integer(const string section, const string key) const;
    double get_double(const string section, const string key) const;
    string get_string(string const section, const string key) const;
    arma::uvec get_list_integer(const string section, const string key) const;
    arma::vec get_list_double(const string section, const string key) const;
    bool set_bool(const string section, const string key, const bool val);
    bool set_integer(const string section, const string key, const int val);
    bool set_double(const string section, const string key, const double val);
    bool set_string(string const section, const string key, const string val);
    bool set_list_integer(const string section, const string key,
                          const arma::ivec list_int);
    bool set_list_double(const string section, const string key,
                         const arma::vec list_double);

private:
    CSimpleIniCaseA _ini;

    GridType get_grid_type(const string key) const;
    void set_grid_type(const string key, const GridType gtype);

    static const string VERSION_SECTION;
    static const string IO_SECTION;
    static const string MODEL_SECTION;
    static const string UNITCELL_SECTION;
    static const string GRID_SECTION;
    static const string ALGO_MDE_SECTION;
    static const string ALGO_SCFT_SECTION;
    static const string ALGO_CELL_SECTION;
    static const string ALGO_CONTOUR_SECTION;
    static const string ALGO_CHARGE_SECTION;
};

#endif
