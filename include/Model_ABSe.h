/**
 * Model_ABSe.h
 * Created at 2011.6.25
 *
 * Model_ABSe is a class derived from Model describing a 
 * charged linear A-B diblock copolymer solution
 *
 * The fields are updated (electrostatic potential field is updated 
 * by solving Poisson-Boltzmann equation), 
 * the propagators are calculated, and the densities are regenerated.
 *
 * HISTORY:
 * 2012.4.11
 *   1. From now on, the history of this file is tracked by Mercurial.
 *   2. Package name: polyorder.
 * 2011.6.27
 *   1. phiC_avg=1.0 with psi=0 test passed.
 *   2. phiC_avg=0.8 with psi=0 test passed.
 *   3. phiC_avg=0.8 with smeared psi and const eps test passed.
 * 2011.6.25
 *   1. original version
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

#ifndef polyorder_model_abse_h
#define polyorder_model_abse_h

#include <string>
#include <iostream>

#include "Model.h"
#include "Config.h"
#include "Grid.h"
#include "Field.h"
#include "FieldE.h"
#include "Yita.h"
#include "Density.h"
#include "DensitySmall.h"
#include "Propagator.h"
#include "CMatFile.h"
#include "Helper.h"
#include "Simpson.h"

using std::string;

class Model_ABSe:public Model{
public:
    Model_ABSe(){}
	Model_ABSe(const Config &cfg){init(cfg);}

	void init(const Config &cfg);
    void reset(const Config &cfg){release_memory();init(cfg);}

	void update();
    double Hw() const;
    double Hs() const;
    double H() const{return Hw()+Hs();}
    double incomp() const;
    double residual_error() const;
    double density_error() const;

    void display() const;
    void save(const string file){_wA->uc().save(file,_wA->Lx(),_wA->Ly(),_wA->Lz());save_field(file);save_density(file);}
    void save_model(const string file);
    void save_field(const string file){_wA->save(file);_wB->save(file);_wS->save(file);_yita->save(file);_psi->save(file);}
    void save_density(const string file){_phiA->save(file);_phiB->save(file);_phiS->save(file);_phiP->save(file);_phiN->save(file);}
    void save_q(const string file){_qA->save(file);_qAc->save(file);_qB->save(file);_qBc->save(file);}

    ~Model_ABSe(){release_memory();}

private:
    int _N,_NA,_NB; // number of statistical segments, N=NA+NB 
    double _fA,_fB;   // fA=1-fB=NA/N
    double _a,_Rg;    // Rg=a*sqrt(N/6)
    // Flory-Huggins interation parameters. ChiN=N*chiAB
    double _chiN,_chiAB; 
    // Flory-Huggins interation parameters between solvent and A, B segment
    double _chiASN,_chiBSN; 
    // volume-averaged densities, phiC_avg=phiA_avg+phiB_avg
    double _phiC_avg,_phiA_avg,_phiB_avg; 
    int _charge_distribution; // semared (0) or annealed (1) case
    // non-position-dependent (0) or non-postition-dependent (1)
    // dielectric constant type
    int _dielectric_constant; 
    // Tapzoidal (0) or Simpson integration scheme for density
    int _density_integration;
    double _cs; // salt concentration
    double _alphaA,_alphaB; // degree of ionization
    int _upsA,_upsB,_upsP,_upsN; // upsilonX, valency
    double _epsA,_epsB,_epsS; // epsilonX, dielectric constant
    double _phiS_avg,_phiP_avg,_phiN_avg; 

    int _Ms;    // for discreting path integral of q
    double _ds;
    int _sA,_sB;
    
    Field *_wA,*_wB,*_wS,*_wP,*_wN;
    FieldE *_psi;
    Yita *_yita;
    Density *_phiA,*_phiB;
    DensitySmall *_phiS,*_phiP,*_phiN;
    Propagator *_qA,*_qB,*_qAc,*_qBc;

    void init_field(const Config &cfg);
    void init_density(const Config &cfg);
    void init_propagator(const Config &cfg);
    void release_memory();
};

#endif

