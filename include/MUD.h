/**
 * MUD.h
 * Created at 2012.4.12
 *
 * MUD is a base calss for multigrid updaters.
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

#ifndef polyorder_mud_h
#define polyorder_mud_h

#include "Updater.h"
#include "Multigrid.h"
#include "Grid.h"

using std::string;

class MUD:public Updater{
public:
    MUD(){}
    MUD(const MUD &rhs):Updater(rhs),_type(rhs._type),_is_guess(rhs._is_guess),_fft2mg_mode(rhs._fft2mg_mode),_fft2mg_interp(rhs._fft2mg_interp){
        for(int i=0;i<23;i++)
          _iparm[i] = rhs._iparm[i];
        for(int i=0;i<8;i++)
          _fparm[i] = rhs._fparm[i];
        for(int i=0;i<4;i++)
          _mgopt[i] = rhs._mgopt[i];
    }

    void set_type(string t){
        _type = MUD_SP;
        for(int type=MUD_SP;type<NUM_MUDPACK_TYPE;type++)
            if(t.c_str()==MudpackTypes[type]) 
                _type = static_cast<MudpackType>(type);
    }
    void set_type(MudpackType t){_type=t;}
    string type() const{return MudpackTypes[_type];}
    int error() const{return _ierror;}
    virtual void set_guess(bool is_guess=false){}
    virtual void set_maxcy(int maxcy=1){}
    virtual void set_tolmax(MUD_FLOAT tolmax=0.0){}
    virtual void set_method(int imethod=0){}
    void set_mgopt(int kcycle=1,int iprer=2,int ipost=1,int intpol=1){
        _mgopt[0] = kcycle; //kcycle: 1 for W-cycle
        _mgopt[1] = iprer; //iprer: number of pre-relaxation
        _mgopt[2] = ipost; //ipost: number of post-relaxation
        _mgopt[3] = intpol; //intpol: 1 for bilinear interpolation
    }
    void set_fft2mg_interp(int interp_kind=0){_fft2mg_interp=interp_kind;}

    void solve(Grid &, const Grid &) const=0;
    void solve(Grid &, const Grid &)=0;

    MUD *clone() const=0;

    ~MUD(){}

protected:
    MudpackType _type; 
    // for 2D case, only 0 to 16 is used; 
    // For 3D mud3sp, only 0 to 21 is used.
    MUD_INT _iparm[23];
    MUD_FLOAT _fparm[8]; // for 2D case, only 0 to 5 is used.
    MUD_INT _mgopt[4];
    blitz::Array<MUD_FLOAT,1> _work;
    MUD_INT _ierror;
    MUD_INT _is_guess;
    // mode 0 for Lx <=> nx = Lx+1; mode 1 for Lx <=> nx = 2*Lx + 1
    int _fft2mg_mode;
    int _fft2mg_interp;

    void _decompose_L(const int L,int &p,int &i);
};

/**
 * Decompose any integer number L into the form
 *      L = p * 2^(i-2)
 * p and i are returned.
 *
 */
inline void MUD:: _decompose_L(const int L,int &p,int &i){
        int n = L;
        i = 2;
        while(n % 2 == 0 and n > 2){
            n /= 2;
            i += 1;
        }
        n = 1;
        n <<= (i-2); // n = 2^(i-2)
        p = L / n;
}

#endif

