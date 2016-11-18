/**
 * Helper.h
 * Created at 2011.6.14
 *
 * Helper provides common helper methods for Polyorder.
 * It mainly includes the helper functions for initializing
 * Lamellar phase pattern.
 *
 * HISTORY:
 * 2012.4.17
 *   1. From now on, the history of this file is tracked by Mercurial.
 *   2. Package name: polyorder.
 * 2011.6.14
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

#ifndef polyorder_helper_h
#define polyorder_helper_h

#include <cmath>
#include <iostream>
#include <string>

#include "common.h"
#include "Config.h"
#include "Grid.h"

class Helper{
public:
    static void init_pattern(Grid &g, const PhasePattern pt,
                             const double c, const double v1,
                             const double v2=_empty);

private:
    static constexpr double _empty=0.0;
    // Only two-component systems are supported currently
    // The fully separation is assumed
    // c should be the concentration of minor phase
    static void init_LAM1(Grid &g,const double c,const double v1,const double v2);
    static void init_LAM2(Grid &g,const double c,const double v1,const double v2);
    static void init_LAM3(Grid &g,const double c,const double v1,const double v2);
    static void init_LAM4(Grid &g,const double c,const double v1,const double v2);
    static void init_LAM5(Grid &g,const double c,const double v1,const double v2);
    static void init_LAM6(Grid &g,const double c,const double v1,const double v2);
};

#endif

