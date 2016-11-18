/**
 * Simpson.h
 * Created at 2011.6.20
 *
 * Simpson is derived from Updater implementing
 * the Composite Simpson's rule for integrating Propagators
 * to obtain Density.
 *
 * HISTORY:
 * 2012.4.1
 *   1. From now on, the history of this file is tracked by Mercurial.
 *   2. Package name: polyorder.
 * 2011.6.20
 *   1. original version
 *   2. All functionalities tested. Passed.
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

#ifndef polyorder_simpson_h
#define polyorder_simpson_h

#include <blitz/array.h>

#include "Updater.h"
#include "Propagator.h"

class Simpson:public Updater{
public:
	void solve(blitz::Array<double,3> data,
               const Propagator &q,
               const Propagator &qc,
               const double cc) const;

    Simpson *clone() const;
};

#endif

