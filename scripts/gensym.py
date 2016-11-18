#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gensym
======

A script for generating symmetry patterns using gyroid package.

Copyright (C) 2012 Yi-Xin Liu

"""

import os.path
import argparse
from ConfigParser import SafeConfigParser

import numpy as np
from gyroid import prepare_scft_input

parser = argparse.ArgumentParser(description='gensym options')
parser.add_argument('-c', '--config',
                    default='param.ini',
                    help='the configuration file of polyorder.')
parser.add_argument('-i', '--image',
                    action='store_true',
                    help='show image if present.')
parser.add_argument('-s', '--save',
                    action='store_true',
                    help='save image if present.')
parser.add_argument('-d', '--dry',
                    action='store_true',
                    help='a dry run without doing actual work.')

args = parser.parse_args()

if __name__ == '__main__':
    cfgfile = args.config
    cfg = SafeConfigParser(allow_no_value=True)
    cfg.read(cfgfile)

    dim = cfg.getint('Grid', 'dimension')
    Lx = cfg.getint('Grid', 'Lx')
    Ly = cfg.getint('Grid', 'Ly')
    Lz = cfg.getint('Grid', 'Lz')
    grid_num_vec = np.array([Lx, Ly, Lz])

    data_file = cfg.get('Grid', 'field_data')
    if data_file is None:
        data_file = 'field_in.mat'

    cryst_system = cfg.get('UnitCell', 'CrystalSystemType')
    sym_group = cfg.get('UnitCell', 'SymmetryGroup')

    if not cfg.get('UnitCell', 'N_list'):
        # N1 = N2 = N3 = 0, use grid_num_vec
        if dim == 1:
            basis_grid_vec = grid_num_vec[0:1]
        if dim == 2:
            basis_grid_vec = grid_num_vec[0:2]
        if dim == 3:
            basis_grid_vec = grid_num_vec[0:3]
    else:
        N = eval(cfg.get('UnitCell', 'N_list'))  # N is a tuple (N1, N2, N3)
        basis_grid_vec = np.array(N)
        if basis_grid_vec.size != dim:
            raise ValueError("The size of N_list and dim not match")

    # c is a tuple of basis coefficients
    basis_c = eval(cfg.get('UnitCell', 'c_list'))
    basis_c = 1.0 * np.array(basis_c)

    a = cfg.getfloat('UnitCell', 'a')
    if not cfg.get('UnitCell', 'b'):
        b = 0.0
    else:
        b = cfg.getfloat('UnitCell', 'b')
    if not cfg.get('UnitCell', 'c'):
        c = 0.0
    else:
        c = cfg.getfloat('UnitCell', 'c')
    if not cfg.get('UnitCell', 'alpha'):
        alpha = 0.0
    else:
        alpha = cfg.getfloat('UnitCell', 'alpha')
    if not cfg.get('UnitCell', 'beta'):
        beta = 0.0
    else:
        beta = cfg.getfloat('UnitCell', 'beta')
    if not cfg.get('UnitCell', 'gamma'):
        gamma = 0.0
    else:
        gamma = cfg.getfloat('UnitCell', 'gamma')
    cryst_param_vec = np.array([a, b, c, alpha, beta, gamma])

    show_img = args.image
    save_img = args.save
    # 'opt/lyx/polyorder/field_in.mat' => 'opt/lyx/polyorder/field_in.png'
    img_file = data_file.replace('.mat', '.png')

    if args.dry:
        print dim
        print grid_num_vec
        print cryst_system
        print cryst_param_vec
        print sym_group
        print basis_grid_vec
        print basis_c
        print data_file
        print show_img
        print save_img
        print img_file
    else:
        prepare_scft_input(dim, grid_num_vec, cryst_system,
                           cryst_param_vec, sym_group,
                           basis_grid_vec, basis_c,
                           data_file, show_img, save_img, img_file)
