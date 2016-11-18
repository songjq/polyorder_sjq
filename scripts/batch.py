#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
batch.py
========

A script for automatic data processing.

Copyright (C) 2012 Yi-Xin Liu

"""

import os
import sys
import glob
import re
import time
import datetime
import argparse
from ConfigParser import SafeConfigParser

import shutil
import numpy as np
import scipy.io

from common import vardict, time_format, now2str 
from common import get_script_batch_key, get_nodes, get_var_with_val
from render import render_1d, render_2d, render_3d

parser = argparse.ArgumentParser(description='batch options')

parser.add_argument('-c', '--config',
                    default='param.ini',
                    help='the configuration file of polyorder.')

parser.add_argument('-d', '--dir',
                    help='the directory to be processed, ' + \
                         '-c or --config is ignored.')

parser.add_argument('-r', '--result', help='The result file.')

args = parser.parse_args()

__version__ = 0.1


def process_file(file):
    result = []

    file_name = os.path.basename(file)
    # ['scft','out','a5.5','5678.mat']
    fragments = file_name.split('_')
    # '5.5'
    str_size = re.search('\d+(\.\d*)?',fragments[-2]).group()
    size = eval(str_size)
    # '5678'
    str_iters = fragments[-1].split('.')[0]
    iters = eval(str_iters)
    
    data = scipy.io.loadmat(file)
    param_file = 'param_out' + \
                 re.search('_[a-zA-Z]+\d+(\.\d*)?',file_name).group() + \
                 '.mat'
    param_file = os.path.join(os.path.dirname(file),param_file)
    param = scipy.io.loadmat(param_file)
    dim = param['dim']
    record_interval = param['record_interval'][0,0]
    index = iters / record_interval
    H = data['H'][index-1,0]
    errRes = data['errRes'][index-1,0]
    incomp = data['incomp'][index-1,0]

    result.append(size)
    result.append(iters)
    result.append(H)
    result.append(errRes)
    result.append(incomp)

    return result, dim


def process_dir(dir):
    '''
    the naming convention of data files: 
           scft_out_{var}{val}_{iters}.mat
    e.g.
           scft_out_a5.5_5678.mat
    '''

    data_files = glob.glob(dir + os.path.sep + 'scft_out*.mat')
    minH_file = '' # the scft_out file with minimum H
    minH = 1e99
    minH_result = []
    minH_dim = 1
    for file in data_files:
        result, dim = process_file(file)
        if(result[2] < minH):
            minH = result[2]
            minH_file = file
            minH_result = result
            minH_dim = dim

    if(not minH_result):
        return []

    img_file = os.path.join(dir, 'phiA_'+str(minH_result[0])+'.png')
    if(dim == 1):
        render_1d('phiA', minH_file, img_file, 2)
    elif(dim == 2):
        render_2d('phiA', minH_file, img_file, 2)
    else:
        render_3d('phiA', minH_file, img_file, 2)

    return minH_result


def output(f,bs_val,result):
    print >>f, '%g\t' % bs_val,
    for val in result:
        print >>f, '%g\t' % val,
    print >>f


if __name__ == '__main__':
    if(not args.dir is None):
        print 'size\titers\tH\t\terrRes\t\tincomp'
        for val in process_dir(args.dir):
            print '%g\t' % val,
        print
        exit(0)

    cfgfile = args.config
    cfg = SafeConfigParser(allow_no_value=True)
    cfg.optionxform = str
    cfg.read(cfgfile)

    # --- data dir ---
    # data dir pattern:
    #    $(data_path)/$(var_with_val}-$(data_path_suffix)/
    #                                       ${batchvar_with_val}/
    data_path =  cfg.get('xscft','dataPath')
    data_path_suffix =  cfg.get('xscft','dataPathSuffix')
    var_with_val = get_var_with_val(cfg)
    dir1 = os.path.join(data_path,var_with_val+data_path_suffix)

    bname = cfg.get('xscft','activeBatchName')
    if(args.result is None):
        result_file = os.path.join(dir1, 'result-'+bname+'.txt')
    else:
        result_file = os.path.join(dir1, args.result)
    f = open(result_file,'w')
    # process each dir
    bs_min = cfg.getfloat('xscft','batchScriptMin')
    bs_max = cfg.getfloat('xscft','batchScriptMax')
    bs_step = cfg.getfloat('xscft','batchScriptStep')
    bs_key = get_script_batch_key(cfg)
    print >>f, bs_key+'\tsize\titers\tH\t\terrRes\t\tincomp'
    for bs_val in np.arange(bs_min,bs_max+bs_step,bs_step):
        batchvar_with_val = vardict[bs_key]
        batchvar_with_val += str(bs_val)
        bs_dir = os.path.join(dir1,batchvar_with_val)
        if not os.path.exists(bs_dir):
            exit(0)
        print bs_dir
        result = process_dir(bs_dir)
        if(result):
            output(f,bs_val,result)
