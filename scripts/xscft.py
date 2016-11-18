#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
xscft.py
========

A script for automatic SCFT program submission.

Before using this script, ensure that RPYC server rpyc_classic.py
is runing on every node. Start the RPYC server use:
    $ /path/to/run_rpyc.py 
run_rpyc.py is a script written by Yi-Xin Liu

Copyright (C) 2012 Yi-Xin Liu

"""

import os
import sys
import time
import datetime
import pickle
from subprocess import PIPE
import argparse
from ConfigParser import SafeConfigParser

import shutil
import psutil
import rpyc
import numpy as np

parser = argparse.ArgumentParser(description='xscft options')

parser.add_argument('-c','--config',
                    default='param.ini',
                    help='the configuration file of polyorder.')

parser.add_argument('-d','--dry',
                    action='store_true',
                    help= 'If present or True, \
                           create directories and generate \
                           param files only, do not copy and submit exe.')

parser.add_argument('-s','--stop',
                    action='store_true',
                    help='If present or True, ' + \
                         'stop all processes documented ' + \
                         'in pid-node-min-max.')

parser.add_argument('-e','--clean',
                    action='store_true',
                    help='If present or True, ' + \
                         'remove active batch file and data directories.')

args = parser.parse_args()

__version__ = 0.1

vardict = {'p' : 'alphaA',
           'e' : 'epsS',
           'eA' : 'epsA',
           'f' : 'fA',
           'k' : 'chiN',
           'kAS' : 'chiAS',
           'kBS' : 'chiBS',
           's' : 'cs',
           'fC' : 'phiC',
          }

default_var = 'f'

nodelist = ['c0109','c0114']

time_format = '%Y%m%d-%H%M%S'

def now2str(format):
    return datetime.datetime.now().strftime(format)


def get_script_batch_key(cfg):
    in_key = cfg.get('xscft','batchScriptVar')
    if vardict.has_key(in_key):
        return in_key
    return default_var


def get_nodes(cfg):
    ''' a list of nodes reading from file '''
    nodes_file = cfg.get('xscft','nodeFile')
    if os.path.exists(nodes_file):
        return [line.strip() for line in open(nodes_file)]
    return nodelist


def get_free_cores(node):
    ''' Number of free cores in node '''
    #cmd = "python -c 'import psutil;print psutil.cpu_percent(interval=1.0,percpu=True)'"
    #p = psutil.Popen(['rsh',node,cmd],stdout=PIPE)
    #out, dummy = p.communicate()
    #time.sleep(1.5) # To avoid "protocal failure in circuit setup"
    try:
        conn = rpyc.classic.connect(node)
    except Exception as e:
        return 0

    core_usage = conn.modules.psutil.cpu_percent(interval=1.0,percpu=True)
    core_free = 100.0 - np.array(core_usage)
    return np.sum(core_free > 50.0)


def get_node_with_free_core(nodes):
    ''' First node with more than 1 free cores '''
    for node in nodes:
        if get_free_cores(node) > 0:
            return node
    return ''


def get_free_node(cfg,wait):
    ''' Wait until at least one node has more than 1 free cores. '''
    nodes = get_nodes(cfg)
    free_node = get_node_with_free_core(nodes)

    time_step = 0.2
    while not free_node:
        print '\nNo free node available. Waiting for \033[34;49m' + \
                str(wait) + '\033[m seconds to try again.'
        sys.stdout.write('Waiting.... \033[35;49m')
        sys.stdout.flush()
        type = 0
        for t in np.arange(0,wait,time_step):
            if type == 0: sys.stdout.write('\b/')
            if type == 1: sys.stdout.write('\b-')
            if type == 2: sys.stdout.write('\b\\')
            if type == 3: sys.stdout.write('\b|')
            type += 1
            if type == 4: type = 0
            sys.stdout.flush()
            time.sleep(time_step);
        print '\033[m\nTry again.... '
        nodes = get_nodes(cfg)
        free_node = get_node_with_free_core(nodes)
    return free_node


def get_var_with_val(cfg):
    s = ''
    temp_dict = vardict.copy()
    del temp_dict[get_script_batch_key(cfg)]
    for var in sorted(temp_dict.keys()):
        s += var
        s += cfg.get('Model',vardict[var])
    return s


def make_dir(dir):
    ''' Create a leaf directory and all intermediate ones if not exists. '''
    if not os.path.exists(dir):
        os.makedirs(dir)


def actual_run(dir,exe,node):
    conn = rpyc.classic.connect(node)
    conn.modules.os.chdir(dir)
    fh = conn.builtin.open('log','a+')
    p = conn.modules.psutil.Popen(['./'+exe],stdout=fh)
    print p.exe + ' was submitted to node \033[35;49m' + \
            node + '\033[m with PID \033[32;49m' + str(p.pid) + '\033[m'
    return p.pid


def stop_processes(pid_node_file):    
    with open(pid_node_file,'r') as pfile:
        pid_node = pickle.load(pfile)
    for pid, node in pid_node.items():
        try:
            conn = rpyc.classic.connect(node)
            p = conn.modules.psutil.Process(pid)
            p.terminate()
            print 'Process with PID \033[32;49m' + str(pid) + \
              '\033[m on node \033[35;49m' + \
              node + '\033[m has been stopped.'
        except Exception as e:
            print 'Process with PID \033[32;49m' + str(pid) + \
              '\033[m on node \033[35;49m' + \
              node + '\033[m does not exist.'


def dry_run(cfg):
    print 'config file: ', args.config
    print 'run type: ', args.dry
    print 'job type: ', args.foreground
    print 'batch key: ', get_script_batch_key(cfg)
    print 'nodes: ', get_nodes(cfg)
    print 'active batch path: ', cfg.get('xscft','activeBatchPath')
    print 'exe path: ', cfg.get('xscft','exePath')
    print 'exe name: ', cfg.get('xscft','exeName')
    print 'data path: ', cfg.get('xscft','dataPath')
    print 'data path suffix: ', cfg.get('xscft','dataPathSuffix')
    print 'wait time: ', cfg.get('xscft','waitTime')


if __name__ == '__main__':
    cfgfile = args.config
    cfg = SafeConfigParser(allow_no_value=True)
    cfg.optionxform = str
    cfg.read(cfgfile)

    if args.dry:
        dry_run(cfg)
        exit(0)

    # --- active batch ---
    active_batch_path = cfg.get('xscft','activeBatchPath')
    make_dir(active_batch_path)
    time_stamp = now2str(time_format)
    active_batch_name = 'batchdir-' + time_stamp
    active_batch_file = os.path.join(active_batch_path,active_batch_name)

    # --- exe ---
    exe_path = cfg.get('xscft','exePath')
    exe_name = cfg.get('xscft','exeName')
    exe_file = os.path.join(exe_path,exe_name)
    if not os.path.exists(exe_file):
        raise ValueError('Cannot find excutable file for scft!')
    
    # --- data dir ---
    # data dir pattern:
    #    $(data_path)/$(var_with_val}-$(data_path_suffix)/
    #                                       ${batchvar_with_val}/
    data_path =  cfg.get('xscft','dataPath')
    data_path_suffix =  cfg.get('xscft','dataPathSuffix')
    var_with_val = get_var_with_val(cfg)
    dir1 = os.path.join(data_path,var_with_val+data_path_suffix)
    make_dir(dir1)
    
    if not args.clean and not args.stop:
        param_file_name = 'param-' + time_stamp + '.ini'
        pid_node_file_name = 'pid-node-' + time_stamp
        pid_node_file = os.path.join(dir1, pid_node_file_name)
        cfg.set('xscft','activeBatchName', active_batch_name)
        cfg.set('xscft','pidNodeName', pid_node_file)
        target_file = os.path.join(dir1, param_file_name)
        with open(target_file, 'w') as tfile:
            cfg.write(tfile)
        #shutil.copy2(cfgfile,target_file)

    # batch in script
    bs_min = cfg.getfloat('xscft','batchScriptMin')
    bs_max = cfg.getfloat('xscft','batchScriptMax')
    bs_step = cfg.getfloat('xscft','batchScriptStep')
    pid_node = {}

    # --- stop processes ---
    if args.stop:
        pname = cfg.get('xscft','pidNodeName')
        pfile = os.path.join(dir1,pname)
        if not os.path.exists(pfile):
            exit(0)
        stop_processes(pfile)
        exit(0)

    # --- do clean up ---
    if args.clean:
        # remove batchdir-%Y%m%d-%H%M%S
        bname = cfg.get('xscft','activeBatchName')
        bfile = os.path.join(active_batch_path,bname)
        if os.path.exists(bfile):
            os.remove(bfile)
        # remove pid-node-min-max
        pname = cfg.get('xscft','pidNodeName')
        pfile = os.path.join(dir1,pname)
        if os.path.exists(pfile):
            os.remove(pfile)
        # remove data dirs
        for bs_val in np.arange(bs_min,bs_max+bs_step,bs_step):
            bs_key = get_script_batch_key(cfg)
            batchvar_with_val = vardict[bs_key]
            batchvar_with_val += str(bs_val)
            bs_dir = os.path.join(dir1,batchvar_with_val)
            if os.path.exists(bs_dir):
                shutil.rmtree(bs_dir)
        # remove top level data dir if it does not contain any dir
        dirlist = os.listdir(dir1)
        has_dir = False
        for dir in dirlist:
            if os.path.isdir(dir):
                has_dir = True
        if not has_dir:
            shutil.rmtree(dir1)
        exit(0)

    for bs_val in np.arange(bs_min,bs_max,bs_step):
        # --- lowest level dir ---
        bs_key = get_script_batch_key(cfg)
        batchvar_with_val = vardict[bs_key]
        batchvar_with_val += str(bs_val)
        bs_dir = os.path.join(dir1,batchvar_with_val)
        make_dir(bs_dir)

        # --- param.ini ---
        param_file = os.path.join(bs_dir,'param.ini')
        cfg.set('Model',vardict[bs_key],str(bs_val))
        with open(param_file,'w') as configfile:
            cfg.write(configfile)

        # --- exe file and submit ---
        bs_exe_name = bs_key + str(bs_val) + var_with_val
        if not args.dry:
            with open(active_batch_file,'a+') as activefile:
                activefile.write(bs_dir+'\n')

            bs_exe_file = os.path.join(bs_dir,bs_exe_name)
            shutil.copy(exe_file,bs_exe_file)

            wait_time = cfg.getint('xscft','waitTime')
            node_list = get_nodes(cfg)
            free_node = get_free_node(cfg,wait_time)

            pid = actual_run(bs_dir,bs_exe_name,free_node)
            pid_node[pid] = free_node

    if not args.dry:
        with open(pid_node_file,'w') as pfile:
            pickle.dump(pid_node,pfile)
