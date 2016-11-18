#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
common.py
=========

Constants and common functions using by polyorder scripts.

Copyright (C) 2012 Yi-Xin Liu

"""

import os
import sys
import time
import datetime

import rpyc

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

default_node_list = ['c0109','c0114']

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
    conn = rpyc.classic.connect(node)
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
