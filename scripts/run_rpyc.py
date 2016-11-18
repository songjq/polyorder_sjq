#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_rpyc
========

A script to start rpyc server on all available nodes.

Copyright (C) 2012 Yi-Xin Liu

"""

import psutil
#from subprocess import PIPE
import rpyc

all_nodes = ['console', 'c0101', 'c0102', 'c0103', 'c0104',
             'c0105', 'c0106', 'c0107', 'c0108', 'c0109',
             'c0110', 'c0111', 'c0112', 'c0113', 'c0114',
             'c0115', 'c0116', 'c0117', 'c0118']


def run_rpyc(node):
    try:
        conn = rpyc.classic.connect(node)
        print 'rpyc_classic is \033[1;37;49m already\033[m ' + \
                'running on \033[35;49m' + \
                node + '\033[m'
    except:
        print 'Start rpyc_classic sever on \033[35;49m' + \
                node + '\033[m.... ',
        cmd = "rsh " + node + " 'nohup rpyc_classic.py -q &'"
        psutil.Popen([cmd], shell=True)
        print 'Success!'


def run_all():
    for node in all_nodes:
        run_rpyc(node)


if __name__ == '__main__':
    run_all()
