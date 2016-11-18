#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
render
======

A structure data renderer.

Copyright (C) 2012 Yi-Xin Liu

"""

import argparse

import numpy as np
import scipy.io
from mayavi import mlab

#import matplotlib
#if(not args.display):
    #matplotlib.use('Agg')
    #mlab.options.offscreen = True #Error in running

import matplotlib.pyplot as plt
from matplotlib import colors

def render_1d(struct_name, data_file, img_file, period, **kwargs):
    ''' render 1D structure.

    :param struct_name: the struct variable name to be rendered
    :type struct_name: string
    :param data_file: the MAT file containing sturct varible
    :type data_file: string
    :param img_file: the file name of the image file
    :type img_file: string
    :param period: how many periods to draw
    :type img_file: integer
    :param show_img: if True, show image on the screen
    :type show_img: bool
    :param kwargs: any extra key words arguments will be passed to plot functions

    '''

    data = scipy.io.loadmat(data_file)

    struct = data[struct_name]
    struct = np.tile(struct,period)

    x = data['x']
    Lx = np.size(x)
    xa = x[0]
    dx = x[1] - xa
    rx = np.zeros(Lx*period)
    for i in xrange(Lx*period):
        rx[i] = i * dx

    # No frame, white background
    fig = plt.figure(dpi=80, facecolor='w')
    # full figure subplot
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(rx,struct,**kwargs)
    plt.savefig(img_file)


def render_2d(struct_name, data_file, img_file, period, 
              levels=None, cmap=None,
              **kwargs):
    ''' Render 2D structure.

    :param struct_name: the struct variable name to be rendered
    :type struct_name: string
    :param data_file: the MAT file containing sturct varible
    :type data_file: string
    :param img_file: the file name of the image file
    :type img_file: string
    :param period: how many periods to draw
    :type img_file: integer
    :param show_img: if True, show image on the screen
    :type show_img: bool
    :param levels: how many contour levels
    :type levels: integer
    :param cmap: colormap for contour plot
    :type cmap: `Colormap`
    :param kwargs: any extra key words arguments will be passed to plot functions

    '''

    data = scipy.io.loadmat(data_file)

    struct = data[struct_name]
    repeat = (period,period)
    struct = np.tile(struct,repeat)

    x = data['x']
    y = data['y']
    Lx, Ly = np.shape(x)
    xa = x[0,0]
    dx1 = x[1,0] - xa
    dx2 = x[0,1] - xa
    yc = y[0,0]
    dy1 = y[1,0] - yc
    dy2 = y[0,1] - yc
    rx = np.zeros((Lx*period,Ly*period))
    ry = np.zeros((Lx*period,Ly*period))
    for (i,j) in np.ndindex(Lx*period,Ly*period):
        rx[i,j] = i * dx1 + j * dx2
        ry[i,j] = i * dy1 + j * dy2

    dx = rx.max() - rx.min()
    dy = ry.max() - ry.min()
    w, h = plt.figaspect(float(dy / dx))  # float is must
    # No frame, white background, w/h aspect ratio figure
    fig = plt.figure(figsize=(w, h), frameon=False,
                     dpi=80, facecolor='w')
    # full figure subplot, no border, no axes
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, axisbg='w')
    # no ticks
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Default: there are 256 contour levels
    if levels is None:
        step = (struct.max() - struct.min()) / 256
        levels = np.arange(struct.min(), struct.max() + step, step)
    # Default: colormap is monochromatic red
    if cmap is None:
        clr = np.zeros((256, 3))
        for i in np.arange(256):
            clr[i, 0] = i / 255.0
        cmap = colors.ListedColormap(clr)
    # actual plot
    ax.contourf(rx, ry, struct, levels=levels,
                cmap=cmap, antialiased=False, **kwargs)
    #ax.contourf(rx,ry,struct)
    plt.savefig(img_file)


def render_3d(struct_name, data_file, img_file, period, **kwargs):
    ''' Render 3D structure.

    :param struct_name: the struct variable name to be rendered
    :type struct_name: string
    :param data_file: the MAT file containing sturct varible
    :type data_file: string
    :param img_file: the file name of the image file
    :type img_file: string
    :param period: how many periods to draw
    :type img_file: integer
    :param show_img: if True, show image on the screen
    :type show_img: bool
    :param kwargs: any extra key words arguments will be passed to plot functions

    '''

    data = scipy.io.loadmat(data_file)

    struct = data[struct_name]
    repeat = (period,period,period)
    struct = np.tile(struct,repeat)

    x = data['x']
    y = data['y']
    z = data['z']
    Lx, Ly, Lz = np.shape(x)
    xa = x[0,0,0]
    dx1 = x[1,0,0] - xa
    dx2 = x[0,1,0] - xa
    dx3 = x[0,0,1] - xa
    yc = y[0,0,0]
    dy1 = y[1,0,0] - yc
    dy2 = y[0,1,0] - yc
    dy3 = y[0,0,1] - yc
    ze = z[0,0,0]
    dz1 = z[1,0,0] - ze
    dz2 = z[0,1,0] - ze
    dz3 = z[0,0,1] - ze
    rx = np.zeros((Lx*period,Ly*period,Lz*period))
    ry = np.zeros((Lx*period,Ly*period,Lz*period))
    rz = np.zeros((Lx*period,Ly*period,Lz*period))
    for (i,j,k) in np.ndindex(Lx*period,Ly*period,Lz*period):
        rx[i,j,k] = i * dx1 + j * dx2 + k * dx3
        ry[i,j,k] = i * dy1 + j * dy2 + k * dy3
        rz[i,j,k] = i * dz1 + j * dz2 + k * dz3

    mlab.contour3d(rx,ry,rz,struct,**kwargs)
    #mlab.pipeline.volume(mlab.pipeline.scalar_field(rx, ry, rz, struct))
    mlab.savefig(img_file)
    mlab.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-a', '--space',
                        default='2',
                        help='the space dimensionality.')
    
    parser.add_argument('-s', '--struct',
                        default='struct',
                        help='the variable name in data file to render.')
    
    parser.add_argument('-d', '--data', default='data.mat',
                        help='the data file to read.')
    
    parser.add_argument('-i', '--image', default='struct.png',
                        help='the image file to write.')
    
    parser.add_argument('-p', '--period', default=2, type=int,
                        help='how many periods to render.' + \
                             'same for each dimension.')
    
    parser.add_argument('-y', '--display', action='store_true',
                        help='dispaly the image')
    
    args = parser.parse_args()

    struct = args.struct
    data_file = args.data
    img_file = args.image
    period = args.period

    if(args.space == '1'):
        render_1d(struct, data_file, img_file, period)
    elif(args.space == '2'):
        render_2d(struct, data_file, img_file, period)
    else:
        render_3d(struct, data_file, img_file, period)

