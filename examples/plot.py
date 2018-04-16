# MIT License
# 
# Copyright (c) 2018 Tom Runia
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-04-16

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_scalogram(power, scales, t, normalize_columns=True, cmap=None, ax=None, scale_legend=True):

    if not cmap:
        cmap = plt.get_cmap("PuBu_r")

    if ax is None:
        fig, ax = plt.subplots()

    if normalize_columns:
        power = power/np.max(power, axis=0)

    T, S = np.meshgrid(t, scales)
    cnt = ax.contourf(T, S, power, 100, cmap=cmap)

    # Fix for saving as PDF (aliasing)
    for c in cnt.collections:
        c.set_edgecolor("face")

    ax.set_yscale('log')
    ax.set_ylabel("Scale (Log Scale)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Wavelet Power Spectrum")

    if scale_legend:

        def format_axes_label(x, pos):
            return "{:.2f}".format(x)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(
            cnt, cax=cax, ticks=[np.min(power), 0, np.max(power)],
            format=ticker.FuncFormatter(format_axes_label))

    return ax






