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
# Date Created: 2018-XX-XX

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from torch_wavelets.wavelets import *



def scale_distribution(wavelet, min_period, max_period, dj=0.125):
    """
    Given a minimum and maximum period compute a distribution of scales. The choice of
    dj depends on the width in spectral space of the wavelet function.
    For the Morlet, dj=0.5 is the largest that still adequately samples scale.
    Smaller dj gives finer scale resolution.

    :param wavelet: wavelet instance from (Morlet, Ricker, MexicanHat, Marr)
    :param min_period: float, minimum period
    :param max_period: float, maximum period
    :param dj: float, scale sample density
    :return: np.ndarray, containing a list of scales
    """

    assert isinstance(wavelet, (Morlet, Ricker, Marr, Mexican_hat))
    scale_min = wavelet.scale_from_period(min_period)
    scale_max = wavelet.scale_from_period(max_period)
    num_scales = int(np.ceil((1.0/dj) * np.log2(scale_max/scale_min)))
    j = np.arange(0, num_scales+1)
    scales = scale_min*np.power(2, j*dj)
    return scales