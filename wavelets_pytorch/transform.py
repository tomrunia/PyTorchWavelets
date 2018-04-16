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
# Date Created: 2018-04-15

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod

import numpy as np
import scipy.signal
import scipy.optimize

import torch
from torch.autograd import Variable

from wavelets_pytorch.wavelets import Morlet
from wavelets_pytorch.network import TorchFilterBank

##########################################################################################

class WaveletTransformBase(metaclass=ABCMeta):

    def __init__(self, dt=1.0, dj=0.125, wavelet=Morlet(), unbias=False, signal_length=None):
        self._dt = dt
        self._dj = dj
        self._wavelet = wavelet
        self._unbias = unbias
        self._signal_length = signal_length
        self._scale_minimum = self.compute_minimum_scale()
        self._scales  = None  # initialize on first call
        self._filters = None  # initialize on first call

    @abstractmethod
    def cwt(self, x):
        raise NotImplementedError

    def _build_filters(self):

        self._scale_minimum = self.compute_minimum_scale()
        self._scales = self.compute_optimal_scales()

        #print('Initializing filter bank with {} filters (scale_min = {:.2f}, scale_max = {:.1f})'.format(
        #    len(self.scales), self.scales[0], self.scales[-1]))

        self._filters = [None]*len(self.scales)
        for scale_idx, scale in enumerate(self._scales):
            # number of points needed to capture wavelet
            M = 10 * scale / self.dt
            # times to use, centred at zero
            t = np.arange((-M + 1) / 2., (M + 1) / 2.) * self.dt
            if len(t) % 2 == 0: t = t[0:-1]  # requires odd filter size
            # sample wavelet and normalise
            norm = (self.dt / scale) ** .5
            self._filters[scale_idx] = norm * self.wavelet(t, scale)

    def compute_optimal_scales(self):
        """
        Form a set of scales to use in the wavelet transform.
        See Torrence & Combo (Eq. 9 and 10).
        """
        if self.signal_length is None:
            raise ValueError('Please specify signal_length before computing optimal scales.')
        J = int((1 / self.dj) * np.log2(self.signal_length * self.dt / self._scale_minimum))
        scales = self._scale_minimum * 2 ** (self.dj * np.arange(0, J + 1))
        return scales

    def compute_minimum_scale(self):
        """
        Choose s0 so that the equivalent Fourier period is 2 * dt.
        :return: float, minimum scale
        """
        dt = self.dt
        def func_to_solve(s):
            return self.fourier_period(s) - 2 * dt
        return scipy.optimize.fsolve(func_to_solve, 1)[0]

    def power(self, x):
        if self.unbias:
            return (np.abs(self.cwt(x)).T ** 2 / self.scales).T
        else:
            return np.abs(self.cwt(x)) ** 2

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        # Needs to recompute scale distribution and filters
        self._dt = value
        self._build_filters()

    @property
    def signal_length(self):
        return self._signal_length

    @signal_length.setter
    def signal_length(self, value):
        # Needs to recompute scale distribution and filters
        self._signal_length = value
        self._build_filters()

    @property
    def wavelet(self):
        return self._wavelet

    @property
    def fourier_period(self):
        """ Return a function that calculates the equivalent Fourier. """
        return getattr(self.wavelet, 'fourier_period')

    @property
    def scale_from_period(self):
        """ Return a function that calculates the wavelet scale from the fourier period """
        return getattr(self.wavelet, 'scale_from_period')

    @property
    def fourier_periods(self):
        """ Return the equivalent Fourier periods for the scales used. """
        assert self._scales is not None, 'Wavelet scales are not initialized.'
        return self.fourier_period(self.scales)

    @property
    def fourier_frequencies(self):
        """ Return the equivalent frequencies. """
        return np.reciprocal(self.fourier_periods)

    @property
    def scales(self):
        return self._scales

    @property
    def dj(self):
        return self._dj

    @property
    def wavelet(self):
        return self._wavelet

    @property
    def unbias(self):
        return self._unbias

##########################################################################################

class WaveletTransform(WaveletTransformBase):

    def __init__(self, dt=1.0, dj=0.125, wavelet=Morlet(), unbias=False, frequency=False):
        super(WaveletTransform,self).__init__(dt, dj, wavelet, unbias)
        self._frequency = frequency

    def cwt(self, x):
        # append batch dimension
        if x.ndim == 1: x = x[None,:]
        # wavelets can be complex so output is complex
        num_examples  = x.shape[0]
        signal_length = x.shape[-1]

        if signal_length != self.signal_length or not self._filters:
            # Signal length different from initial, need to recompute optimal scales
            self.signal_length = signal_length

        cwt = np.zeros((num_examples, len(self.scales), x.shape[-1]), dtype=np.complex)
        for example_idx in range(num_examples):
            cwt[example_idx] = self._compute_single(x[example_idx])

        # Squeeze batch dimension if single example
        if num_examples == 1:
            cwt = cwt.squeeze(0)
        return cwt

    def _compute_single(self, x):
        assert x.ndim == 1, 'input signal must have single dimension.'
        output = np.zeros((len(self.scales), len(x)), dtype=np.complex)
        for scale_idx, filt in enumerate(self._filters):
            output[scale_idx,:] = scipy.signal.fftconvolve(x, filt, mode='same')
        return output

##########################################################################################

class WaveletTransformTorch(WaveletTransformBase):

    def __init__(self, dt=1.0, dj=0.125, wavelet=Morlet(), unbias=False, cuda=True):
        super(WaveletTransformTorch, self).__init__(dt, dj, wavelet, unbias)
        self._cuda = cuda
        self._extractor = TorchFilterBank(self._filters, cuda)

    def cwt(self, x):

        if x.ndim == 1:
            # Append batch_size and chn_in dimensions [T] => [N,CHN_IN,T]
            x = x[None,None,:]
        elif x.ndim == 2:
            # Append chn_in dimension [N,T] => [N,CHN_IN,T]
            x = x[:,None,:]

        num_examples  = x.shape[0]
        signal_length = x.shape[-1]

        if signal_length != self.signal_length or not self._filters:
            # Signal length different from initial, need to recompute optimal scales
            self.signal_length = signal_length

        # Move to GPU and convole signals
        x = Variable(torch.from_numpy(x).type(torch.FloatTensor), requires_grad=False)
        if self._cuda: x = x.cuda()
        cwt = self._extractor(x)

        # Move back to CPU
        cwt = cwt.data
        if self._cuda: cwt = cwt.cpu()
        cwt = cwt.numpy()

        # Combine real and imag parts
        cwt = cwt[:,:,0,:] + cwt[:,:,1,:]*1j

        # Squeeze batch dimension if single example
        if num_examples == 1:
            cwt = cwt.squeeze(0)

        return cwt

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        super(WaveletTransformTorch, self.__class__).dt.fset(self, value)
        self._extractor.set_filters(self._filters)

    @property
    def signal_length(self):
        return self._signal_length

    @signal_length.setter
    def signal_length(self, value):
        super(WaveletTransformTorch, self.__class__).signal_length.fset(self, value)
        self._extractor.set_filters(self._filters)