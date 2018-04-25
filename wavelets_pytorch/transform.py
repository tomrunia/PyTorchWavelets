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

import six
from abc import ABCMeta, abstractmethod

import numpy as np
import scipy.signal
import scipy.optimize

import torch
from torch.autograd import Variable

from wavelets_pytorch.wavelets import Morlet
from wavelets_pytorch.network import TorchFilterBank

##########################################################################################

@six.add_metaclass(ABCMeta)
class WaveletTransformBase(object):
    """

    Base class for the Continuous Wavelet Transform as described in:
        "Torrence & Combo, A Practical Guide to Wavelet Analysis (BAMS, 1998)"

    This class is a abstract super class for child classes:
        WaveletTransform      => implements CWT in SciPy
        WaveletTransformTorch => implements CWT in PyTorch

    For a more detailed explanation of the parameters, the original code serves as reference:
        https://github.com/aaren/wavelets/blob/master/wavelets/transform.py#L145

    """

    def __init__(self, dt=1.0, dj=0.125, wavelet=Morlet(), unbias=False):
        """
        :param dt: float, sample spacing
        :param dj: float, scale distribution parameter
        :param wavelet: wavelet object, see 'wavelets.py'
        :param unbias: boolean, whether to unbias the power spectrum
        """
        self._dt = dt
        self._dj = dj
        self._wavelet = wavelet
        self._unbias = unbias
        self._scale_minimum = self.compute_minimum_scale()
        self._signal_length = None  # initialize on first call
        self._scales  = None        # initialize on first call
        self._filters = None        # initialize on first call

    @abstractmethod
    def cwt(self, x):
        raise NotImplementedError

    def _build_filters(self):
        """
        Determines the optimal scale distribution (see. Torrence & Combo, Eq. 9-10),
        and then initializes the filter bank consisting of rescaled versions
        of the mother wavelet. Also includes normalization. Code is based on:
        https://github.com/aaren/wavelets/blob/master/wavelets/transform.py#L88
        """
        self._scale_minimum = self.compute_minimum_scale()
        self._scales = self.compute_optimal_scales()

        self._filters = [None]*len(self.scales)
        for scale_idx, scale in enumerate(self._scales):
            # Number of points needed to capture wavelet
            M = 10 * scale / self.dt
            # Times to use, centred at zero
            t = np.arange((-M + 1) / 2., (M + 1) / 2.) * self.dt
            if len(t) % 2 == 0: t = t[0:-1]  # requires odd filter size
            # Sample wavelet and normalise
            norm = (self.dt / scale) ** .5
            self._filters[scale_idx] = norm * self.wavelet(t, scale)

    def compute_optimal_scales(self):
        """
        Determines the optimal scale distribution (see. Torrence & Combo, Eq. 9-10).
        :return: np.ndarray, collection of scales
        """
        if self.signal_length is None:
            raise ValueError('Please specify signal_length before computing optimal scales.')
        J = int((1 / self.dj) * np.log2(self.signal_length * self.dt / self._scale_minimum))
        scales = self._scale_minimum * 2 ** (self.dj * np.arange(0, J + 1))
        return scales

    def compute_minimum_scale(self):
        """
        Choose s0 so that the equivalent Fourier period is 2 * dt.
        See Torrence & Combo Sections 3f and 3h.
        :return: float, minimum scale level
        """
        dt = self.dt
        def func_to_solve(s):
            return self.fourier_period(s) - 2 * dt
        return scipy.optimize.fsolve(func_to_solve, 1)[0]

    def power(self, x):
        """
        Performs CWT and converts to a power spectrum (scalogram).
        See Torrence & Combo, Section 4d.
        :param x: np.ndarray, batch of input signals of shape [n_batch,signal_length]
        :return: np.ndarray, scalogram for each signal [n_batch,n_scales,signal_length]
        """
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

    @property
    def complex_wavelet(self):
        return np.iscomplexobj(self._filters[0])

    @property
    def output_dtype(self):
        return np.complex128 if self.complex_wavelet else np.float64

##########################################################################################

class WaveletTransform(WaveletTransformBase):

    def __init__(self, dt=1.0, dj=0.125, wavelet=Morlet(), unbias=False):
        """
        This is SciPy version of the CWT filter bank. Main work for this filter bank
        is performed by the convolution implementated in 'scipy.signal.convolve'

        :param dt: float, sample spacing
        :param dj: float, scale distribution parameter
        :param wavelet: wavelet object, see 'wavelets.py'
        :param unbias: boolean, whether to unbias the power spectrum
        """
        super(WaveletTransform,self).__init__(dt, dj, wavelet, unbias)

    def cwt(self, x):
        """
        Implements the continuous wavelet transform on a batch of signals. All signals
        in the batch must have the same length, otherwise manual zero padding has to be
        applied. On the first call, the signal length is used to determines the optimal
        scale distribution and uses this for initialization of the wavelet filter bank.
        If there is only one example in the batch the batch dimension is squeezed.

        :param x: np.ndarray, batch of signals of shape [n_batch,signal_length]
        :return: np.ndarray, CWT for each signal in the batch [n_batch,n_scales,signal_length]
        """

        # Append batch dimension
        if x.ndim == 1:
            x = x[None,:]

        num_examples  = x.shape[0]
        signal_length = x.shape[-1]

        if signal_length != self.signal_length or not self._filters:
            # First call initializtion, or change in signal length. Note that calling
            # this also determines the optimal scales and initialized the filter bank.
            self.signal_length = signal_length

        # Wavelets can be complex so output is complex (np.float64 or np.complex128)
        cwt = np.zeros((num_examples, len(self.scales), x.shape[-1]), self.output_dtype)
        for example_idx in range(num_examples):
            cwt[example_idx] = self._compute_single(x[example_idx])

        # Squeeze batch dimension if single example
        if num_examples == 1:
            cwt = cwt.squeeze(0)
        return cwt

    def _compute_single(self, x):
        assert x.ndim == 1, 'Input signal must have single dimension.'
        output = np.zeros((len(self.scales), len(x)), self.output_dtype)
        for scale_idx, filt in enumerate(self._filters):
            output[scale_idx,:] = scipy.signal.convolve(x, filt, mode='same')
        return output

##########################################################################################

class WaveletTransformTorch(WaveletTransformBase):

    def __init__(self, dt=1.0, dj=0.125, wavelet=Morlet(), unbias=False, cuda=True):
        """
        This is PyTorch version of the CWT filter bank. Main work for this filter bank
        is performed by the convolution implementated in 'torch.nn.Conv1d'. Actual
        convolutions are performed by the helper class defined in 'network.py' which
        implements a 'torch.nn.module' that contains the convolution filters.

        :param dt: float, sample spacing
        :param dj: float, scale distribution parameter
        :param wavelet: wavelet object, see 'wavelets.py'
        :param unbias: boolean, whether to unbias the power spectrum
        :param cuda: boolean, whether to run convolutions on the GPU
        """
        super(WaveletTransformTorch, self).__init__(dt, dj, wavelet, unbias)
        self._cuda = cuda
        self._extractor = TorchFilterBank(self._filters, cuda)

    def cwt(self, x):
        """
        Implements the continuous wavelet transform on a batch of signals. All signals
        in the batch must have the same length, otherwise manual zero padding has to be
        applied. On the first call, the signal length is used to determines the optimal
        scale distribution and uses this for initialization of the wavelet filter bank.
        If there is only one example in the batch the batch dimension is squeezed.

        :param x: np.ndarray, batch of signals of shape [n_batch,signal_length]
        :return: np.ndarray, CWT for each signal in the batch [n_batch,n_scales,signal_length]
        """

        if x.ndim == 1:
            # Append batch_size and chn_in dimensions
            # [signal_length] => [n_batch,1,signal_length]
            x = x[None,None,:]
        elif x.ndim == 2:
            # Just append chn_in dimension
            # [n_batch,signal_length] => [n_batch,1,signal_length]
            x = x[:,None,:]

        num_examples  = x.shape[0]
        signal_length = x.shape[-1]

        if signal_length != self.signal_length or not self._filters:
            # First call initializtion, or change in signal length. Note that calling
            # this also determines the optimal scales and initialized the filter bank.
            self.signal_length = signal_length

        # Move to GPU and perform CWT computation
        x = torch.from_numpy(x).type(torch.FloatTensor)
        x.requires_grad_(requires_grad=False)

        if self._cuda: x = x.cuda()
        cwt = self._extractor(x)

        # Move back to CPU
        cwt = cwt.detach()
        if self._cuda:  cwt = cwt.cpu()
        cwt = cwt.numpy()

        if self.complex_wavelet:
            # Combine real and imag parts, returns object of shape
            # [n_batch,n_scales,signal_length] of type np.complex128
            cwt = (cwt[:,:,0,:] + cwt[:,:,1,:]*1j).astype(self.output_dtype)
        else:
            # Just squeeze the chn_out dimension (=1) to obtain an object of shape
            # [n_batch,n_scales,signal_length] of type np.float64
            cwt = np.squeeze(cwt, 2).astype(self.output_dtype)

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