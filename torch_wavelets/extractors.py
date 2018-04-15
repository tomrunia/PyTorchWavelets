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

import torch.nn as nn

from torch_wavelets.wavelets import Morlet

##########################################################################################

class TemporalFilterBankBase(metaclass=ABCMeta):

    def __init__(self, dt=1.0, dj=0.125, wavelet=Morlet(), unbias=False, signal_length=512):

        self._dt = dt
        self._dj = dj
        self._wavelet = wavelet
        self._unbias = unbias
        self._signal_length = signal_length

        self._scale_minimum = self.compute_minimum_scale()
        self._scales = self.compute_optimal_scales()
        self._filters = self._init_filters()

    @abstractmethod
    def compute(self, signal):
        raise NotImplementedError

    def _init_filters(self):
        filters = [None]*len(self.scales)
        for scale_idx, scale in enumerate(self._scales):
            # number of points needed to capture wavelet
            M = 10 * scale / self.dt
            # times to use, centred at zero
            t = np.arange((-M + 1) / 2., (M + 1) / 2.) * dt
            if len(t) % 2 == 0: t = t[0:-1]  # requires odd filter size
            # sample wavelet and normalise
            norm = (self.dt / scale) ** .5
            filters[scale_idx] = norm * self.wavelet(t, scale)
        return filters

    def compute_optimal_scales(self):
        """
        Form a set of scales to use in the wavelet transform.
        See Torrence & Combo (Eq. 9 and 10).
        """
        J = int((1 / self.dj) * np.log2(self.signal_length * self.dt / self._scale_minimum))
        scales = self._scale_minimum * 2 ** (dj * np.arange(0, J + 1))
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
            return (np.abs(self.compute(x)).T ** 2 / self.scales).T
        else:
            return np.abs(self.compute(x)) ** 2

    @property
    def fourier_period(self):
        return getattr(self.wavelet, 'fourier_period')

    @property
    def scales(self):
        return self._scales

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, dt):
        self._dt = dt
        # Needs to recompute scale distribution and filters
        self._scale_minimum = self.compute_minimum_scale()
        self._scales = self.compute_optimal_scales()
        self._filters = self._init_filters()

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
    def signal_length(self):
        return self._signal_length


##########################################################################################

class TemporalFilterBankSciPy(TemporalFilterBankBase):

    def __init__(self, dt=1.0, dj=0.125, wavelet=Morlet(), unbias=False, frequency=False):
        super(TemporalFilterBankSciPy,self).__init__(dt, dj, wavelet, unbias)
        self._frequency = frequency

    def compute(self, x):
        # append batch dimension
        if x.ndim == 1: x = x[None,:]
        # wavelets can be complex so output is complex
        num_examples = x.shape[0]
        output = np.zeros((num_examples, len(self.scales), x.shape[-1]), dtype=np.complex)
        for example_idx in range(num_examples):
            output[example_idx] = self._compute_single(x[example_idx])
        if num_examples == 1:
            output = output.squeeze(0)
        return output

    def _compute_single(self, x):
        assert x.ndim == 1, 'input signal must have single dimension.'
        output = np.zeros((len(self.scales), len(x)), dtype=np.complex)
        for scale_idx, filt in enumerate(self._filters):
            output[scale_idx,:] = scipy.signal.fftconvolve(x, filt, mode='same')
        return output


##########################################################################################

class TemporalFilterBankTorch(TemporalFilterBankBase, nn.Module):

    def __init__(self, dt=1.0, dj=0.125, wavelet=Morlet(), unbias=False):
        super(TemporalFilterBankTorch, self).__init__(dt, dj, wavelet, unbias)
        super(nn.Module, self).__init__()

    def compute(self, x):
        return x

    def _init_filters(self, scales, verbose=False):

        filters = [None]*len(scales)

        for ind, scale in enumerate(scales):
            # number of points needed to capture wavelet
            M = 10 * scale / dt
            # times to use, centred at zero
            t = np.arange((-M + 1) / 2., (M + 1) / 2.) * dt
            if len(t) % 2 == 0: t = t[0:-1]  # needs odd filter size

            # sample wavelet and normalise
            norm = (dt / scale) ** .5
            wavelet_data = norm * wavelet(t, scale)
            wavelet_data = np.asarray([np.real(wavelet_data), np.imag(wavelet_data)])
            wavelet_data = np.expand_dims(wavelet_data, 1)

            padding = self._get_padding('SAME', len(t))
            conv = nn.Conv1d(1, 2, kernel_size=len(t), padding=padding, bias=False)
            conv.weight.data = torch.from_numpy(wavelet_data.astype(np.float64))
            conv.require_gradient = False
            if self._cuda: conv.cuda()
            filters[ind] = conv

            print('filter {:02d} | scale = {:.2f} | shape = {}'.format(ind, scale, conv.weight.shape))

        return filters

    @staticmethod
    def _get_padding(padding_type, kernel_size):
        assert isinstance(kernel_size, int)
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            return (kernel_size - 1) // 2
        return 0




if __name__ == "__main__":

    import torch_wavelets.utils as utils
    import matplotlib.pyplot as plt

    fps = 20
    dt  = 1.0/fps
    dj  = 0.125
    w0  = 6
    unbias = False
    wavelet = Morlet()

    t_min = 0
    t_max = 10
    t = np.linspace(t_min, t_max, (t_max-t_min)*fps)

    batch_size = 12

    # Generate a batch of sine waves with random frequency
    random_frequencies = np.random.uniform(-0.5, 2.0, size=batch_size)
    batch = np.asarray([np.sin(2*np.pi*f*t) for f in random_frequencies])

    wa = TemporalFilterBankSciPy(dt, dj, wavelet, unbias)
    power = wa.power(batch)

    fig, ax = plt.subplots(3, 4, figsize=(16,8))
    ax = ax.flatten()
    for i in range(batch_size):
        utils.plot_scalogram(power[i], wa.scales, t, ax=ax[i])
        ax[i].axhline(1.0 / random_frequencies[i], lw=1, color='k')
    plt.show()






