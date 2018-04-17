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

from wavelets_pytorch.wavelets import Morlet, Ricker, DOG
from wavelets_pytorch.transform import WaveletTransform, WaveletTransformTorch
from examples.plot import plot_scalogram

"""
Example script to plot SciPy and PyTorch implementation outputs side-to-side.
"""

fps = 20
dt  = 1.0/fps
dj  = 0.125
unbias = False
batch_size = 32
wavelet = Morlet(w0=6)

t_min = 0
t_max = 10
t = np.linspace(t_min, t_max, (t_max-t_min)*fps)

######################################
# Generating batch of random sinusoidals

random_frequencies = np.random.uniform(0.5, 4.0, size=batch_size)
batch  = np.asarray([np.sin(2*np.pi*f*t) for f in random_frequencies])
batch += np.random.normal(0, 0.2, batch.shape)  # Gaussian noise

######################################
# Performing wavelet transform

wa = WaveletTransform(dt, dj, wavelet, unbias=unbias)
wa_torch = WaveletTransformTorch(dt, dj, wavelet, unbias=unbias, cuda=True)

power = wa.power(batch)
power_torch = wa_torch.power(batch)

######################################
# Plotting

fig, ax = plt.subplots(1, 3, figsize=(12,3))
ax = ax.flatten()
ax[0].plot(t, batch[0])
ax[0].set_title(r'$f(t) = \sin(2\pi \cdot f t) + \mathcal{N}(\mu,\,\sigma^{2})$')
ax[0].set_xlabel('Time (s)')

# Plot scalogram for SciPy implementation
plot_scalogram(power[0], wa.fourier_periods, t, ax=ax[1], scale_legend=False)
ax[1].axhline(1.0 / random_frequencies[0], lw=1, color='k')
ax[1].set_title('Scalogram (SciPy)'.format(1.0/random_frequencies[0]))

# Plot scalogram for PyTorch implementation
plot_scalogram(power_torch[0], wa_torch.fourier_periods, t, ax=ax[2])
ax[2].axhline(1.0 / random_frequencies[0], lw=1, color='k')
ax[2].set_title('Scalogram (Torch)'.format(1.0/random_frequencies[0]))
ax[2].set_ylabel('')
ax[2].set_yticks([])

plt.tight_layout()
plt.show()