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

import time
import numpy as np

from wavelets_pytorch.transform import WaveletTransform
from wavelets_pytorch.transform import WaveletTransformTorch

######################################

fps = 20
dt  = 1.0/fps
dj  = 0.125
unbias = False
t_min = 0

batch_sizes = np.asarray([1,8,16,32,64,128,256,512], np.int32)
durations = np.asarray([5,10,25,50,100], np.int32)
signal_lengths = durations*fps

num_runs = 5

runtimes_scipy = np.zeros((len(batch_sizes), len(signal_lengths), num_runs), np.float32)
runtimes_torch = np.zeros((len(batch_sizes), len(signal_lengths), num_runs), np.float32)

for batch_ind, batch_size in enumerate(batch_sizes):

    for length_ind, t_max in enumerate(durations):

        t = np.linspace(t_min, t_max, (t_max-t_min)*fps)
        print('#'*60)
        print('Benchmarking | BatchSize = {}, SignalLength = {}'.format(batch_size, signal_lengths[length_ind]))

        for run_ind in range(num_runs):

            random_frequencies = np.random.uniform(-0.5, 4.0, size=batch_size)
            batch = np.asarray([np.sin(2*np.pi*f*t) for f in random_frequencies])

            # Perform batch computation of SciPy implementation
            t_start = time.time()
            wa = WaveletTransform(dt, dj, unbias=unbias)
            power = wa.power(batch)
            runtimes_scipy[batch_ind,length_ind,run_ind] = time.time() - t_start
            #print("  Run {}/{} | SciPy: {:.2f}s".format(run_ind+1, num_runs, runtimes[batch_ind,length_ind,run_ind,0]))

            # Perform batch computation of Torch implementation
            t_start = time.time()
            wa = WaveletTransformTorch(dt, dj, unbias=unbias)
            power = wa.power(batch)
            runtimes_torch[batch_ind,length_ind,run_ind] = time.time() - t_start
            #print("  Run {}/{} | Torch: {:.2f}s".format(run_ind+1, num_runs, runtimes[batch_ind,length_ind,run_ind,1]))

        avg_scipy = np.mean(runtimes_scipy[batch_ind,length_ind,:])
        avg_torch = np.mean(runtimes_torch[batch_ind,length_ind,:])
        print('  Average SciPy: {:.2f}s'.format(avg_scipy))
        print('  Average Torch: {:.2f}s'.format(avg_torch))

np.save('./runtimes_scipy.npy', runtimes_scipy)
np.save('./runtimes_torch.npy', runtimes_torch)
