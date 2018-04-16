# Continuous Wavelet Transforms in PyTorch

This is a PyTorch implementation for the wavelet analysis outlined in [Torrence
and Compo (BAMS, 1998)](http://paos.colorado.edu/research/wavelets/). The code complements the excellent [implementation](https://github.com/aaren/wavelets/)
of Aaron O'Leary with a PyTorch wrapper to enable fast convolution on the GPU. Specifically, the code was written to speed-up the CWT computation for a large number of 1D signals and relies on `torch.nn.Conv1d` for convolution. 

![PyTorch Wavelets](/assets/scalogram_comparison.png "Scalogram Comparison")

### Usage ###

```
import numpy as np
from wavelets_pytorch.transform import WaveletTransform        # SciPy version
from wavelets_pytorch.transform import WaveletTransformTorch   # PyTorch version

dt = 0.1         # sampling frequency
dj = 0.125       # scale distribution parameter
batch_size = 32  # how many signals to process in parallel

# Batch of signals to process
batch = [batch_size x signal_length] 

# Initialize wavelet filter banks (scipy and torch implementation)
wa_scipy = WaveletTransform(dt, dj)
wa_torch = WaveletTransformTorch(dt, dj, cuda=True)

# Performing wavelet transform (and compute scalogram)
cwt_scipy = wa_scipy.cwt(batch)
cwt_torch = wa_torch.cwt(batch)

# For plotting, see the examples/plot.py function.
# ...
```

## Supported Wavelets

The wavelet implementations are taken from [here](https://github.com/aaren/wavelets/blob/master/wavelets/wavelets.py). Default is the Morlet wavelet.

## Benchmark

Performing parallel CWT computation on the GPU using PyTorch results in a significant speed-up. Increasing the batch size will give faster runtimes. The plot below shows a comaprison between the `scipy` versus `torch` implementation as function of the batch size `N` and input signal length. These results were obtained on a powerful Linux machine with NVIDIA Titan X GPU.

![PyTorchWavelets Benchmark](/assets/runtime_versus_signal_length.png "Runtimes Benchmark")

## Installation

```sh
pip install git+https://github.com/tomrunia/wavelets_pytorch
```

or install from a local copy:
```sh
git clone https://github.com/tomrunia/wavelets_pytorch.git
cd wavelets_pytorch
pip install -r requirements.txt
python setup.py install
# Optional: Run testsuite
pip install -r test-requirements.txt
nosetests
```

## Requirements

- Python 3.6 (other versions might work but have not been tested)
- Numpy (developed with 1.14.1)
- Scipy (developed with 1.0.0)
- PyTorch (developed with 0.3.1)

The core of the PyTorch implementation relies on the `torch.nn.Conv1d` module.

## License

MIT License

Copyright (c) 2018 Tom Runia (tomrunia@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
