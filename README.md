# Continuous Wavelet Transforms in PyTorch

This is a PyTorch implementation for the wavelet analysis outlined in [Torrence
and Compo (BAMS, 1998)](http://paos.colorado.edu/research/wavelets/). The code builds upon the excellent [implementation](https://github.com/aaren/wavelets/)
of Aaron O'Leary by adding a PyTorch filter bank wrapper to enable fast convolution on the GPU. Specifically, the code was written to speed-up the CWT computation for a large number of 1D signals and relies on `torch.nn.Conv1d` for convolution. 

![PyTorch Wavelets](/assets/scalogram_comparison.png "Scalogram Comparison")

## Citation

If you found this code useful, please cite our paper [Repetition Estimation](https://link.springer.com/article/10.1007/s11263-019-01194-0) (IJCV, 2019):

    @article{runia2019repetition,
      title={Repetition estimation},
      author={Runia, Tom FH and Snoek, Cees GM and Smeulders, Arnold WM},
      journal={International Journal of Computer Vision},
      volume={127},
      number={9},
      pages={1361--1383},
      year={2019},
      publisher={Springer}
    }
    
## Usage

In addition to the PyTorch implementation defined in `WaveletTransformTorch` the original SciPy version is also included in `WaveletTransform` for completeness. As the GPU implementation highly benefits from parallelization, the `cwt` and `power` methods expect signal batches of shape `[num_signals,signal_length]` instead of individual signals. 

```python
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

Performing parallel CWT computation on the GPU using PyTorch results in a significant speed-up. Increasing the batch size will give faster runtimes. The plot below shows a comaprison between the `scipy` versus `torch` implementation as function of the batch size `N` and input signal length. These results were obtained on a powerful Linux desktop with NVIDIA Titan X GPU.

<a href="/assets/runtime_versus_signal_length.png"><img src="/assets/runtime_versus_signal_length.png" width="700px" ></a>

## Installation

Clone and install:

```sh
git clone https://github.com/tomrunia/PyTorchWavelets.git
cd PyTorchWavelets
pip install -r requirements.txt
python setup.py install
```

## Requirements

- Python 2.7 or 3.6 (other versions might also work)
- Numpy (developed with 1.14.1)
- Scipy (developed with 1.0.0)
- PyTorch >= 0.4.0

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
