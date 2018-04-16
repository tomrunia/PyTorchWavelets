# Continuous Wavelet Transforms in PyTorch

This is a PyTorch implementation for the wavelet analysis outlined in [Torrence
and Compo][TC_Home] (BAMS, 1998). The code is based on the excellent [SciPy implementation](https://github.com/aaren/wavelets/)
of Aaron O'Leary.

### Usage ###

```
todo
```

#### How would you plot this? ####

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
T, S = np.meshgrid(t, scales)
ax.contourf(T, S, power, 100)
ax.set_yscale('log')
fig.savefig('test_wavelet_power_spectrum.png')
```

See the [tests](./tests.py) for more plotting examples.

#### What wavelet functions can I use? ####

The default is to use the Morlet. The Ricker (aka Mexican hat, aka
Marr) is also available.

You can write your own wavelet functions, in either time or
frequency. Just follow the example of Morlet in the source.

You specify the function to use when starting the analysis:

```python
from wavelets import Ricker

wa = WaveletAnalysis(data=x, wavelet=Ricker(), dt=dt)
```

### Installation ###

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

### Requirements ###

- Python 3.6 (other versions might work but have not been tested)
- Numpy (developed with 1.14.1)
- Scipy (developed with 1.0.0)
- PyTorch (developed with 0.3.1)

Scipy is only used for `signal.fftconvolve` and `optimize.fsolve`,
and could potentially be removed.

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
