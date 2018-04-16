import sys
from setuptools import setup

if sys.version_info < (3,4):
    sys.exit('Sorry, Python versions lower than 3.4 are not supported.')

setup(
    name='wavelets_pytorch',
    version='0.1',
    python_requires='>3.4.0',
    author='Tom Runia',
    author_email='tomrunia@gmail.com',
    url='https://github.com/tomrunia/PyTorchWavelets',
    description='Wavelet Transform in PyTorch',
    long_description='Fast CPU/CUDA implementation of the Continuous Wavelet Transform in PyTorch.',
    license='MIT',
    packages=['wavelets_pytorch'],
    scripts=[]
)