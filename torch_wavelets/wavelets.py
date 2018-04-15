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
# Author: Aaron O'Leary (dev@aaren.me)
# Date Created: 2016-02-28

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy
import scipy.signal
import scipy.optimize
import scipy.special
from scipy.misc import factorial

__all__ = ['Morlet', 'Paul', 'DOG', 'Ricker', 'Marr', 'Mexican_hat']


class Morlet(object):
    def __init__(self, w0=6):
        """w0 is the nondimensional frequency constant. If this is
        set too low then the wavelet does not sample very well: a
        value over 5 should be ok; Terrence and Compo set it to 6.
        """
        self.w0 = w0
        if w0 == 6:
            # value of C_d from TC98
            self.C_d = 0.776

    def __call__(self, *args, **kwargs):
        return self.time(*args, **kwargs)

    def time(self, t, s=1.0, complete=True):
        """
        Complex Morlet wavelet, centred at zero.

        Parameters
        ----------
        t : float
            Time. If s is not specified, this can be used as the
            non-dimensional time t/s.
        s : float
            Scaling factor. Default is 1.
        complete : bool
            Whether to use the complete or the standard version.

        Returns
        -------
        out : complex
            Value of the Morlet wavelet at the given time

        See Also
        --------
        scipy.signal.gausspulse

        Notes
        -----
        The standard version::

            pi**-0.25 * exp(1j*w*x) * exp(-0.5*(x**2))

        This commonly used wavelet is often referred to simply as the
        Morlet wavelet.  Note that this simplified version can cause
        admissibility problems at low values of `w`.

        The complete version::

            pi**-0.25 * (exp(1j*w*x) - exp(-0.5*(w**2))) * exp(-0.5*(x**2))

        The complete version of the Morlet wavelet, with a correction
        term to improve admissibility. For `w` greater than 5, the
        correction term is negligible.

        Note that the energy of the return wavelet is not normalised
        according to `s`.

        The fundamental frequency of this wavelet in Hz is given
        by ``f = 2*s*w*r / M`` where r is the sampling rate.

        """
        w = self.w0

        x = t / s

        output = np.exp(1j * w * x)

        if complete:
            output -= np.exp(-0.5 * (w ** 2))

        output *= np.exp(-0.5 * (x ** 2)) * np.pi ** (-0.25)

        return output

    # Fourier wavelengths
    def fourier_period(self, s):
        """Equivalent Fourier period of Morlet"""
        return 4 * np.pi * s / (self.w0 + (2 + self.w0 ** 2) ** .5)

    def scale_from_period(self, period):
        """
        Compute the scale from the fourier period.
        Returns the scale
        """
        # Solve 4 * np.pi * scale / (w0 + (2 + w0 ** 2) ** .5)
        #  for s to obtain this formula
        coeff = np.sqrt(self.w0 * self.w0 + 2)
        return (period * (coeff + self.w0)) / (4. * np.pi)

    # Frequency representation
    def frequency(self, w, s=1.0):
        """Frequency representation of Morlet.

        Parameters
        ----------
        w : float
            Angular frequency. If `s` is not specified, i.e. set to 1,
            this can be used as the non-dimensional angular
            frequency w * s.
        s : float
            Scaling factor. Default is 1.

        Returns
        -------
        out : complex
            Value of the Morlet wavelet at the given frequency
        """
        x = w * s
        # Heaviside mock
        Hw = np.array(w)
        Hw[w <= 0] = 0
        Hw[w > 0] = 1
        return np.pi ** -.25 * Hw * np.exp((-(x - self.w0) ** 2) / 2)

    def coi(self, s):
        """The e folding time for the autocorrelation of wavelet
        power at each scale, i.e. the timescale over which an edge
        effect decays by a factor of 1/e^2.

        This can be worked out analytically by solving

            |Y_0(T)|^2 / |Y_0(0)|^2 = 1 / e^2
        """
        return 2 ** .5 * s


class Paul(object):
    def __init__(self, m=4):
        """Initialise a Paul wavelet function of order `m`.
        """
        self.m = m

    def __call__(self, *args, **kwargs):
        return self.time(*args, **kwargs)

    def time(self, t, s=1.0):
        """
        Complex Paul wavelet, centred at zero.

        Parameters
        ----------
        t : float
            Time. If `s` is not specified, i.e. set to 1, this can be
            used as the non-dimensional time t/s.
        s : float
            Scaling factor. Default is 1.

        Returns
        -------
        out : complex
            Value of the Paul wavelet at the given time

        The Paul wavelet is defined (in time) as::

            (2 ** m * i ** m * m!) / (pi * (2 * m)!) \
                    * (1 - i * t / s) ** -(m + 1)

        """
        m = self.m
        x = t / s

        const = (2 ** m * 1j ** m * factorial(m)) \
            / (np.pi * factorial(2 * m)) ** .5
        functional_form = (1 - 1j * x) ** -(m + 1)

        output = const * functional_form

        return output

    # Fourier wavelengths
    def fourier_period(self, s):
        """Equivalent Fourier period of Paul"""
        return 4 * np.pi * s / (2 * self.m + 1)

    def scale_from_period(self, period):
        raise NotImplementedError()

    # Frequency representation
    def frequency(self, w, s=1.0):
        """Frequency representation of Paul.

        Parameters
        ----------
        w : float
            Angular frequency. If `s` is not specified, i.e. set to 1,
            this can be used as the non-dimensional angular
            frequency w * s.
        s : float
            Scaling factor. Default is 1.

        Returns
        -------
        out : complex
            Value of the Paul wavelet at the given frequency

        """
        m = self.m
        x = w * s
        # Heaviside mock
        Hw = 0.5 * (np.sign(x) + 1)

        # prefactor
        const = 2 ** m / (m * factorial(2 * m - 1)) ** .5

        functional_form = Hw * (x) ** m * np.exp(-x)

        output = const * functional_form

        return output

    def coi(self, s):
        """The e folding time for the autocorrelation of wavelet
        power at each scale, i.e. the timescale over which an edge
        effect decays by a factor of 1/e^2.

        This can be worked out analytically by solving

            |Y_0(T)|^2 / |Y_0(0)|^2 = 1 / e^2
        """
        return s / 2 ** .5


class DOG(object):
    def __init__(self, m=2):
        """Initialise a Derivative of Gaussian wavelet of order `m`."""
        if m == 2:
            # value of C_d from TC98
            self.C_d = 3.541
        elif m == 6:
            self.C_d = 1.966
        else:
            pass
        self.m = m

    def __call__(self, *args, **kwargs):
        return self.time(*args, **kwargs)

    def time(self, t, s=1.0):
        """
        Return a Derivative of Gaussian wavelet,

        When m = 2, this is also known as the "Mexican hat", "Marr"
        or "Ricker" wavelet.

        It models the function::

            ``A d^m/dx^m exp(-x^2 / 2)``,

        where ``A = (-1)^(m+1) / (gamma(m + 1/2))^.5``
        and   ``x = t / s``.

        Note that the energy of the return wavelet is not normalised
        according to `s`.

        Parameters
        ----------
        t : float
            Time. If `s` is not specified, this can be used as the
            non-dimensional time t/s.
        s : scalar
            Width parameter of the wavelet.

        Returns
        -------
        out : float
            Value of the DOG wavelet at the given time

        Notes
        -----
        The derivative of the Gaussian has a polynomial representation:

        from http://en.wikipedia.org/wiki/Gaussian_function:

        "Mathematically, the derivatives of the Gaussian function can be
        represented using Hermite functions. The n-th derivative of the
        Gaussian is the Gaussian function itself multiplied by the n-th
        Hermite polynomial, up to scale."

        http://en.wikipedia.org/wiki/Hermite_polynomial

        Here, we want the 'probabilists' Hermite polynomial (He_n),
        which is computed by scipy.special.hermitenorm

        """
        x = t / s
        m = self.m

        # compute the Hermite polynomial (used to evaluate the
        # derivative of a Gaussian)
        He_n = scipy.special.hermitenorm(m)
        gamma = scipy.special.gamma

        const = (-1) ** (m + 1) / gamma(m + 0.5) ** .5
        function = He_n(x) * np.exp(-x ** 2 / 2)

        return const * function

    def fourier_period(self, s):
        """Equivalent Fourier period of derivative of Gaussian"""
        return 2 * np.pi * s / (self.m + 0.5) ** .5

    def scale_from_period(self, period):
        raise NotImplementedError()

    def frequency(self, w, s=1.0):
        """Frequency representation of derivative of Gaussian.

        Parameters
        ----------
        w : float
            Angular frequency. If `s` is not specified, i.e. set to 1,
            this can be used as the non-dimensional angular
            frequency w * s.
        s : float
            Scaling factor. Default is 1.

        Returns
        -------
        out : complex
            Value of the derivative of Gaussian wavelet at the
            given time
        """
        m = self.m
        x = s * w
        gamma = scipy.special.gamma
        const = -1j ** m / gamma(m + 0.5) ** .5
        function = x ** m * np.exp(-x ** 2 / 2)
        return const * function

    def coi(self, s):
        """The e folding time for the autocorrelation of wavelet
        power at each scale, i.e. the timescale over which an edge
        effect decays by a factor of 1/e^2.

        This can be worked out analytically by solving

            |Y_0(T)|^2 / |Y_0(0)|^2 = 1 / e^2
        """
        return 2 ** .5 * s


class Ricker(DOG):
    def __init__(self):
        """The Ricker, aka Marr / Mexican Hat, wavelet is a
        derivative of Gaussian order 2.
        """
        DOG.__init__(self, m=2)
        # value of C_d from TC98
        self.C_d = 3.541


# aliases for DOG2
Marr = Ricker
Mexican_hat = Ricker