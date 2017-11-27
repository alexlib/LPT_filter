import os
import numpy as np
from numpy import random as random
from numpy.linalg import solve
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.integrate as integrate
from mpl_toolkits.mplot3d import axes3d
from scipy.ndimage import gaussian_filter1d as gaussian

from scipy.ndimage import _ni_support, _nd_image

def correlate1d(input, weights, axis=-1, output=None, mode="reflect",
                cval=0.0, origin=0):
    """Calculate a one-dimensional correlation along the given axis.
    The lines of the array along the given axis are correlated with the
    given weights.
    Parameters
    ----------
    %(input)s
    weights : array
        One-dimensional sequence of numbers.
    %(axis)s
    %(output)s
    %(mode)s
    %(cval)s
    %(origin)s
    Examples
    --------
    >>> from scipy.ndimage import correlate1d
    >>> correlate1d([2, 8, 0, 4, 1, 9, 9, 0], weights=[1, 3])
    array([ 8, 26,  8, 12,  7, 28, 36,  9])
    """
    input = np.asarray(input)
    if np.iscomplexobj(input):
        raise TypeError('Complex type not supported')
    output, return_value = _ni_support._get_output(output, input)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or weights.shape[0] < 1:
        raise RuntimeError('no filter weights given')
    if not weights.flags.contiguous:
        weights = weights.copy()
    axis = _ni_support._check_axis(axis, input.ndim)
    if (len(weights) // 2 + origin < 0) or (len(weights) // 2 +
                                            origin > len(weights)):
        raise ValueError('invalid origin')
    mode = _ni_support._extend_mode_to_code(mode)
    _nd_image.correlate1d(input, weights, axis, output, mode, cval,
                          origin)
    return return_value



def convolve1d(input, weights, axis=-1, output=None, mode="reflect",
               cval=0.0, origin=0):
    """Calculate a one-dimensional convolution along the given axis.
    The lines of the array along the given axis are convolved with the
    given weights.
    Parameters
    ----------
    %(input)s
    weights : ndarray
        One-dimensional sequence of numbers.
    %(axis)s
    %(output)s
    %(mode)s
    %(cval)s
    %(origin)s
    Returns
    -------
    convolve1d : ndarray
        Convolved array with same shape as input
    Examples
    --------
    >>> from scipy.ndimage import convolve1d
    >>> convolve1d([2, 8, 0, 4, 1, 9, 9, 0], weights=[1, 3])
    array([14, 24,  4, 13, 12, 36, 27,  0])
    """
    weights = weights[::-1]
    origin = -origin
    if not len(weights) & 1:
        origin -= 1
    return correlate1d(input, weights, axis, output, mode, cval, origin)


def _gaussian_kernel1d(sigma, order, radius):
    """
    Computes a 1D Gaussian convolution kernel.
    """
    if order < 0:
        raise ValueError('order must be non-negative')
    p = np.polynomial.Polynomial([0, 0, -0.5 / (sigma * sigma)])
    x = np.arange(-radius, radius + 1)
    phi_x = np.exp(p(x), dtype=np.double)
    phi_x /= phi_x.sum()
    if order > 0:
        q = np.polynomial.Polynomial([1])
        p_deriv = p.deriv()
        for _ in range(order):
            # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
            # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
            q = q.deriv() + q * p_deriv
        phi_x *= q(x)
    return phi_x

def my_kernel1d(sigma, order, radius, dt):
    order = int(order)
    if order < 0:
        raise ValueError('order must be non-negative')
    p = np.polynomial.Polynomial([0, 0, -0.5 / (sigma * sigma)])
    n = 2 * radius + 1
    x = np.linspace(-radius, radius, n, endpoint = True) * dt
    phi_x = np.exp(p(x), dtype=np.double)

    if order == 0:
    	return phi_x / phi_x.sum()

    else:
        q = np.polynomial.Polynomial([1])
        p_deriv = p.deriv()
        for _ in range(order):
            # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
            # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
            q = q.deriv() + q * p_deriv
        phi_x *= q(x)

        A = np.array([[phi_x.sum(), len(x)],
        			[np.convolve(phi_x, x**order, 'valid')[0], (x**order).sum()]])
        B = np.array([0, order])

        coef = solve(A, B)

    return phi_x * coef[0] + coef[1]


def gaussian_filter1d(input, radius, axis=-1, order=0, output=None,
                      mode="reflect", cval=0.0, truncate=4.0, dt=None):
    """One-dimensional Gaussian filter.
    Parameters
    ----------
    %(input)s
    sigma : scalar
        standard deviation for Gaussian kernel
    %(axis)s
    order : int, optional
        An order of 0 corresponds to convolution with a Gaussian
        kernel. A positive order corresponds to convolution with
        that derivative of a Gaussian.
    %(output)s
    %(mode)s
    %(cval)s
    truncate : float, optional
        Truncate the filter at this many standard deviations.
        Default is 4.0.
    Returns
    -------
    gaussian_filter1d : ndarray
    Examples
    --------
    >>> from scipy.ndimage import gaussian_filter1d
    >>> gaussian_filter1d([1.0, 2.0, 3.0, 4.0, 5.0], 1)
    array([ 1.42704095,  2.06782203,  3.        ,  3.93217797,  4.57295905])
    >>> gaussian_filter1d([1.0, 2.0, 3.0, 4.0, 5.0], 4)
    array([ 2.91948343,  2.95023502,  3.        ,  3.04976498,  3.08051657])
    >>> import matplotlib.pyplot as plt
    >>> np.random.seed(280490)
    >>> x = np.random.randn(101).cumsum()
    >>> y3 = gaussian_filter1d(x, 3)
    >>> y6 = gaussian_filter1d(x, 6)
    >>> plt.plot(x, 'k', label='original data')
    >>> plt.plot(y3, '--', label='filtered, sigma=3')
    >>> plt.plot(y6, ':', label='filtered, sigma=6')
    >>> plt.legend()
    >>> plt.grid()
    >>> plt.show()
    """
    sigma = radius * dt / truncate
    # make the radius of the filter equal to truncate standard deviations
    lw = int(radius)
    # Since we are calling correlate, not convolve, revert the kernel
    weights = my_kernel1d(sigma, order, lw, dt)[::-1]
    return correlate1d(input, weights, axis, output, mode, cval, 0)

dt = 0.01
t = np.arange(0, 2*np.pi, dt)
x = np.sin(t) + 0.01 * random.normal(scale = 0.3, size = len(t))
# x = np.sin(t) + 0.01 * random.rand(len(t))

u = np.zeros_like(x)
a = np.zeros_like(x)
l = 50

for i in range(l, len(t) - l):
	xx = x[i-l:i+l+1]

	u[i] = gaussian_filter1d(xx, radius = l, order = 1, truncate = 1, dt = dt)[l]
	a[i] = gaussian_filter1d(xx, radius = l, order = 2, truncate = 1, dt = dt)[l]

plt.figure()
plt.plot(t, x, label = 'x')
plt.plot(t, u, label = 'u')
plt.plot(t, a, label = 'a')
plt.legend()
plt.grid()
plt.show()