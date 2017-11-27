import numpy as np
from numpy.linalg import solve
import scipy.interpolate as interp
import numpy.polynomial.polynomial as P
from scipy.ndimage import gaussian_filter1d as gaussian

from scipy.ndimage import _ni_support, _nd_image

from scipy.special import factorial

def get_weights(derivative, stencil, dt):
	d = int(derivative)
	s = int(stencil)
	n = (s - 1) / 2

	b = np.zeros(s)

	b[d] = factorial(d)
	b /= dt**d

	A = np.zeros((s, s))

	for i in range(s):
		for j in range(s):
			if j == n and i != 0:
				continue

			A[i,j] = (j - n)**i

	weights = solve(A, b)

	return weights

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

class filters:
	def __init__(self, names, streamline, T, dt, l):
		self.streamline = streamline
		self.T = T
		self.dt = dt
		self.l = l

		self.U = {}
		self.A = {}

		for n in names:
			self.U[n] = []
			self.A[n] = []

	def process(self):
		l = self.l

		for n in self.U.keys():

			for i in range(len(self.T)):
				U = np.zeros((len(self.T[i]), 4))
				A = np.zeros((len(self.T[i]), 4))

				for j in range(l, len(self.T[i]) - l):

					x = self.streamline[i][j-l:j+l+1, 0]
					y = self.streamline[i][j-l:j+l+1, 1]
					z = self.streamline[i][j-l:j+l+1, 2]

					X = self.streamline[i][j-l:j+l+1]

					t = self.T[i][j-l:j+l+1]

					if n == "Finite difference":
						weights = get_weights(1, l * 2 + 1, self.dt)

						U[j, 0] = (x * weights).sum()
						U[j, 1] = (y * weights).sum()
						U[j, 2] = (z * weights).sum()

						weights = get_weights(2, l * 2 + 1, self.dt)

						A[j, 0] = (x * weights).sum()
						A[j, 1] = (y * weights).sum()
						A[j, 2] = (z * weights).sum()

					elif n == "Spline":
						spx = interp.UnivariateSpline(t, x, k = 2)
						spy = interp.UnivariateSpline(t, y, k = 2)
						spz = interp.UnivariateSpline(t, z, k = 2)

						U[j, 0] = spx(t[l], 1)
						U[j, 1] = spy(t[l], 1)
						U[j, 2] = spz(t[l], 1)

						A[j, 0] = spx(t[l], 2)
						A[j, 1] = spy(t[l], 2)
						A[j, 2] = spz(t[l], 2)

					elif n == "Gaussian":
						U[j, 0] = gaussian_filter1d(x, radius = l, order = 1, truncate = 1, dt = self.dt)[l] #/ self.dt
						U[j, 1] = gaussian_filter1d(y, radius = l, order = 1, truncate = 1, dt = self.dt)[l] #/ self.dt
						U[j, 2] = gaussian_filter1d(z, radius = l, order = 1, truncate = 1, dt = self.dt)[l] #/ self.dt

						A[j, 0] = gaussian_filter1d(x, radius = l, order = 2, truncate = 1, dt = self.dt)[l] #/ self.dt**2
						A[j, 1] = gaussian_filter1d(y, radius = l, order = 2, truncate = 1, dt = self.dt)[l] #/ self.dt**2
						A[j, 2] = gaussian_filter1d(z, radius = l, order = 2, truncate = 1, dt = self.dt)[l] #/ self.dt**2

					elif n == "Polynomial":
						c = P.polyfit(t, X, deg = 2)

						c1 = P.polyder(c)
						c2 = P.polyder(c1)

						U[j, :-1] = P.polyval(t[l], c1)
						A[j, :-1] = P.polyval(t[l], c2)

					else:
						raise(Error("Unknow filter type"))

				U[:, -1] = np.sqrt(U[:, 0]**2 + U[:, 1]**2 + U[:, 2]**2)
				A[:, -1] = np.sqrt(A[:, 0]**2 + A[:, 1]**2 + A[:, 2]**2)

				self.U[n].append(U)
				self.A[n].append(A)

		return self.U, self.A


