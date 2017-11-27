from scipy.special import factorial
from numpy.linalg import solve
import numpy as np

import os
from numpy import random as random
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.integrate as integrate
from mpl_toolkits.mplot3d import axes3d
from scipy.ndimage import gaussian_filter1d as gaussian
import numpy.polynomial.polynomial as P

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

dt = 0.01
t = np.arange(0, 2*np.pi, dt)
x = np.sin(t) + 0.0001 * random.normal(size = len(t))

u = np.zeros_like(x)
a = np.zeros_like(x)
l = 10

for i in range(l, len(t) - l):
	xx = x[i-l:i+l+1]

	u[i] = (xx * get_weights(1, l * 2 + 1, dt)).sum()
	a[i] = (xx * get_weights(2, l * 2 + 1, dt)).sum()


plt.figure()
plt.plot(t, x, label = 'x')
plt.plot(t, u, label = 'u')
plt.plot(t, a, label = 'a')
plt.legend()
plt.grid()
plt.show()