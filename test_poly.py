import os
import numpy as np
from numpy import random as random
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.integrate as integrate
from mpl_toolkits.mplot3d import axes3d
from scipy.ndimage import gaussian_filter1d as gaussian
import numpy.polynomial.polynomial as P

N = 100
err_U = 0
err_A = 0

for _ in range(N):

	dt = 0.001
	t = np.arange(0, 2*np.pi, dt)
	x = np.sin(t) + 0.001 * random.normal(size = len(t))

	u = np.zeros_like(x)
	a = np.zeros_like(x)
	l = 55

	for i in range(l, len(t) - l):
		xx = x[i-l:i+l+1]
		tt = t[i-l:i+l+1]

		c = P.polyfit(tt, xx, deg = 2)

		c1 = P.polyder(c)
		c2 = P.polyder(c1)

		u[i] = P.polyval(t[i], c1)
		a[i] = P.polyval(t[i], c2)

	err_U += np.mean(np.abs((u[l:-l] - np.cos(t)[l:-l]) / np.cos(t)[l:-l]))
	err_A += np.mean(np.abs((a[l:-l] + np.sin(t)[l:-l]) / np.sin(t)[l:-l]))

print err_U / N, err_A / N

plt.figure()
plt.plot(t, x, label = 'x')
plt.plot(t, u, label = 'u')
plt.plot(t, a, label = 'a')
plt.legend()
plt.grid()
plt.show()