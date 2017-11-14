import os
import numpy as np
from numpy import random as random
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.integrate as integrate
from mpl_toolkits.mplot3d import axes3d
from scipy.ndimage import gaussian_filter1d as gaussian

dt = 0.01
t = np.arange(0, 2*np.pi, dt)
x = np.sin(t) + 0.001 * random.rand(len(t))

u = np.zeros_like(x)
a = np.zeros_like(x)
l = 5

for i in range(l, len(t) - l):
	xx = x[i-l:i+l+1]

	u[i] = gaussian(xx, 1.0, order = 1, truncate = l)[l] / dt
	a[i] = gaussian(xx, 1.0, order = 2, truncate = l)[l] / dt**2

plt.figure()
plt.plot(t, x, label = 'x')
plt.plot(t, u, label = 'u')
plt.plot(t, a, label = 'a')
plt.legend()
plt.grid()
plt.show()