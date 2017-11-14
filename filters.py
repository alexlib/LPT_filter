import numpy as np
import scipy.interpolate as interp
import numpy.polynomial.polynomial as P
from scipy.ndimage import gaussian_filter1d as gaussian

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
						U[j, :-1] = 0.5 * (X[l+1] - X[l-1]) / self.dt

						A[j, :-1] = (X[l+1] - 2 * X[l] + X[l-1]) / self.dt**2

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
						U[j, 0] = gaussian(x, 1.0, order = 1, truncate = l)[l] / self.dt
						U[j, 1] = gaussian(y, 1.0, order = 1, truncate = l)[l] / self.dt
						U[j, 2] = gaussian(z, 1.0, order = 1, truncate = l)[l] / self.dt

						A[j, 0] = gaussian(x, 1.0, order = 2, truncate = l)[l] / self.dt**2
						A[j, 1] = gaussian(y, 1.0, order = 2, truncate = l)[l] / self.dt**2
						A[j, 2] = gaussian(z, 1.0, order = 2, truncate = l)[l] / self.dt**2

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


