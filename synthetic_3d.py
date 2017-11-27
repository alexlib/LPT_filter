import os
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.integrate as integrate
from mpl_toolkits.mplot3d import axes3d
from scipy.ndimage import gaussian_filter1d as gaussian
from numpy import random as np_random
from numpy.linalg import norm as norm
import json
import pickle

import filters

class Burger(object):
	def __init__(self, N_burger, Nx, Ny, Nz,
					xmin, xmax, ymin, ymax, zmin, zmax,
					gamma = 200, sigma = 0.01, nu = 1):

		self.xmin = xmin
		self.xmax = xmax

		self.ymin = ymin
		self.ymax = ymax

		self.zmin = zmin
		self.zmax = zmax

		x, dx = np.linspace(xmin, xmax, num = Nx, endpoint = True, retstep = True)
		y, dy = np.linspace(ymin, ymax, num = Ny, endpoint = True, retstep = True)
		z, dz = np.linspace(zmin, zmax, num = Nz, endpoint = True, retstep = True)

		self.X, self.Y, self.Z = np.meshgrid(x, y, z)

		self.centroids = []

		for _ in range(N_burger):
			x0 = random.uniform(xmin, xmax)
			y0 = random.uniform(ymin, ymax)
			sign = random.choice([-1, 1])

			self.centroids.append((x0, y0, sign))

		self.gamma = gamma
		self.sigma = sigma
		self.nu = nu

	def one_vortex(self, x0, y0, sign, x = None, y = None):
		if x is None:
			x = self.X

		if y is None:
			y = self.Y

		R = np.sqrt((x - x0)**2 + (y - y0)**2)
		U = self.gamma * (1 - np.exp(-self.sigma * R**2 / 4 / self.nu)) / (2 * np.pi * R)

		dx = x - x0
		dy = y - y0

		u = sign * (- dy / R * U) - 0.5 * self.sigma * dx
		v = sign * (dx / R * U) - 0.5 * self.sigma * dy

		return u, v

	def flow_field(self):
		self.vx = np.zeros(self.X.shape)
		self.vy = np.zeros(self.Y.shape)
		self.vz = np.zeros(self.Z.shape)


		for c in self.centroids:

			new_vx, new_vy = self.one_vortex(c[0], c[1], c[2])
		
			self.vx += new_vx
			self.vy += new_vy

		self.vz = self.sigma * self.Z

	def get_Vel(self, x, y, z):
		ux = 0
		uy = 0

		for c in self.centroids:
			x0 = c[0]
			y0 = c[1]
			sign = c[2]

			new_vx, new_vy = self.one_vortex(x0, y0, sign, x, y)

			ux += new_vx
			uy += new_vy

		uz = self.sigma * z

		return ux, uy, uz

	def get_Grad(self, x, y, axis):
		du_dx = 0
		du_dy = 0

		for c in self.centroids:
			x0 = c[0]
			y0 = c[1]
			sign = c[2]

			f = (x - x0)**2 + (y - y0)**2

			exp = np.exp(-self.sigma * f / (4 * self.nu))

			if axis == 'X':
				du_dx += sign * (x - x0) * (y - y0) * self.gamma * ((1 - exp) / (np.pi * f**2) - 
							self.sigma * exp / (4 * self.nu) / (np.pi * f) ) - self.sigma / 2

				du_dy += sign * self.gamma * (- (1 - exp) / (2 * np.pi * f) +
							(y - y0)**2 * ((1 - exp) / (np.pi * f**2) - 
							self.sigma * exp / (4 * self.nu) / (np.pi * f) ) )

				du_dz = 0

			elif axis == 'Y':
				du_dx += sign * self.gamma * ((1 - exp) / (2 * np.pi * f) -
							(x - x0)**2 * ((1 - exp) / (np.pi * f**2) - 
							self.sigma * exp / (4 * self.nu) / (np.pi * f) ) )

				du_dy += sign * (x - x0) * (y - y0) * self.gamma * (-(1 - exp) / (np.pi * f**2) + 
							self.sigma * exp / (4 * self.nu) / (np.pi * f) ) - self.sigma / 2

				du_dz = 0

			elif axis == 'Z':
				du_dx = 0
				du_dy = 0
				du_dz = self.sigma

			else:
				raise(Error("Invaid axis"))

		return du_dx, du_dy, du_dz

	def checkBoundary(self, streamline, t):
		for i in range(len(t)):
			x = streamline[i][0]
			y = streamline[i][1]
			z = streamline[i][2]

			if x > self.xmin and x < self.xmax \
				and y > self.ymin and y < self.ymax \
				and z > self.zmin and z < self.zmax:

				continue

			else:
				break

		return streamline[:i,:], t[:i]

	def trajectory(self, N_traj, dt, t_end):
		grid_arr = np.array(zip(self.X.flatten(), self.Y.flatten(), self.Z.flatten()))
		dfun = lambda p,t: [interp.griddata(grid_arr, self.vx.flatten(), np.array([p]))[0], 
							interp.griddata(grid_arr, self.vy.flatten(), np.array([p]))[0],
							interp.griddata(grid_arr, self.vz.flatten(), np.array([p]))[0]]


		t = np.arange(0,t_end+dt,dt)
		self.streamline = []
		self.T = []
		self.U = []
		self.A = []

		for i in range(N_traj):
			p0 = (random.uniform(self.xmin, self.xmax), 
					random.uniform(self.ymin, self.ymax), 
					random.uniform(self.zmin, self.zmax))

			streamline = integrate.odeint(dfun, p0, t)
			streamline, temp = self.checkBoundary(streamline, t)

			self.streamline.append(streamline)
			self.T.append(temp)
			self.U.append(np.empty((len(self.T[i]), 4)))
			self.A.append(np.empty((len(self.T[i]), 4)))

			for j in range(len(self.T[i])):
				x = self.streamline[i][j,0]
				y = self.streamline[i][j,1]
				z = self.streamline[i][j,2]

				ux, uy, uz = self.get_Vel(x, y, z)

				self.U[i][j,0] = ux
				self.U[i][j,1] = uy
				self.U[i][j,2] = uz

				du_dx, du_dy, du_dz = self.get_Grad(x, y, 'X')

				self.A[i][j,0] = ux * du_dx + uy * du_dy + uz * du_dz

				du_dx, du_dy, du_dz = self.get_Grad(x, y, 'Y')

				self.A[i][j,1] = ux * du_dx + uy * du_dy + uz * du_dz

				du_dx, du_dy, du_dz = self.get_Grad(x, y, 'Z')

				self.A[i][j,2] = ux * du_dx + uy * du_dy + uz * du_dz

			self.U[i][:,-1] = np.sqrt(self.U[i][:,0]**2 + self.U[i][:,1]**2 + self.U[i][:,2]**2)
			self.A[i][:,-1] = np.sqrt(self.A[i][:,0]**2 + self.A[i][:,1]**2 + self.A[i][:,2]**2)

		# return self.streamline, self.U, self.A, self.T


	def distort(self, skip, delta):
		streamline = []
		U = []
		A = []
		T = []

		for i in range(len(self.T)):
			streamline.append(self.streamline[i][::skip])
			streamline[i] += delta * np_random.rand(streamline[i].shape[0],
														streamline[i].shape[1])
			U.append(self.U[i][::skip])
			A.append(self.A[i][::skip])
			T.append(self.T[i][::skip])

		return streamline, U, A, T


	def make_plot(self, U_hat, A_hat, T_hat):
		plt.figure()
		for i in range(len(T_hat)):
			for n in U_hat.keys():
				plt.plot(T_hat[i], U_hat[n][i][:,-1], '*', label = '{}_{}'.format(n, i))
			plt.plot(self.T[i], self.U[i][:,-1], label = 'GT {}'.format(i))
		plt.legend()
		plt.grid()

		plt.figure()
		for i in range(len(T_hat)):
			for n in A_hat.keys():
				plt.plot(T_hat[i], A_hat[n][i][:,-1], '*', label = '{}_{}'.format(n, i))
			plt.plot(self.T[i], self.A[i][:,-1], label = 'GT {}'.format(i))
		plt.legend()
		plt.grid()


		fig = plt.figure()
		ax = fig.gca(projection='3d')
		for i in range(len(self.T)):
			ax.plot(self.streamline[i][:,0], self.streamline[i][:,1], self.streamline[i][:,2],
					label = 'Traj {}'.format(i))
		plt.legend()

		fig = plt.figure()
		ax = fig.gca(projection='3d')
		ax.quiver(self.X, self.Y, self.Z, self.vx, self.vy, self.vz, length = 10.0)

def Calc_error(U, U_hat, A, A_hat, T, l):
	err_U = {}
	err_A = {}

	# plt.figure()

	for n in U_hat.keys():
		err_U[n] = []
		err_A[n] = []

		for i in range(len(U)):
			if len(U[i]) < 2 * l + 1:
				continue
			print ("Trajectory {} / {}".format(i, len(U)))
			# err_U[n].append(norm(U[i][:,-1] - U_hat[n][i][:,-1] , ord = 1))
			# err_A[n].append(norm(A[i][:,-1] - A_hat[n][i][:,-1] , ord = 1))

			err_U[n].append( np.mean(np.abs(U[i][l:-l,-1] - U_hat[n][i][l:-l,-1]) / U[i][l:-l,-1]) )
			err_A[n].append( np.mean(np.abs(A[i][l:-l,-1] - A_hat[n][i][l:-l,-1]) / A[i][l:-l,-1]) )

			# plt.plot(T[i][l:-l], U_hat[n][i][l:-l,-1], label = "{}_{}".format(n, i))

			# plt.plot(T[i][l:-l], U[i][l:-l,-1], label = "GT_{}".format(i))


		err_U[n] = np.mean(np.array(err_U[n]))
		err_A[n] = np.mean(np.array(err_A[n]))


		print ("Velocity {}: {}".format(n, err_U[n]))
		print ("Acceleration {}: {}".format(n, err_A[n]))

	# plt.legend()

	return err_U, err_A


def filter(names, streamline, T, l, dt):

	f = filters.filters(names, streamline, T, dt, l)

	return f.process()

def main():
	xmin, xmax = -50, 50
	ymin, ymax = -50, 50
	zmin, zmax = 0, 100

	Nx = Ny = Nz = 11

	N_burger = 1
	burger = Burger(N_burger, Nx, Ny, Nz, xmin, xmax, ymin, ymax, zmin, zmax, gamma = 100)
	burger.flow_field()


	dt = 0.1
	N_t = 100
	t_end = N_t * dt
	N_traj = 2
	burger.trajectory(N_traj, dt, t_end)

	skip = np.linspace(2, 5, 4).astype(int)
	N_sample = np.linspace(3, 5, 3).astype(int)
	delta = np.linspace(1e-3, 1e-1, 1) 


	names = ["Spline", "Finite difference", "Polynomial", "Gaussian"]

	err_U = {}
	err_A = {}

	err_U["skip"] = skip
	err_U["delta"] = delta
	err_U["N_sample"] = N_sample

	for n in names:
		err_U[n] = np.zeros((len(skip), len(N_sample), len(delta)))
		err_A[n] = np.zeros_like(err_U[n])

	for i in range(len(skip)):
		for j in range(len(N_sample)):
			for k in range(len(delta)):

				streamline, U, A, T = burger.distort(skip[i], delta[k])

				dt_ = dt * skip[i]

				U_hat, A_hat = filter(names, streamline, T, N_sample[j], dt_)

				err_U_, err_A_ = Calc_error(U, U_hat, A, A_hat, T, N_sample[j])

				for n in names:
					err_U[n][i,j,k] = err_U_[n]
					err_A[n][i,j,k] = err_A_[n]


	with open("err_U", 'w') as f:
		pickle.dump(err_U, f)

	with open("err_A", 'w') as f:
		pickle.dump(err_A, f)

	burger.make_plot(U_hat, A_hat, T)
	plt.show()

	print err_A["Gaussian"][:,:,0]

if __name__ == "__main__":
	main()