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

def dist(x, y, x0, y0):
	return np.sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0))

def decompose(x, y, x0, y0, vel, R, sign):
	dx = x - x0
	dy = y - y0
	u = sign * (- dy / R * vel) - 0.5 * 0.01 * dx
	v = sign * (dx / R * vel) - 0.5 * 0.01 * dy
	return u, v

def velocity(d, gamma = 100, sigma = 0.01, nu = 1):
	return gamma * (1 - np.exp(-sigma * d**2 / 4 / nu)) / (2 * np.pi * d)

def Burger(X, Y, x0, y0, sign):
	R = np.sqrt((X - x0) * (X - x0) + (Y - y0) * (Y - y0))
	U = velocity(R)
	vx, vy = decompose(X, Y, x0, y0, U, R, sign)

	return vx, vy

def Vel(x, y, z, centroids, gamma = 100, sigma = 0.01, nu = 1):
	ux = 0
	uy = 0

	for i in range(len(centroids)):
		x0 = centroids[i][0]
		y0 = centroids[i][1]
		sign = centroids[i][2]

		new_vx, new_vy = Burger(x, y, x0, y0, sign)

		ux += new_vx
		uy += new_vy

	uz = sigma * z

	return ux, uy, uz


def Grad(x, y, centroids, axis, gamma = 100, sigma = 0.01, nu = 1):
	du_dx = 0
	du_dy = 0

	for i in range(len(centroids)):
		x0 = centroids[i][0]
		y0 = centroids[i][1]
		sign = centroids[i][2]

		f = (x - x0)**2 + (y - y0)**2

		exp = np.exp(-sigma * f / (4 * nu))

		if axis == 'X':
			du_dx += sign * (x - x0) * (y - y0) * gamma * ((1 - exp) / (np.pi * f**2) - 
						sigma * exp / (4 * nu) / (np.pi * f) ) - sigma / 2

			du_dy += sign * gamma * (- (1 - exp) / (2 * np.pi * f) +
						(y - y0)**2 * ((1 - exp) / (np.pi * f**2) - 
						sigma * exp / (4 * nu) / (np.pi * f) ) )

			du_dz = 0

		elif axis == 'Y':
			du_dx += sign * gamma * ((1 - exp) / (2 * np.pi * f) -
						(x - x0)**2 * ((1 - exp) / (np.pi * f**2) - 
						sigma * exp / (4 * nu) / (np.pi * f) ) )

			du_dy += sign * (x - x0) * (y - y0) * gamma * (-(1 - exp) / (np.pi * f**2) + 
						sigma * exp / (4 * nu) / (np.pi * f) ) - sigma / 2

			du_dz = 0

		elif axis == 'Z':
			du_dx = 0
			du_dy = 0
			du_dz = sigma

		else:
			raise(Error("Invaid axis"))

	return du_dx, du_dy, du_dz

def checkBoundary(streamline, t, xmin = -50, xmax = 50, ymin = -50, ymax = 50,
					zmin = 0, zmax = 100):
	for i in range(len(t)):
		x = streamline[i][0]
		y = streamline[i][1]
		z = streamline[i][2]

		if x > xmin and x < xmax and y > ymin and y < ymax and z > zmin and z < zmax:
			continue

		else:
			break

	return streamline[:i,:], t[:i]


def flow_field(N, Nx, Ny, Nz, xmin, xmax, ymin, ymax, zmin, zmax):
	x, dx = np.linspace(xmin, xmax, num = Nx, endpoint = True, retstep = True)
	y, dy = np.linspace(ymin, ymax, num = Ny, endpoint = True, retstep = True)
	z, dz = np.linspace(zmin, zmax, num = Nz, endpoint = True, retstep = True)

	X, Y, Z = np.meshgrid(x, y, z)

	vx = np.zeros(X.shape)
	vy = np.zeros(Y.shape)
	vz = np.zeros(Z.shape)

	centroids = []
	for _ in range(N):
		x0 = random.uniform(-50, 50)
		y0 = random.uniform(-50, 50)
		sign = random.choice([-1, 1])

		new_vx, new_vy = Burger(X, Y, x0, y0, sign)
		
		vx += new_vx
		vy += new_vy

		centroids.append((x0, y0, sign))

	vz = 0.01 * Z

	return X, Y, Z, vx, vy, vz, centroids

def trajectory(N, X, Y, Z, vx, vy, vz, centroids, dt, t1):
	grid_arr = np.array(zip(X.flatten(), Y.flatten(), Z.flatten()))
	dfun = lambda p,t: [interp.griddata(grid_arr, vx.flatten(), np.array([p]))[0], 
						interp.griddata(grid_arr, vy.flatten(), np.array([p]))[0],
						interp.griddata(grid_arr, vz.flatten(), np.array([p]))[0]]


	t = np.arange(0,t1+dt,dt)
	streamline = []
	T = []
	U = []
	A = []
	# A_cs = []

	# css = []

	for i in range(N):
		p0 = (random.uniform(-50, 50), random.uniform(-50, 50), random.uniform(0, 100))
		streamline.append(integrate.odeint(dfun,p0,t))
		streamline[i], temp = checkBoundary(streamline[i], t)

		T.append(temp)
		U.append(np.empty((len(T[i]), 4)))
		A.append(np.empty((len(T[i]), 4)))

		for j in range(len(T[i])):
			x = streamline[i][j,0]
			y = streamline[i][j,1]
			z = streamline[i][j,2]

			ux, uy, uz = Vel(x, y, z, centroids)

			U[i][j,0] = ux
			U[i][j,1] = uy
			U[i][j,2] = uz

			du_dx, du_dy, du_dz = Grad(x, y, centroids, 'X')

			A[i][j,0] = ux * du_dx + uy * du_dy + uz * du_dz

			du_dx, du_dy, du_dz = Grad(x, y, centroids, 'Y')

			A[i][j,1] = ux * du_dx + uy * du_dy + uz * du_dz

			du_dx, du_dy, du_dz = Grad(x, y, centroids, 'Z')

			A[i][j,2] = ux * du_dx + uy * du_dy + uz * du_dz

		U[i][:,-1] = np.sqrt(U[i][:,0]**2 + U[i][:,1]**2 + U[i][:,2]**2)
		A[i][:,-1] = np.sqrt(A[i][:,0]**2 + A[i][:,1]**2 + A[i][:,2]**2)


		# csx = interp.CubicSpline(T[i], streamline[i][:,0])
		# csy = interp.CubicSpline(T[i], streamline[i][:,1])
		# css.append((csx, csy))

		# ax = csx(T[i], 2)
		# ay = csy(T[i], 2)
		# A_cs.append(np.sqrt(ax**2 + ay**2))

	return streamline, U, A, T

def distort(streamline, U, A, T, skip, delta):
	new_streamline = []
	new_U = []
	new_A = []
	new_T = []

	for i in range(len(T)):
		new_streamline.append(streamline[i][::skip])
		new_streamline[i] += delta * np_random.rand(new_streamline[i].shape[0],
													new_streamline[i].shape[1])
		new_U.append(U[i][::skip])
		new_A.append(A[i][::skip])
		new_T.append(T[i][::skip])

	return new_streamline, new_U, new_A, new_T

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

def make_plot(X, Y, Z, vx, vy, vz, streamline, U, U_hat, A, A_hat, T, T_hat):
	plt.figure(1)
	for i in range(len(T)):
		for n in U_hat.keys():
			plt.plot(T_hat[i], U_hat[n][i][:,-1], '*', label = '{}_{}'.format(n, i))
		plt.plot(T[i], U[i][:,-1], label = 'GT {}'.format(i))
		# plt.plot(T[i], U[i], label = 'U {}'.format(i))
	plt.legend()
	plt.grid()

	plt.figure(2)
	for i in range(len(T)):
		for n in A_hat.keys():
			plt.plot(T_hat[i], A_hat[n][i][:,-1], '*', label = '{}_{}'.format(n, i))
		plt.plot(T[i], A[i][:,-1], label = 'GT {}'.format(i))
		# plt.plot(T[i], U[i], label = 'U {}'.format(i))
	plt.legend()
	plt.grid()


	fig = plt.figure(3)
	# plt.streamplot(X, Y, vx, vy)
	ax = fig.gca(projection='3d')
	for i in range(len(T)):
		ax.plot(streamline[i][:,0], streamline[i][:,1], streamline[i][:,2],
				label = 'Traj {}'.format(i))
	plt.legend()

	fig = plt.figure(4)
	ax = fig.gca(projection='3d')
	ax.quiver(X, Y, Z, vx, vy, vz, length = 10.0)
	plt.show()

def main():
	xmin, xmax = -50, 50
	ymin, ymax = -50, 50
	zmin, zmax = 0, 100

	Nx = Ny = Nz = 11

	N_burger = 1
	X, Y, Z, vx, vy, vz, centroids = flow_field(N_burger, Nx, Ny, Nz, 
											xmin, xmax, ymin, ymax, zmin, zmax)


	dt = 0.1
	t_end = 10
	N_traj = 5
	streamline, U, A, T = trajectory(N_traj, X, Y, Z, vx, vy, vz, centroids, dt, t_end)

	skip = np.linspace(1, 10, 5).astype(int)
	delta = np.linspace(1e-3, 1e-1, 1) 
	N_sample = np.linspace(3, 5, 1)


	names = ["Spline", "Finite difference", "Polynomial", "Gaussian"]

	err_U = {}
	err_A = {}

	err_U["skip"] = skip
	err_U["delta"] = delta
	err_U["N_sample"] = N_sample

	for n in names:
		err_U[n] = np.zeros((len(skip), len(delta), len(N_sample)))
		err_A[n] = np.zeros_like(err_U[n])

	for i in range(len(skip)):
		for j in range(len(delta)):
			for k in range(len(N_sample)):

				streamline_new, U_new, A_new, T_new = distort(streamline, U, A, T,
																skip[i], delta[j])

				dt_ = dt * skip[i]
				l = int(N_sample[k] / dt_ + 0.5)

				U_hat, A_hat = filter(names, streamline_new, T_new, l, dt_)

				err_U_, err_A_ = Calc_error(U_new, U_hat, A_new, A_hat, T_new, l)

				for n in names:
					err_U[n][i,j,k] = err_U_[n]
					err_A[n][i,j,k] = err_A_[n]


	with open("err_U", 'w') as f:
		pickle.dump(err_U, f)

	with open("err_A", 'w') as f:
		pickle.dump(err_A, f)
	make_plot(X, Y, Z, vx, vy, vz, streamline, U, U_hat, A, A_hat, T, T_new)
	plt.show()

if __name__ == "__main__":
	main()