import os
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.integrate as integrate

def dist(x, y, x0, y0):
	return np.sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0))

def decompose(x, y, x0, y0, vel, R, sign):
	dx = x - x0
	dy = y - y0
	u = sign * (- dy / R * vel) #- 0.5 * 0.01 * dx
	v = sign * (dx / R * vel) #- 0.5 * 0.01 * dy
	return u, v

def velocity(d, gamma = 50, sigma = 0.01, nu = 1):
	return gamma * (1 - np.exp(-sigma * d**2 / 4 / nu)) / (2 * np.pi * d)

def Burger(X, Y, x0, y0, sign):
	R = np.sqrt((X - x0) * (X - x0) + (Y - y0) * (Y - y0))
	U = velocity(R)
	vx, vy = decompose(X, Y, x0, y0, U, R, sign)

	return vx, vy

def Vel(x, y, centroids, gamma = 50, sigma = 0.01, nu = 1):
	ux = 0
	uy = 0

	for i in range(len(centroids)):
		x0 = centroids[i][0]
		y0 = centroids[i][1]
		sign = centroids[i][2]

		new_vx, new_vy = Burger(x, y, x0, y0, sign)

		ux += new_vx
		uy += new_vy

	return ux, uy


def Grad(x, y, centroids, axis, gamma = 50, sigma = 0.01, nu = 1):
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
						sigma * exp / (4 * nu) / (np.pi * f) ) #- sigma / 2

			du_dy += sign * gamma * (- (1 - exp) / (2 * np.pi * f) +
						(y - y0)**2 * ((1 - exp) / (np.pi * f**2) - 
						sigma * exp / (4 * nu) / (np.pi * f) ) )

		elif axis == 'Y':
			du_dx += sign * gamma * ((1 - exp) / (2 * np.pi * f) -
						(x - x0)**2 * ((1 - exp) / (np.pi * f**2) - 
						sigma * exp / (4 * nu) / (np.pi * f) ) )

			du_dy += sign * (x - x0) * (y - y0) * gamma * (-(1 - exp) / (np.pi * f**2) + 
						sigma * exp / (4 * nu) / (np.pi * f) ) #- sigma / 2

		else:
			continue

	return du_dx, du_dy

def checkBoundary(streamline, t, xmin = -50, xmax = 50, ymin = -50, ymax = 50):
	for i in range(len(t)):
		x = streamline[i][0]
		y = streamline[i][1]

		if x > xmin and x < xmax and y > ymin and y < ymax:
			continue

		else:
			break

	return streamline[:i,:], t[:i]

Nx = 21
Ny = 21

x, dx = np.linspace(-50, 50, num = Nx, endpoint = True, retstep = True)
y, dy = np.linspace(-50, 50, num = Ny, endpoint = True, retstep = True)
X, Y = np.meshgrid(x, y)

vx = np.zeros(X.shape)
vy = np.zeros(Y.shape)

N = 3
centroids = []
for _ in range(N):
	x0 = random.uniform(-50, 50)
	y0 = random.uniform(-50, 50)
	sign = random.choice([-1, 1])

	new_vx, new_vy = Burger(X, Y, x0, y0, sign)
	
	vx += new_vx
	vy += new_vy

	centroids.append((x0, y0, sign))

xyarr = np.array(zip(X.flatten(),Y.flatten()))
dfun = lambda p,t: [interp.griddata(xyarr, vx.flatten(), np.array([p]), 'cubic')[0], 
					interp.griddata(xyarr, vy.flatten(), np.array([p]), 'cubic')[0]]

N = 3
dt = 0.01
t0 = 0
t1 = 100
t = np.arange(t0,t1+dt,dt)
streamline = []
T = []
U = []
acc = []
A = []
A_cs = []

css = []

for i in range(N):
	p0 = (random.uniform(-50, 50), random.uniform(-50, 50))
	streamline.append(integrate.odeint(dfun,p0,t))
	streamline[i], temp = checkBoundary(streamline[i], t)
	T.append(temp)

	acc.append(np.empty((len(T[i]),2)))
	temp_u = []

	for j in range(len(T[i])):
		x = streamline[i][j,0]
		y = streamline[i][j,1]

		ux, uy = Vel(x, y, centroids)
		temp_u.append(ux)

		du_dx, du_dy = Grad(x, y, centroids, 'X')

		acc[i][j,0] = ux * du_dx + uy * du_dy

		du_dx, du_dy = Grad(x, y, centroids, 'Y')

		acc[i][j,1] = ux * du_dx + uy * du_dy

	U.append(np.array(temp_u))
	A.append(np.sqrt(acc[i][:,0]**2 + acc[i][:,1]**2))


	csx = interp.CubicSpline(T[i], streamline[i][:,0])
	csy = interp.CubicSpline(T[i], streamline[i][:,1])
	css.append((csx, csy))

	ax = csx(T[i], 2)
	ay = csy(T[i], 2)
	A_cs.append(np.sqrt(ax**2 + ay**2))


plt.figure(1)
for i in range(N):
	plt.plot(T[i], A_cs[i], '*', label = 'Inter {}'.format(i))
	plt.plot(T[i], A[i], label = 'GT {}'.format(i))
	# plt.plot(T[i], U[i], label = 'U {}'.format(i))
plt.legend()
plt.grid()


plt.figure(2)
plt.streamplot(X, Y, vx, vy)
for i in range(N):
	plt.plot(streamline[i][:,0], streamline[i][:,1], label = 'Traj {}'.format(i))
plt.legend()
plt.show()