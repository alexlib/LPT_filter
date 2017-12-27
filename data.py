import numpy as np
from scipy.integrate import ode
from scipy.stats import norm
import matplotlib.pyplot as plt

import synthetic_3d as syn

def foo(t, y):
	return [y[3], y[4], y[5], y[6], y[7], y[8], 0, 0, 0]

class Data(object):
	def __init__(self, mode, burger, pdf, sigma_a = 0.1, sigma_z = 0.05):

		self.mode = mode
		self.sigma_a = sigma_a

		self.H = np.zeros((9,9))
		self.H[0, 0] = 1
		self.H[1, 1] = 1
		self.H[2, 2] = 1

		self.h = np.zeros(9)
		self.h[:3] = 1

		self.burger = burger
		self.sigma_z = sigma_z * burger.L0
		self.tau = 2 * (burger.L0 / burger. U0)
		self.pdf = pdf

		self.alpha = 0.9

	def create_sequence_sample(self, steps, dt):

		X0 = np.array([np.random.uniform(self.burger.xmin, self.burger.xmax),
						np.random.uniform(self.burger.xmin, self.burger.xmax),
						np.random.uniform(self.burger.xmin, self.burger.xmax)])

		U0 = np.array([np.random.uniform(-self.burger.U0 / 2, self.burger.U0 / 2),
						np.random.uniform(-self.burger.U0 / 2, self.burger.U0 / 2),
						np.random.uniform(-self.burger.U0 / 2, self.burger.U0 / 2)])

		A0 = np.array([np.random.normal(self.pdf[0, 0], self.pdf[0, 1]),
						np.random.normal(self.pdf[1, 0], self.pdf[1, 1]),
						np.random.normal(self.pdf[2, 0], self.pdf[2, 1])])

		t0 = 0

		s = []
		s.append(np.concatenate((X0, U0, A0)))
		m = []
		m.append(np.concatenate((X0, U0, A0)))

		r = ode(foo).set_integrator('dopri5')
		r.set_initial_value(s[0], t0)

		step = 1
		while r.successful() and step < steps:

			r.integrate(r.t + dt)

			s.append(r.y)

			r.y[-3:] = self.alpha * r.y[-3:] + (1 - self.alpha) * np.array([np.random.normal(self.pdf[0, 0], self.pdf[0, 1]),
						np.random.normal(self.pdf[1, 0], self.pdf[1, 1]),
						np.random.normal(self.pdf[2, 0], self.pdf[2, 1])])

			Z = np.dot(self.H, s[-1]) + np.random.normal(0, self.sigma_z, s[-1].shape) * self.h

			if len(m) > 1:
				Z[3:6] = (Z[:3] - m[-1][:3]) / dt
				Z[6:] = (Z[:3] - 2 * m[-1][:3] + m[-2][:3]) / dt**2

			m.append(Z)
			step += 1

		return np.array(s), np.array(m)

	def checkBoundary(self, X):
		if X[0] > self.burger.xmin and X[0] < self.burger.xmax \
						and X[1] > self.burger.ymin and X[1] < self.burger.ymax \
						and X[2] > self.burger.zmin and X[2] < self.burger.zmax:

			return True

		else:
			return False

	def create_sequence_flow(self, steps, dt):

		X0 = np.array([np.random.uniform(self.burger.xmin, self.burger.xmax),
						np.random.uniform(self.burger.ymin, self.burger.ymax),
						np.random.uniform(self.burger.zmin, self.burger.zmax)])

		U0 = np.array([np.random.uniform(-self.burger.U0 / 2, self.burger.U0 / 2),
						np.random.uniform(-self.burger.U0 / 2, self.burger.U0 / 2),
						np.random.uniform(-self.burger.U0 / 2, self.burger.U0 / 2)])

		U_flow = self.burger.get_Vel(X0[0], X0[1], X0[2])

		A0 = (U_flow - U0) / self.tau

		t0 = 0

		s = []
		s.append(np.concatenate((X0, U0, A0)))
		m = []
		m.append(np.concatenate((X0, U0, A0)))

		r = ode(foo).set_integrator('dopri5')
		r.set_initial_value(s[0], t0)

		step = 1
		while r.successful() and step < steps and self.checkBoundary(s[-1][:3]):

			r.integrate(r.t + dt)

			s.append(r.y)

			X = r.y[:3]
			U = r.y[3:6]
			A = r.y[-3:]

			U_flow = self.burger.get_Vel(X[0], X[1], X[2])

			r.y[-3:] = (U_flow - U) / self.tau

			Z = np.dot(self.H, s[-1]) + np.random.normal(0, self.sigma_z, s[-1].shape) * self.h

			if len(m) > 1:
				Z[3:6] = (Z[:3] - m[-1][:3]) / dt
				Z[6:] = (Z[:3] - 2 * m[-1][:3] + m[-2][:3]) / dt**2

			m.append(Z)
			step += 1

		return np.array(s), np.array(m)


	def create_dataset(self, samples, steps, dt):

		data = np.empty((0, 9))
		label = np.empty((0, 9))
		while data.shape[0] < samples * steps:
			if self.mode == "flow":
				X, Z = self.create_sequence_flow(steps, dt)

			elif self.mode == "sample":
				X, Z = self.create_sequence_sample(steps, dt)

			else:
				Raise(Error("Wrong mode"))

			if X.shape[0] < steps:
				continue

			data = np.vstack((data, Z))
			label = np.vstack((label, X))

		# data = np.reshape(data, (self.N_sample, self.steps, -1))
		# label = np.reshape(label, (self.N_sample, self.steps, -1))

		return data, label

def generate_data(mode, samples, steps):

	burger, dt = generate_flow()
	if mode == "flow":
		data = Data(mode, burger, None)
	elif mode == "sample":
		Y_ = np.load("Y.npy")
		pdf = np.zeros((3, 2))

		for i in range(len(pdf)):
			pdf[i,:] = np.array(norm.fit(Y_[:, i - 3]))

		data = Data(mode, burger, pdf)

	else:
		raise(Error("Wrong mode"))

	X, Y = data.create_dataset(samples, steps, dt)

	return X, Y

def generate_flow():
	gamma = 100
	sigma = 0.01
	nu = 1

	L0 = np.sqrt(2 * nu / sigma)
	U0 = gamma / L0

	xmin, xmax = -10 * L0, 10 * L0
	ymin, ymax = -10 * L0, 10 * L0
	zmin, zmax = 0, 20 * L0

	Nx = Ny = Nz = 11

	N_burger = 1
	burger = syn.Burger(N_burger, Nx, Ny, Nz, xmin, xmax, ymin, ymax, zmin, zmax,
					gamma = gamma, sigma = sigma, nu = nu)

	dt = 0.1 * L0 / U0

	return burger, dt

def main():
	burger, dt = generate_flow()

	data = Data("flow", burger, None)


	samples = 1
	steps = 500

	X, Y = data.create_dataset(samples, steps, dt)


	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot(X[:steps,0] / L0,
				X[:steps,1] / L0,
				X[:steps,2] / L0)
	ax.set_xlim( (xmin / L0, xmax / L0) )
	ax.set_ylim( (ymin / L0, ymax / L0) )
	ax.set_zlim( (zmin / L0, zmax / L0) )
	plt.legend()

	plt.show()

if __name__ == "__main__":
	main()