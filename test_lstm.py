import synthetic_3d as syn
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import CuDNNLSTM, CuDNNGRU, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.integrate import ode
from scipy.stats import norm

import data
import filters


def foo(t, y):
	return [y[3], y[4], y[5], y[6], y[7], y[8], 0, 0, 0]

class Data(object):
	def __init__(self, burger, samples, steps, dt, sigma_z = 0.05):

		self.burger = burger
		self.samples = samples
		self.steps = steps
		self.sigma_z = sigma_z
		self.dt = dt
		self.tau = 0.5

		self.H = np.zeros((9,9))
		self.H[0, 0] = 1
		self.H[1, 1] = 1
		self.H[2, 2] = 1

		self.h = np.zeros(9)
		self.h[:3] = 1

	def checkBoundary(self, X):
		if X[0] > self.burger.xmin and X[0] < self.burger.xmax \
						and X[1] > self.burger.ymin and X[1] < self.burger.ymax \
						and X[2] > self.burger.zmin and X[2] < self.burger.zmax:

			return True

		else:
			return False

	def create_sequence(self):

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
		while r.successful() and step < self.steps and self.checkBoundary(s[-1][:3]):

			r.integrate(r.t + self.dt)

			s.append(r.y)

			X = r.y[:3]
			U = r.y[3:6]
			A = r.y[-3:]

			U_flow = self.burger.get_Vel(X[0], X[1], X[2])

			r.y[-3:] = (U_flow - U) / self.tau
			# r.y[-3:] += np.array([np.random.choice([-1, 1]) * np.random.exponential(0.1),
			# 				np.random.choice([-1, 1]) * np.random.exponential(0.1),
			# 				np.random.choice([-1, 1]) * np.random.exponential(0.1)])

			Z = np.dot(self.H, s[-1]) + np.random.normal(0, self.sigma_z, s[-1].shape) * self.h

			if len(m) > 1:
				Z[3:6] = (Z[:3] - m[-1][:3]) / self.dt
				Z[6:] = (Z[:3] - 2 * m[-1][:3] + m[-2][:3]) / self.dt**2

			m.append(Z)
			step += 1

		return np.array(s), np.array(m)

	def create_dataset(self):

		data = np.empty((0, 9))
		label = np.empty((0, 9))
		while data.shape[0] < self.steps * self.samples:
			X, Z = self.create_sequence()

			if X.shape[0] < self.steps:
				continue

			data = np.vstack((data, Z))
			label = np.vstack((label, X))

		# data = np.reshape(data, (self.samples, self.steps, -1))
		# label = np.reshape(label, (self.samples, self.steps, -1))

		return data, label

def main():

	X = np.load("X.npy")
	scaler_X = MinMaxScaler()
	scaler_X.fit(X)

	Y = np.load("Y_.npy")
	scaler_Y = MinMaxScaler()
	scaler_Y.fit(Y)

	# data = Data(burger, samples, N_t, dt, sigma_z = 0.1)
	# X, Y_ = data.create_dataset()

	burger, dt = data.generate_flow()

	samples = 1
	steps = 500

	X, Y_ = data.generate_data("flow", 1, 500)

	# np.save("X", X)
	# np.save("Y_", Y_)

	# return


	# np.save("new_Traj.npy", Y_)

	# Y_ = np.load("Traj.npy")
	# mu, std = norm.fit(Y_[:, -1])

	# print mu, std

	# plt.figure()
	# plt.hist(Y_[:, -3:], bins = 25, density = True)
	# xmin, xmax = plt.xlim()
	# x = np.linspace(xmin, xmax, 100)
	# p = norm.pdf(x, mu, std)
	# plt.plot(x, p, 'k')
	# plt.legend()
	# plt.show()

	streamline = []
	streamline.append(X[:,:3])
	T = []
	t = np.arange(0, dt * len(X), dt)
	T.append(t)

	names = ['Polynomial']
	f = filters.filters(names, streamline, T, dt, 5)

	U, A = f.process()

	axis = 1

	X = scaler_X.transform(X)

	model = load_model("my_model")

	Y = model.predict(X.reshape((samples, -1, 9)), batch_size = samples)
	Y = np.squeeze(Y)
	Y = scaler_Y.inverse_transform(Y)

	# X = scaler.inverse_transform(X)

	plt.figure()
	plt.plot(Y_[:, axis], label = "GT")
	plt.plot(Y[:, axis], label = "RNN")
	plt.legend()

	plt.figure()
	plt.plot(Y_[:, axis + 3], label = "GT")
	plt.plot(Y[:, axis + 3], label = "RNN")
	plt.plot(U['Polynomial'][0][:, axis], label = "Polynomial")
	plt.legend()

	plt.figure()
	plt.plot(Y_[:, axis + 3], label = "GT")
	plt.plot(Y[:, axis + 6], label = "RNN")
	plt.plot(A['Polynomial'][0][:, axis], label = "Polynomial")
	plt.legend()

	fig = plt.figure()
	ax = fig.gca(projection='3d')

	for i in range(samples):
		ax.plot(Y_[:,0] / burger.L0,
				Y_[:,1] / burger.L0,
				Y_[:,2] / burger.L0,
				label = 'Traj {}'.format(i))
	ax.set_xlim( (burger.xmin / burger.L0, burger.xmax / burger.L0) )
	ax.set_ylim( (burger.ymin / burger.L0, burger.ymax / burger.L0) )
	ax.set_zlim( (burger.zmin / burger.L0, burger.zmax / burger.L0) )
	plt.legend()

	plt.show()


if __name__ == "__main__":
	main()