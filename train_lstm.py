import numpy as np
import matplotlib.pyplot as plt
import math
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import CuDNNLSTM, CuDNNGRU, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy.polynomial.polynomial as P
from scipy.integrate import ode
from scipy.stats import norm
import argparse
import pickle

import data
import synthetic_3d as syn

# def foo(t, y):
# 	return [y[3], y[4], y[5], y[6], y[7], y[8], 0, 0, 0]

# class Data(object):
# 	def __init__(self, burger, pdf, N_samples, steps, dt, sigma_a = 0.1, sigma_z = 0.05):

# 		self.N_samples = N_samples
# 		self.steps = steps
# 		self.sigma_a = sigma_a
# 		self.sigma_z = sigma_z
# 		self.dt = dt

# 		self.H = np.zeros((9,9))
# 		self.H[0, 0] = 1
# 		self.H[1, 1] = 1
# 		self.H[2, 2] = 1

# 		self.h = np.zeros(9)
# 		self.h[:3] = 1

# 		self.burger = burger
# 		self.pdf = pdf

# 		self.alpha = 0.9

# 	def create_sequence(self):

# 		X0 = np.array([np.random.uniform(self.burger.xmin, self.burger.xmax),
# 						np.random.uniform(self.burger.xmin, self.burger.xmax),
# 						np.random.uniform(self.burger.xmin, self.burger.xmax)])

# 		U0 = np.array([np.random.uniform(-self.burger.U0 / 2, self.burger.U0 / 2),
# 						np.random.uniform(-self.burger.U0 / 2, self.burger.U0 / 2),
# 						np.random.uniform(-self.burger.U0 / 2, self.burger.U0 / 2)])

# 		A0 = np.array([np.random.normal(pdf[0, 0], pdf[0, 1]),
# 						np.random.normal(pdf[1, 0], pdf[1, 1]),
# 						np.random.normal(pdf[2, 0], pdf[2, 1])])

# 		t0 = 0

# 		s = []
# 		s.append(np.concatenate((X0, U0, A0)))
# 		m = []
# 		m.append(np.concatenate((X0, U0, A0)))

# 		r = ode(foo).set_integrator('dopri5')
# 		r.set_initial_value(s[0], t0)

# 		step = 1
# 		while r.successful() and step < self.steps:

# 			r.integrate(r.t + self.dt)

# 			s.append(r.y)

# 			r.y[-3:] = np.array([np.random.normal(pdf[0, 0], pdf[0, 1]),
# 						np.random.normal(pdf[1, 0], pdf[1, 1]),
# 						np.random.normal(pdf[2, 0], pdf[2, 1])])

# 			Z = np.dot(self.H, s[-1]) + np.random.normal(0, self.sigma_z, s[-1].shape) * self.h

# 			if len(m) > 1:
# 				Z[3:6] = (Z[:3] - m[-1][:3]) / self.dt
# 				Z[6:] = (Z[:3] - 2 * m[-1][:3] + m[-2][:3]) / self.dt**2

# 			m.append(Z)
# 			step += 1

# 		return np.array(s), np.array(m)

# 	def create_dataset(self):

# 		data = np.empty((0, 9))
# 		label = np.empty((0, 9))
# 		for _ in range(self.N_samples):
# 			X, Z = self.create_sequence()
# 			data = np.vstack((data, Z))
# 			label = np.vstack((label, X))

# 		# data = np.reshape(data, (self.N_samples, self.steps, -1))
# 		# label = np.reshape(label, (self.N_samples, self.steps, -1))

# 		return data, label

def ployFit(x, dt, l):
	t = np.arange(0, len(x) * dt + dt, dt)
	u = np.zeros_like(x)
	a = np.zeros_like(x)

	for i in range(l, len(x) - l):
		xx = x[i-l:i+l+1]
		tt = t[i-l:i+l+1]

		c = P.polyfit(tt, xx, deg = 2)

		c1 = P.polyder(c)
		c2 = P.polyder(c1)

		u[i] = P.polyval(t[i], c1)
		a[i] = P.polyval(t[i], c2)

	return u, a


def main():
	# Y_ = np.load("new_Traj.npy")
	# pdf = np.zeros((3, 2))
	# for i in range(len(pdf)):
	# 	pdf[i,:] = np.array(norm.fit(Y_[:, i - 3]))

	# dataset = Data(burger, pdf, samples, steps, dt)
	# X, Y = dataset.create_dataset()

	ap = argparse.ArgumentParser()
	ap.add_argument("--mode", type = str, default = "flow")
	ap.add_argument("--samples", type = int, default = 5000)
	ap.add_argument("--steps", type = int, default = 200)
	ap.add_argument("--lr", type = float, default = 0.001)
	ap.add_argument("--epochs", type = int, default = 500)
	ap.add_argument("--load", action = 'store_true')

	args = ap.parse_args()

	if args.load:
		X = np.load("X.npy")
		Y = np.load("Y.npy")
		with open("header.pickle", 'r') as f:
			samples, steps = pickle.load(f)

	else:
		X, Y = data.generate_data(args.mode, args.samples, args.steps)

		np.save("X", X)
		np.save("Y", Y)

		with open("header.pickle", 'wb') as f:
			pickle.dump((args.samples, args.steps))

	scaler_X = MinMaxScaler()
	X = scaler_X.fit_transform(X)

	scaler_Y = MinMaxScaler()
	Y = scaler_Y.fit_transform(Y)

	X = X.reshape((args.samples, args.steps, -1))
	Y = Y.reshape((args.samples, args.steps, -1))

	batch_size = 32

	model = Sequential()
	model.add(LSTM(56, batch_input_shape = (None, None, 9), return_sequences = True))
	model.add(LSTM(128, return_sequences = True))
	model.add(LSTM(56, return_sequences = True))
	model.add(Dense(9, activation = None))
	opt = optimizers.RMSprop(lr = args.lr)
	model.compile(loss='mean_squared_error', optimizer=opt)

	model.fit(X, Y,
				epochs = args.epochs, batch_size = batch_size, verbose = 2, shuffle = True)

	model.save("my_model")

	# Y_pred = model.predict(X[-1,:,:].reshape((1, steps, 9)), batch_size = 1)
	# Y_pred = np.squeeze(Y_pred)
	# Y_pred = scaler_Y.inverse_transform(Y_pred)

	# Y[-1,:,:] = scaler_Y.inverse_transform(Y[-1,:,:])

	# x = scaler_X.inverse_transform(X[-1,:,:])
	# u_hat, a_hat = ployFit(x[:,0], dt, 2)

	# plt.figure()

	# plt.plot(Y[-1,:,0], label = "GT")
	# plt.plot(Y_pred[:,0], label = "RNN")
	# plt.legend()

	# plt.figure()

	# plt.plot(Y[-1,:,3], label = "GT")
	# plt.plot(Y_pred[:,3], label = "RNN")
	# plt.plot(u_hat, label = "Poly")
	# plt.legend()

	# plt.figure()

	# plt.plot(Y[-1,:,6], label = "GT")
	# plt.plot(Y_pred[:,6], label = "RNN")
	# plt.plot(a_hat, label = "Poly")
	# plt.legend()

	# plt.show()


if __name__ == "__main__":
	main()