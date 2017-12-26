import numpy as np
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import CuDNNLSTM, CuDNNGRU, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy.polynomial.polynomial as P
from scipy.integrate import ode

class Data(object):
	def __init__(self, N_sample, steps, dt, sigma_a = 0.1, sigma_z = 0.05):

		self.N_sample = N_sample
		self.steps = steps
		self.sigma_a = sigma_a
		self.sigma_z = sigma_z
		self.dt = dt

		self.A = np.array([[1, dt, 0.5 * dt**2], [0, 1, dt], [0, 0, 1]])
		self.H = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])

	def create_sequence(self):

		X0 = np.array([np.random.uniform(0, 1),
						np.random.uniform(0, 0.1),
						np.random.uniform(0, 0.01)]).transpose()

		X = []
		X.append(X0)
		Z = []
		Z.append(X0)

		for _ in range(self.steps):
			X_ = np.dot(self.A, X[-1]) + np.random.normal(0, self.sigma_a) * np.array([0, 0, 1]).transpose()
			X.append(X_)

			Z_ = np.dot(self.H, X_) + np.random.normal(0, self.sigma_z, X_.shape) * np.array([1, 0, 0]).transpose()

			if len(Z) > 1:
				Z_[1] = (Z_[0] - Z[-1][0]) / self.dt
				Z_[2] = (Z_[0] - 2 * Z[-1][0] + Z[-2][0]) / self.dt**2

			Z.append(Z_)

		return np.array(X[1:]), np.array(Z[1:])

	def create_dataset(self):

		data = np.empty((0, 3))
		label = np.empty((0, 3))
		for _ in range(self.N_sample):
			X, Z = self.create_sequence()
			data = np.vstack((data, Z))
			label = np.vstack((label, X))

		# data = np.reshape(data, (self.N_sample, self.steps, -1))
		# label = np.reshape(label, (self.N_sample, self.steps, -1))

		return data, label

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


sample = 1000
steps = 100
dt = 0.1

dataset = Data(sample, steps, dt)
X, Y = dataset.create_dataset()

scaler = MinMaxScaler(feature_range=(-1,1))
X = scaler.fit_transform(X)
Y = scaler.fit_transform(Y)

X = X.reshape((sample, steps, -1))
Y = Y.reshape((sample, steps, -1))

batch_size = 32

model = Sequential()
model.add(LSTM(56, batch_input_shape = (None, None, 3), return_sequences = True))
model.add(LSTM(128, return_sequences = True))
model.add(LSTM(56, return_sequences = True))
# model.add(Dense(3, activation = None))
model.add(Dense(3, activation = None))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X[:-1,:,:], Y[:-1,:,:],
			epochs = 200, batch_size = batch_size, verbose = 2, shuffle = True)

model.save("my_model")

Y_pred = model.predict(X[-1,:,:].reshape((1, steps, 3)), batch_size = 1)
Y_pred = np.squeeze(Y_pred)
Y_pred = scaler.inverse_transform(Y_pred)

Y[-1,:,:] = scaler.inverse_transform(Y[-1,:,:])

x = scaler.inverse_transform(X[-1,:,:])
u_hat, a_hat = ployFit(x[:,0], dt, 2)

plt.figure()

plt.plot(Y[-1,:,0], label = "GT")
plt.plot(Y_pred[:,0], label = "RNN")
plt.legend()

plt.figure()

plt.plot(Y[-1,:,1], label = "GT")
plt.plot(Y_pred[:,1], label = "RNN")
plt.plot(u_hat, label = "Poly")
plt.legend()

plt.figure()

plt.plot(Y[-1,:,2], label = "GT")
plt.plot(Y_pred[:,2], label = "RNN")
plt.plot(a_hat, label = "Poly")
plt.legend()

plt.show()