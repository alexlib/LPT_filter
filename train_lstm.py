import numpy as np
import matplotlib.pyplot as plt
import math
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy.polynomial.polynomial as P
from scipy.integrate import ode
from scipy.stats import norm
import argparse
import pickle
import os
import shutil

import data
import synthetic_3d as syn


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

	# dataset = Data(burger, pdf, samples, steps, dt)
	# X, Y = dataset.create_dataset()

	ap = argparse.ArgumentParser()
	ap.add_argument("--mode", type = str, default = "flow")
	ap.add_argument("--samples", type = int, default = 5000)
	ap.add_argument("--steps", type = int, default = 200)
	ap.add_argument("--lr", type = float, default = 0.001)
	ap.add_argument("--epochs", type = int, default = 500)
	ap.add_argument("--load", action = 'store_true')
	ap.add_argument("--logdir", type = str)

	args = ap.parse_args()

	if args.load:
		X = np.load("X.npy")
		Y = np.load("Y.npy")
		with open("header.pickle", 'r') as f:
			samples, steps = pickle.load(f)

	else:
		X, Y = data.generate_data(args.mode, args.samples, args.steps)

		if os.path.exists(args.logdir):
			shutil.rmtree(args.logdir)
		os.mkdir(args.logdir)
		os.chdir(args.logdir)

		np.save("X", X)
		np.save("Y", Y)

		with open("header.pickle", 'wb') as f:
			pickle.dump((args.samples, args.steps), f)

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