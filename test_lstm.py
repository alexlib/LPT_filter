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
import argparse

import data
import filters


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--mode", type = str, default = "flow")
	ap.add_argument("--samples", type = int, default = 1)
	ap.add_argument("--steps", type = int, default = 500)

	args = ap.parse_args()

	X = np.load("X.npy")
	scaler_X = MinMaxScaler()
	scaler_X.fit(X)

	Y = np.load("Y_.npy")
	scaler_Y = MinMaxScaler()
	scaler_Y.fit(Y)

	# data = Data(burger, samples, N_t, dt, sigma_z = 0.1)
	# X, Y_ = data.create_dataset()

	burger, dt = data.generate_flow()

	X, Y_ = data.generate_data(args.mode, args.samples, args.steps)

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

	model = load_model("my_model")

	X = scaler_X.transform(X)
	Y = model.predict(X.reshape((args.samples, -1, 9)), batch_size = args.samples)
	Y = np.squeeze(Y)
	Y = scaler_Y.inverse_transform(Y)

	X = scaler_X.inverse_transform(X)

	_, axarr = plt.subplots(3, sharex = True)
	for i in range(3):
		axarr[i].plot(Y_[:, i], label = "GT")
		axarr[i].plot(Y[:, i], label = "RNN")
	plt.legend()

	
	_, axarr = plt.subplots(3, sharex = True)
	for i in range(3):
		axarr[i].plot(Y_[:, i + 3], label = "GT")
		axarr[i].plot(Y[:, i + 3], label = "RNN")
		axarr[i].plot(U['Polynomial'][0][:, i], label = "Polynomial")
	plt.legend()


	_, axarr = plt.subplots(3, sharex = True)
	for axis in range(3):
		axarr[axis].plot(Y_[:, axis + 3], label = "GT")
		axarr[axis].plot(Y[:, axis + 6], label = "RNN")
		axarr[axis].plot(A['Polynomial'][0][:, axis], label = "Polynomial")
	plt.legend()

	fig = plt.figure()
	ax = fig.gca(projection='3d')

	for i in range(args.samples):
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