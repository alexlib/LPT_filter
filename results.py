import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def make_plot(X, dict, axis, i, j):
	plt.figure()
	for key, value in sorted(dict.iteritems()):
		if axis == 0:
			y = value[:,i,j]
		elif axis == 1:
			y = value[i,:,j]
		elif axis == 2:
			y = value[i,j,:]
		else:
			raise(Error("axis out of bound"))

		plt.plot(X[axis], y, '*', label = key)

	plt.legend()


def main():
	with open("err_U", 'r') as f:
		err_U = pickle.load(f)

	with open("err_A", 'r') as f:
		err_A = pickle.load(f)


	skip = err_U["skip"]
	del err_U["skip"]

	delta = err_U["delta"]
	del err_U["delta"]

	N_sample = err_U["N_sample"]
	del err_U["N_sample"]

	X = [skip, delta, N_sample]

	make_plot(X, err_U, 0, 0, 0)
	make_plot(X, err_A, 0, 0, 0)
	plt.show()

if __name__ == "__main__":
	main()