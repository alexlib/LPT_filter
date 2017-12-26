import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

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


def surf_plot(X, Y, dict, delta):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	for key, value in sorted(dict.iteritems()):
		if key == "Polynomial":
			Z = value[:,:,delta].transpose()
			surf = ax.plot_surface(X, Y, Z, cmap = cm.coolwarm,
								linewidth = 0, shade = True)

	# fig.legend()
	ax.set_xlabel("skip")
	ax.set_ylabel("width")
	ax.set_zlabel("error")
	ax.set_zscale('log')
	fig.colorbar(surf, shrink = 0.75, aspect = 8)


def main():
	with open("err_U", 'r') as f:
		err_U = pickle.load(f)

	with open("err_A", 'r') as f:
		err_A = pickle.load(f)


	skip = err_U["skip"]
	del err_U["skip"]

	delta = err_U["delta"]
	del err_U["delta"]

	width = err_U["width"]
	del err_U["width"]

	# X = [skip, delta, N_sample]

	# make_plot(X, err_U, 0, 0, 0)
	# make_plot(X, err_A, 0, 0, 0)
	# plt.show()

	# print err_A["Finite difference"][:,:,0]

	X, Y = np.meshgrid(skip, width)
	
	surf_plot(X, Y, err_A, 0)
	plt.show()

if __name__ == "__main__":
	main()