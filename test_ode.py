import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, ode

def f(t, state):
	return [state[2], state[3], state[4], state[5], 0, 0]


def foo(t, y):
	return [y[3], y[4], y[5], y[6], y[7], y[8], 0, 0, 0]

class Data(object):
	def __init__(self, N_sample, steps, dt, sigma_a = 0.1, sigma_z = 0.05):

		self.N_sample = N_sample
		self.steps = steps
		self.sigma_a = sigma_a
		self.sigma_z = sigma_z
		self.dt = dt

		self.H = np.zeros((9,9))
		self.H[0, 0] = 1
		self.H[1, 1] = 1
		self.H[2, 2] = 1

		self.h = np.zeros(9)
		self.h[:3] = 1

	def create_sequence(self):

		X0 = np.array([np.random.uniform(0, 1),
						np.random.uniform(0, 1),
						np.random.uniform(0, 1)])

		U0 = np.array([np.random.uniform(0, 0.1),
						np.random.uniform(0, 0.1),
						np.random.uniform(0, 0.1)])

		A0 = np.array([np.random.choice([-1, 1]) * np.random.exponential(self.sigma_a),
						np.random.choice([-1, 1]) * np.random.exponential(self.sigma_a),
						np.random.choice([-1, 1]) * np.random.exponential(self.sigma_a)])

		t0 = 0

		s = []
		s.append(np.concatenate((X0, U0, A0)))
		m = []
		m.append(np.concatenate((X0, U0, A0)))

		r = ode(foo).set_integrator('dopri5')
		r.set_initial_value(s[0], t0)

		step = 1
		while r.successful() and step < self.steps:

			r.integrate(r.t + self.dt)

			s.append(r.y)

			r.y[-3:] += np.array([np.random.choice([-1, 1]) * np.random.exponential(self.sigma_a),
							np.random.choice([-1, 1]) * np.random.exponential(self.sigma_a),
							np.random.choice([-1, 1]) * np.random.exponential(self.sigma_a)])


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
		for _ in range(self.N_sample):
			X, Z = self.create_sequence()
			data = np.vstack((data, Z))
			label = np.vstack((label, X))

		data = np.reshape(data, (self.N_sample, self.steps, -1))
		label = np.reshape(label, (self.N_sample, self.steps, -1))

		return data, label

# y0, t0 = [1, 0, 0.1, 0.2, 0.01, 0.02], 0
# t1 = 10
# dt = 0.1

# r = ode(f).set_integrator('dopri5')
# r.set_initial_value(y0, t0)

# t = list()
# t.append(t0)
# y = list()
# y.append(y0)

# while r.successful() and r.t <= t1:
# 	r.integrate(r.t + dt)
# 	t.append(r.t)
# 	y.append(r.y)

# 	r.y[-2] = np.random.choice([-1, 1]) * np.random.exponential(0.1)
# 	r.y[-1] = np.random.choice([-1, 1]) * np.random.exponential(0.1)


sample = 10
steps = 100
dt = 0.1

dataset = Data(sample, steps, dt)
X, Y = dataset.create_dataset()


plt.figure()

plt.plot(Y[-1,:,:3], label = "GT")
plt.legend()

plt.figure()

plt.plot(Y[-1,:,3:6], label = "GT")
plt.legend()

plt.figure()

plt.plot(Y[-1,:,6:], label = "GT")
plt.legend()

plt.show()