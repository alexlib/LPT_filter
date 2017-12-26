import numpy as np
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import CuDNNLSTM, CuDNNGRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def create_dataset(dataset, t, look_back=1):
	dataX, T, dataY = [], [], []
	for i in range(look_back - 1, len(t)):
		a = dataset[i - look_back + 1:i+1,0]
		dataX.append(a)
		T.append(t[i - look_back + 1:i+1])
		dataY.append(dataset[i,0])
	return np.array(dataX), np.array(T), np.array(dataY)

np.random.seed(7)

t = np.linspace(0, 2, 1001)
x = np.exp(t) + 0.1 * np.random.rand(len(t))
x = np.reshape(x, (-1,1))

scaler = MinMaxScaler(feature_range=(-1,1))
dataset = scaler.fit_transform(x)

train_size = int(len(dataset) * 0.66)
test_size = len(dataset) - train_size
train, test = dataset[:train_size], dataset[train_size:]

look_back = 10
trainX, trainT, trainY = create_dataset(train, t[:train_size], look_back)
testX, testT, testY = create_dataset(test, t[train_size:], 1)

trainX = np.reshape(trainX, (trainX.shape[0], -1, 1))
trainT = np.reshape(trainT, (trainT.shape[0], -1, 1))
trainX = np.concatenate((trainX, trainT), axis = 2)
testX = np.reshape(testX, (testX.shape[0], -1, 1))
testT = np.reshape(testT, (testT.shape[0], -1, 1))
testX = np.concatenate((testX, testT), axis = 2)

model = Sequential()
model.add(Dense(28, input_shape = (None, 2), activation = 'tanh'))
model.add(CuDNNLSTM(56, return_sequences = True))
model.add(CuDNNLSTM(128, return_sequences = True))
model.add(CuDNNLSTM(512, return_sequences = True))
model.add(CuDNNLSTM(1024, return_sequences = True))
model.add(CuDNNLSTM(56))
model.add(Dense(56, activation = 'tanh'))
model.add(Dense(28, activation = 'tanh'))
model.add(Dense(1, activation = None))
model.compile(loss='mean_squared_error', optimizer='SGD')

model.fit(trainX, trainY, epochs = 10, verbose = 2, shuffle = True)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:,0] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back,0] = trainPredict[:,0]

# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:,0] = np.nan
testPredictPlot[-len(testPredict):,0] = testPredict[:,0]

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.plot(np.exp(t))
plt.show()