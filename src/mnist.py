import numpy as np

from network import NeuralNetwork
from layers import DenseLayer, ActivationLayer
from activations import tanh, dtanh
from losses import mse, dmse

from keras.datasets import mnist
from keras.utils import np_utils

# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255

y_train = np_utils.to_categorical(y_train)

x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)

# Network
model = NeuralNetwork()
model.add(DenseLayer(28*28, 100))
model.add(ActivationLayer(tanh, dtanh))
model.add(DenseLayer(100, 50))
model.add(ActivationLayer(tanh, dtanh))
model.add(DenseLayer(50, 10))
model.add(ActivationLayer(tanh, dtanh))

model.use(mse, dmse)
model.fit(x_train[0:1000], y_train[0:1000], epochs=35, learning_rate=0.1)

# test on 3 samples
out = model.predict(x_test[0:3])
print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(y_test[0:3])