import numpy as np

from network import NeuralNetwork
from layers import DenseLayer, ActivationLayer
from activations import tanh, dtanh
from losses import mse, dmse


x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

model = NeuralNetwork()
model.add(DenseLayer(2, 3))
model.add(ActivationLayer(tanh, dtanh))
model.add(DenseLayer(3, 1))
model.add(ActivationLayer(tanh, dtanh))

model.use(mse, dmse)

model.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

out = model.predict(x_train)
print(out)