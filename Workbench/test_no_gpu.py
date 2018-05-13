import keras
# testing Keras
print("Keras version: ", keras.__version__)
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

print(network.summary())