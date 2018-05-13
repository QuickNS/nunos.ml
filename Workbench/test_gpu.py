# To run the below script on GPU in a Docker container on a Linux VM,
# follow the instructions in the readme file.

import tensorflow as tf
import keras

# testing Tensorflow
with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape = [2, 3], name = 'a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape = [3, 2], name = 'b')
    c = tf.matmul(a, b)
	
sess = tf.Session(config = tf.ConfigProto(log_device_placement = True))
print("RESULTS using GPU:")
print(sess.run(c))

# testing Keras
print("Keras version: ", keras.__version__)
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

print(network.summary())


