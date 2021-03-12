#script tensorl2.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

n_output_nodes = 3
n_final_layer_nodes = 1

model = Sequential()

dense_layer = Dense(n_output_nodes, input_shape=(2,),activation='sigmoid')
model.add(dense_layer)
dense_layer = Dense(n_final_layer_nodes,activation='sigmoid')
model.add(dense_layer)

# perform calculation
x_input = tf.constant([[3,4],[4,5],[5,6]],tf.float32,shape=(3,2))
print(f'shape of x_input {tf.shape(x_input).numpy()}')
print("calling computation")
y = model.call(x_input)
print("Calulated output y:")
print(y.numpy())


