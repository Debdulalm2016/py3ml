#script tensorl3.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

class SubClassModel(Model):
    def __init__(self, n_output_nodes):
        super(SubClassModel, self).__init__()
        self.dense_layer = Dense(n_output_nodes,input_shape=(2,),activation='sigmoid')

    def call(self, inputs):
        return self.dense_layer(inputs)


# perform calculation
n_output_nodes = 3
my_model = SubClassModel(n_output_nodes)
x_input = tf.constant([[3,4],[4,5],[5,6]],tf.float32,shape=(3,2))
print(f'shape of x_input {tf.shape(x_input).numpy()}')
print("calling computation")
y = my_model.call(x_input)
print("Calulated output y:")
print(y.numpy())


