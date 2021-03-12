#script tensorl1.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class OurDenseLayer(tf.keras.layers.Layer):
    def __init__(self, n_output_nodes):
        super(OurDenseLayer,self).__init__()
        self.n_output_nodes = n_output_nodes

    def build(self,input_shape):
        input_d = int(input_shape[-1])
        self.W = self.add_weight("weight",shape=[input_d,self.n_output_nodes])
        self.b = self.add_weight("bias", shape=[1,self.n_output_nodes])

    def call(self,x):
        z = tf.add(tf.matmul(tf.transpose(self.W),tf.transpose(x)),tf.transpose(self.b))
        y = tf.sigmoid(z)
        return y

# set random seed and perform calculation
tf.random.set_seed(1)
layer = OurDenseLayer(3)
layer.build((1,2))
x_input = tf.constant([[3,4]],tf.float32,shape=(1,2))
print(f'shape of x_input {tf.shape(x_input).numpy()}')
print("calling computation")
y = layer.call(x_input)
print("Calulated output y:")
print(y.numpy())


