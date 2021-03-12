#script tensorl0.py
import tensorflow as tf
import numpy as np

number = tf.constant(3.1416,tf.float32)
print(f'number is a {tf.rank(number).numpy()} - d tensor')
print(number.numpy())
matrix = tf.constant([[2,1],[1,3]],tf.int32)
print(f'matrix is a {tf.rank(matrix).numpy()} - d tensor with shape {tf.shape(matrix).numpy().tolist()}')
assert isinstance(matrix,tf.Tensor), "matrix must be a tf tensor object"
print(matrix.numpy())
input_shape = [1,2]
print(input_shape[-1])
# print("image tensor")
# image = tf.zeros(shape=(3,4),dtype=tf.int32)
# assert isinstance(image,tf.Tensor), "image must be a tf tensor object"
# print(image.numpy())
a = tf.constant(10)
b = tf.constant(15)
c1  = tf.add(a,b)
print(c1)
def func_calc(a,b):
    c = tf.add(a,b)
    d = tf.subtract(b,5)
    e = tf.multiply(c,d)
    return e
a,b = 20,25
e_out = func_calc(a,b)
print(e_out)

