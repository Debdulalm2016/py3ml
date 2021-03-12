#script tensorl5d.py
import tensorflow as tf
import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

n_output_nodes = 10
n_final_layer_nodes = 1

model = Sequential()

dense_layer = Dense(n_output_nodes, input_shape=(2,))
model.add(dense_layer)
dense_layer = Dense(n_final_layer_nodes)
model.add(dense_layer)

#prepare training data
x_train = tf.constant([[1,2],[0,0],[2,2],[2,3],[1,1],[3,3],[4,1],[4,4],[5,5],[4,2],[5,2],[4,0],[5,1],[3,5]],tf.float32,shape=(14,2))
y_train = tf.constant([6,0,8,10,4,12,10,16,20,12,14,8,12,16],tf.float32)

inp_ds = tf.data.Dataset.from_tensor_slices((x_train,y_train))
print(inp_ds)
print(list(inp_ds.as_numpy_iterator()))

batched_inp_ds = inp_ds.batch(5)
print(batched_inp_ds)
print(list(batched_inp_ds.as_numpy_iterator()))


# test data
x_test = tf.constant([[3,4],[4,5],[1,4]],tf.float32,shape=(3,2))
y_test = tf.constant([14,18,10],tf.float32)

#loss_fn = tf.keras.

model.compile(optimizer = 'sgd',
              loss = 'mse',
              metrics = ['mse'])
#model.fit(x_train,y_train,epochs = 5)
model.fit(batched_inp_ds,epochs=5)

print(model.summary())
model.evaluate(x_test,y_test, verbose=2)
#print("calling computation")
#y = model.call(x_test)
#print("Calulated output y:")
#print(y.numpy())
#print("recalling computation")
#y_out = model(x_test)
#print(y_out.numpy())



