# script: mnistdata.py
import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()
#x_train, x_test = x_train/255.0, x_test/255.0
print(type(x_train))
print(tf.shape(x_train).numpy())
print(tf.shape(y_train).numpy())
#print(x_train[:1])
#print(y_train[:1])
assert isinstance(x_train,np.ndarray), "x_train is not an numpy array"

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28,28)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation = 'sigmoid')
    ])

predictions = model(x_train[:1])
print("prediction from untrained model:")
print(predictions)
print('calling prediction')
x_input = tf.constant(x_train[:1])
preds = model.call(x_input)
print(preds)


