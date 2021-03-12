#script tensorl4.py
# Function minimization using SGD
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#x = tf.Variable([tf.random.normal([1])])
x = tf.Variable(tf.random.normal([1]))
print(f'Initiaizing x = {x.numpy()}')

learning_rate = 1e-2
history = []
target = 4
x_f = tf.constant(target,tf.float32)

for i in range(200):
    with tf.GradientTape() as tape:
        loss = (x - x_f) * (x - x_f)

    grad = tape.gradient(loss, x)
    new_x = x - learning_rate * grad
    x.assign(new_x)
    history.append(x.numpy()[0])

print(f'history - {history}')

# plot optimization progress
#plt.plot(history)
plt.plot([0,200],[target,target])
#plt.legend(('pedicted','true'))
plt.xlabel('Iteration')
plt.ylabel('value of x')
plt.show()


