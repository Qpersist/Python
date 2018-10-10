import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
num_point = 1000
vectors_set = []
for i in range(num_point):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1*0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

plt.scatter(x_data, y_data, c='r')
plt.show()

w = tf.Variable(tf.random_normal([1], -1.0, 1.0), name='w')
b = tf.Variable(tf.zeros([1]), name='b')

y = w * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data), name='loss')

optimizer = tf.train.GradientDescentOptimizer(0.5)

train = optimizer.minimize(loss, name='loss')

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('w= ', sess.run(w), 'b= ', sess.run(b), 'loss= ', sess.run(loss))
    for step in range(20):
        sess.run(train)
        print('w= ', sess.run(w), 'b= ', sess.run(b), 'loss= ', sess.run(loss))