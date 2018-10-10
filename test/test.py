import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
graph = tf.Graph()
# with graph.as_default():
#     foo = tf.Variable(3, name='foo')
#     bar = tf.Variable(2, name='bar')
#     result = foo + bar
#     initialize = tf.global_variables_initializer()
#
# with tf.Session(graph=graph) as sess:
#     sess.run(initialize)
#     res = sess.run(result)
a = tf.constant([1, 2, 3, 4])
b = tf.square(a)
res = tf.random_uniform((4, 5, 2), -1, 1)
with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(res))
