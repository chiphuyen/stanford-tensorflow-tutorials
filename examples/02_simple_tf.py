""" Simple TensorFlow's ops
Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf

# Example 1: Simple ways to create log file writer
a = tf.constant(2, name='a')
b = tf.constant(3, name='b')
x = tf.add(a, b, name='add')
writer = tf.summary.FileWriter('./graphs/simple', tf.get_default_graph()) 
with tf.Session() as sess:
    # writer = tf.summary.FileWriter('./graphs', sess.graph) 
    print(sess.run(x))
writer.close() # close the writer when you’re done using it

# Example 2: The wonderful wizard of div
a = tf.constant([2, 2], name='a')
b = tf.constant([[0, 1], [2, 3]], name='b')

with tf.Session() as sess:
    print(sess.run(tf.div(b, a)))
    print(sess.run(tf.divide(b, a)))
    print(sess.run(tf.truediv(b, a)))
    print(sess.run(tf.floordiv(b, a)))
    # print(sess.run(tf.realdiv(b, a)))
    print(sess.run(tf.truncatediv(b, a)))
    print(sess.run(tf.floor_div(b, a)))

# Example 3: multiplying tensors
a = tf.constant([10, 20], name='a')
b = tf.constant([2, 3], name='b')

with tf.Session() as sess:
    print(sess.run(tf.multiply(a, b)))
    print(sess.run(tf.tensordot(a, b, 1)))

# Example 4: Python native type
t_0 = 19 
x = tf.zeros_like(t_0) 					# ==> 0
y = tf.ones_like(t_0) 					# ==> 1

t_1 = ['apple', 'peach', 'banana']
x = tf.zeros_like(t_1) 					# ==> ['' '' '']
# y = tf.ones_like(t_1) 				# ==> TypeError: Expected string, got 1 of type 'int' instead.

t_2 = [[True, False, False],
       [False, False, True],
       [False, True, False]] 
x = tf.zeros_like(t_2) 					# ==> 3x3 tensor, all elements are False
y = tf.ones_like(t_2) 					# ==> 3x3 tensor, all elements are True

print(tf.int32.as_numpy_dtype())

# Example 5: printing your graph's definition
my_const = tf.constant([1.0, 2.0], name='my_const')
print(tf.get_default_graph().as_graph_def())