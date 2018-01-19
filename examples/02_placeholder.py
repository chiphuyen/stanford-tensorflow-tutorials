""" Placeholder and feed_dict example
Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Lecture 02
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

# Example 1: feed_dict with placeholder

# a is a placeholderfor a vector of 3 elements, type tf.float32
a = tf.placeholder(tf.float32, shape=[3])
b = tf.constant([5, 5, 5], tf.float32)

# use the placeholder as you would a constant
c = a + b  # short for tf.add(a, b)

writer = tf.summary.FileWriter('graphs/placeholders', tf.get_default_graph())
with tf.Session() as sess:
    # compute the value of c given the value of a is [1, 2, 3]
    print(sess.run(c, {a: [1, 2, 3]}))                 # [6. 7. 8.]
writer.close()


# Example 2: feed_dict with variables
a = tf.add(2, 5)
b = tf.multiply(a, 3)

with tf.Session() as sess:
    print(sess.run(b))                                 # >> 21
    # compute the value of b given the value of a is 15
    print(sess.run(b, feed_dict={a: 15}))              # >> 45