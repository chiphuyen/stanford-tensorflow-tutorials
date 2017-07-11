""" Examples to demonstrate ops level randomization
Author: Chip Huyen
Prepared for the class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

# Example 1: session is the thing that keeps track of random state
c = tf.random_uniform([], -10, 10, seed=2)

with tf.Session() as sess:
    print(sess.run(c)) # >> 3.57493
    print(sess.run(c)) # >> -5.97319

# Example 2: each new session will start the random state all over again.
c = tf.random_uniform([], -10, 10, seed=2)

with tf.Session() as sess:
    print(sess.run(c)) # >> 3.57493

with tf.Session() as sess:
    print(sess.run(c)) # >> 3.57493

# Example 3: with operation level random seed, each op keeps its own seed.
c = tf.random_uniform([], -10, 10, seed=2)
d = tf.random_uniform([], -10, 10, seed=2)

with tf.Session() as sess:
    print(sess.run(c)) # >> 3.57493
    print(sess.run(d)) # >> 3.57493

# Example 4: graph level random seed
tf.set_random_seed(2)
c = tf.random_uniform([], -10, 10)
d = tf.random_uniform([], -10, 10)

with tf.Session() as sess:
    print(sess.run(c)) # >> 9.12393
    print(sess.run(d)) # >> -4.53404
    