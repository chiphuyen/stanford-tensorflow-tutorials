""" Variable exmaples
Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Lecture 02
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf

# Example 1: creating variables
s = tf.Variable(2, name='scalar') 
m = tf.Variable([[0, 1], [2, 3]], name='matrix') 
W = tf.Variable(tf.zeros([784,10]), name='big_matrix')
V = tf.Variable(tf.truncated_normal([784, 10]), name='normal_matrix')

s = tf.get_variable('scalar', initializer=tf.constant(2)) 
m = tf.get_variable('matrix', initializer=tf.constant([[0, 1], [2, 3]]))
W = tf.get_variable('big_matrix', shape=(784, 10), initializer=tf.zeros_initializer())
V = tf.get_variable('normal_matrix', shape=(784, 10), initializer=tf.truncated_normal_initializer())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(V.eval())

# Example 2: assigning values to variables
W = tf.Variable(10)
W.assign(100)
with tf.Session() as sess:
    sess.run(W.initializer)
    print(sess.run(W))                    	# >> 10

W = tf.Variable(10)
assign_op = W.assign(100)
with tf.Session() as sess:
    sess.run(assign_op)
    print(W.eval())                     	# >> 100

# create a variable whose original value is 2
a = tf.get_variable('scalar', initializer=tf.constant(2)) 
a_times_two = a.assign(a * 2)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) 
    sess.run(a_times_two)                 	# >> 4
    sess.run(a_times_two)                 	# >> 8
    sess.run(a_times_two)                 	# >> 16

W = tf.Variable(10)
with tf.Session() as sess:
    sess.run(W.initializer)
    print(sess.run(W.assign_add(10)))     	# >> 20
    print(sess.run(W.assign_sub(2)))     	# >> 18

# Example 3: Each session has its own copy of variable
W = tf.Variable(10)
sess1 = tf.Session()
sess2 = tf.Session()
sess1.run(W.initializer)
sess2.run(W.initializer)
print(sess1.run(W.assign_add(10)))        	# >> 20
print(sess2.run(W.assign_sub(2)))        	# >> 8
print(sess1.run(W.assign_add(100)))        	# >> 120
print(sess2.run(W.assign_sub(50)))        	# >> -42
sess1.close()
sess2.close()

# Example 4: create a variable with the initial value depending on another variable
W = tf.Variable(tf.truncated_normal([700, 10]))
U = tf.Variable(W * 2)