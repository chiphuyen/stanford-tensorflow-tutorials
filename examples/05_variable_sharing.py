""" Examples to demonstrate variable sharing
CS 20: 'TensorFlow for Deep Learning Research'
cs20.stanford.edu
Chip Huyen (chiphuyen@cs.stanford.edu)
Lecture 05
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

x1 = tf.truncated_normal([200, 100], name='x1')
x2 = tf.truncated_normal([200, 100], name='x2')

def two_hidden_layers(x):
    assert x.shape.as_list() == [200, 100]
    w1 = tf.Variable(tf.random_normal([100, 50]), name='h1_weights')
    b1 = tf.Variable(tf.zeros([50]), name='h1_biases')
    h1 = tf.matmul(x, w1) + b1
    assert h1.shape.as_list() == [200, 50]  
    w2 = tf.Variable(tf.random_normal([50, 10]), name='h2_weights')
    b2 = tf.Variable(tf.zeros([10]), name='2_biases')
    logits = tf.matmul(h1, w2) + b2
    return logits

def two_hidden_layers_2(x):
    assert x.shape.as_list() == [200, 100]
    w1 = tf.get_variable('h1_weights', [100, 50], initializer=tf.random_normal_initializer())
    b1 = tf.get_variable('h1_biases', [50], initializer=tf.constant_initializer(0.0))
    h1 = tf.matmul(x, w1) + b1
    assert h1.shape.as_list() == [200, 50]  
    w2 = tf.get_variable('h2_weights', [50, 10], initializer=tf.random_normal_initializer())
    b2 = tf.get_variable('h2_biases', [10], initializer=tf.constant_initializer(0.0))
    logits = tf.matmul(h1, w2) + b2
    return logits

# logits1 = two_hidden_layers(x1)
# logits2 = two_hidden_layers(x2)

# logits1 = two_hidden_layers_2(x1)
# logits2 = two_hidden_layers_2(x2)

# with tf.variable_scope('two_layers') as scope:
#     logits1 = two_hidden_layers_2(x1)
#     scope.reuse_variables()
#     logits2 = two_hidden_layers_2(x2)

# with tf.variable_scope('two_layers') as scope:
#     logits1 = two_hidden_layers_2(x1)
#     scope.reuse_variables()
#     logits2 = two_hidden_layers_2(x2)

def fully_connected(x, output_dim, scope):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as scope:
        w = tf.get_variable('weights', [x.shape[1], output_dim], initializer=tf.random_normal_initializer())
        b = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        return tf.matmul(x, w) + b

def two_hidden_layers(x):
    h1 = fully_connected(x, 50, 'h1')
    h2 = fully_connected(h1, 10, 'h2')

with tf.variable_scope('two_layers') as scope:
    logits1 = two_hidden_layers(x1)
    # scope.reuse_variables()
    logits2 = two_hidden_layers(x2)

writer = tf.summary.FileWriter('./graphs/cool_variables', tf.get_default_graph())
writer.close()