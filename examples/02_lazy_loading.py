""" Example to demonstrate how the graph definition gets
bloated because of lazy loading
Author: Chip Huyen
Prepared for the class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf 

######################################## 
## NORMAL LOADING   			      ##
## print out a graph with 1 Add node  ## 
########################################

x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
z = tf.add(x, y)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter('./graphs/l2', sess.graph)
	for _ in range(10):
		sess.run(z)
	print(tf.get_default_graph().as_graph_def())
	writer.close()

######################################## 
## LAZY LOADING   					  ##
## print out a graph with 10 Add nodes## 
########################################

x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter('./graphs/l2', sess.graph)
	for _ in range(10):
		sess.run(tf.add(x, y))
	print(tf.get_default_graph().as_graph_def()) 
	writer.close()