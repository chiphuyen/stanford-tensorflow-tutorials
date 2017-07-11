"""
Example to demonstrate the ops of tf.Variables()
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

# Example 1: how to run assign op
W = tf.Variable(10)
assign_op = W.assign(100)

with tf.Session() as sess:
	sess.run(W.initializer)
	print(W.eval()) # >> 10
	print(sess.run(assign_op)) # >> 100

# Example 2: tricky example
# create a variable whose original value is 2
my_var = tf.Variable(2, name="my_var") 

# assign 2 * my_var to my_var and run the op my_var_times_two
my_var_times_two = my_var.assign(2 * my_var)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run(my_var_times_two)) # >> 4
	print(sess.run(my_var_times_two)) # >> 8
	print(sess.run(my_var_times_two)) # >> 16

# Example 3: each session maintains its own copy of variables
W = tf.Variable(10)
sess1 = tf.Session()
sess2 = tf.Session()

# You have to initialize W at each session
sess1.run(W.initializer)
sess2.run(W.initializer)

print(sess1.run(W.assign_add(10))) # >> 20
print(sess2.run(W.assign_sub(2))) # >> 8

print(sess1.run(W.assign_add(100))) # >> 120
print(sess2.run(W.assign_sub(50))) # >> -42

sess1.close()
sess2.close()