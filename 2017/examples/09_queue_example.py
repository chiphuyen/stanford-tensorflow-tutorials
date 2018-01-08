""" Example to demonstrate how to use queues
Author: Chip Huyen
Prepared for the class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf

N_SAMPLES = 1000
NUM_THREADS = 4
# Generating some simple data
# create 1000 random samples, each is a 1D array from the normal distribution (10, 1)
data = 10 * np.random.randn(N_SAMPLES, 4) + 1 
# create 1000 random labels of 0 and 1
target = np.random.randint(0, 2, size=N_SAMPLES) 

queue = tf.FIFOQueue(capacity=50, dtypes=[tf.float32, tf.int32], shapes=[[4], []])

enqueue_op = queue.enqueue_many([data, target])
data_sample, label_sample = queue.dequeue()

# create ops that do something with data_sample and label_sample

# create NUM_THREADS to do enqueue
qr = tf.train.QueueRunner(queue, [enqueue_op] * NUM_THREADS)
with tf.Session() as sess:
	# create a coordinator, launch the queue runner threads.
	coord = tf.train.Coordinator()
	enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
	try:
		for step in range(100): # do to 100 iterations
			if coord.should_stop():
				break
			data_batch, label_batch = sess.run([data_sample, label_sample])
			print(data_batch)
			print(label_batch)
	except Exception as e:
		coord.request_stop(e)
	finally:
		coord.request_stop()
		coord.join(enqueue_threads)