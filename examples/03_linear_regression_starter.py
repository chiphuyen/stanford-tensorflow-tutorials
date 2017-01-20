"""
Simple linear regression example in TensorFlow
This program tries to predict the number of thefts from 
the number of fire in the city of Chicago
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

DATA_FILE = 'data/fire_theft.xls'

# Phase 1: Assemble the graph
# Step 1: read in data from the .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

# Step 2: create placeholders for input X (number of fire) and label Y (number of theft)


# Step 3: create weight and bias, initialized to 0
# name your variables w and b


# Step 4: predict Y (number of theft) from the number of fire
# name your variable Y_predicted


# Step 5: use the square error as the loss function
# name your variable loss

# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
 
# Phase 2: Train our model
with tf.Session() as sess:
	# Step 7: initialize the necessary variables, in this case, w and b
	# TO - DO	

	# Step 8: train the model
	for i in range(100): # run 100 epochs
		total_loss = 0
		for x, y in data:
			# Session runs optimizer to minimize loss and fetch the value of loss
			# TO DO: write sess.run()
			total_loss += l
		print "Epoch {0}: {1}".format(i, total_loss/n_samples)
	
# plot the results
# X, Y = data.T[0], data.T[1]
# plt.plot(X, Y, 'bo', label='Real data')
# plt.plot(X, X * w + b, 'r', label='Predicted data')
# plt.legend()
# plt.show()