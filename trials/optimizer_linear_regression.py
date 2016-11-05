import tensorflow as tf
import numpy as np
from time import time
from utils import graph

'''
    Basic test to see the difference between AdamOptimizer and GradientDescendOptimizers

    Try to find W such as Y ~ X*W
    In this case, W should be ~ 2.

    They return the exact same results on simple datasets

    GradientDescentOptimizer results after 20 runs. Time: 42.3339672089
    AdamOptimizer's results after 20 runs. Time: 63.8974962234

    As expected, Adam takes longer to run because it requires more computation to be performed
    for each parameter in each training step (to maintain the moving averages and variance,
    and calculate the scaled gradient); and more state to be retained for each parameter
    (approximately tripling the size of the model to store the average and variance for each parameter)

'''

N = 10 # the number of iterations you want to train
EXPS = 20 # the number of times you want to run the experiments
weights1 = []
weights2 = []

start = time()

X = np.linspace(-1, 1, 101)
x_placeholder = tf.placeholder("float")
y_placeholder = tf.placeholder("float")

for _ in range(EXPS):
    Y = 2 * X + np.random.randn(*X.shape) * 0.3 # create random Y values with some random noise
    W1 = tf.Variable(0.0, name="weights") # the weight matrix. In this case, it's a scalar. Initialize value to 0
    W2 = tf.Variable(0.0, name="weights")

    y_pred1 = tf.mul(x_placeholder, W1) # predict y based on the weight W
    y_pred2 = tf.mul(x_placeholder, W2) # predict y based on the weight W

    loss1 = tf.square(y_placeholder - y_pred1) # use the classic loss squared for the loss function.
    loss2 = tf.square(y_placeholder - y_pred2)

    train_op1 = tf.train.AdamOptimizer(0.01).minimize(loss1) # add an optimizer to minize loss
    train_op2 = tf.train.AdamOptimizer(0.01).minimize(loss2)

    with tf.Session() as sess:
        tf.initialize_all_variables().run() # initialize variables (in this case just W)

        for _ in range(N): # number of iterations
            for (x, y) in zip(X, Y):
                sess.run(train_op1, feed_dict={x_placeholder: x, y_placeholder: y})
                sess.run(train_op2, feed_dict={x_placeholder: x, y_placeholder: y})

        weights1.append(sess.run(W1))
        weights2.append(sess.run(W2))

print len(weights1)
graph([i for i in range(EXPS)], "Epochs", weights1, "Adam", weights2, "GD", "Predicted W")



