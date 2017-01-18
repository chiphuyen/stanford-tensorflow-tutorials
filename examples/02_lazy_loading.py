"""
Example to demonstrate how the graph definition gets
bloated because of lazy loading
"""
import tensorflow as tf 

# normal loading
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
z = tf.add(x, y)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter('./my_graph/l2', sess.graph)
	for _ in range(10):
		sess.run(z)
	print tf.get_default_graph().as_graph_def()
	writer.close()

# lazy loading
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter('./my_graph/l2', sess.graph)
	for _ in range(10):
		sess.run(tf.add(x, y))
	print tf.get_default_graph().as_graph_def()
	writer.close()