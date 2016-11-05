import tensorflow as tf

a = tf.constant(5)
b = tf.constant(10)
c = tf.add(a, b)
sess = tf.Session()
print sess.run(c)

writer = tf.train.SummaryWriter('./my_graph', sess.graph)

writer.close()
sess.close()