import tensorflow as tf
'''
    A simple program to understand how to run a graph
    tf.Session.run(fetches, feed_dict=None, options=None, run_metadata=None)
    fetches: the values you want to return
'''

a = tf.placeholder("float")
b = tf.placeholder("float")

x = tf.constant(10)

y1 = tf.mul(a, b)
y2 = tf.add(a, b)

with tf.Session() as sess:
    x = sess.run(x)
    print x

    result = sess.run(y1, feed_dict={a: 2, b: 8})
    print "Product is %f" %result

    result = sess.run([y1, y2], feed_dict={a: 3, b: 10})
    # result is a list that consists of 2 elements: _product and _sum
    print "Product is %f. Sum is %f." %(result[0], result[1])

# if you only write "x = sess.run(y)" without feed_dict,
# the error will be "TypeError: run() takes at least 2 arguments (2 given)"
# which is very misleading