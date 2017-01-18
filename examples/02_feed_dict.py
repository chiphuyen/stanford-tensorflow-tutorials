"""
Example to demonstrate the use of feed_dict
"""
import tensorflow as tf

# Example 1: feed_dict with placeholder
# create a placeholder of type float 32-bit, value is a vector of 3 elements
a = tf.placeholder(tf.float32, shape=[3])

# create a constant of type float 32-bit, value is a vector of 3 elements
b = tf.constant([5, 5, 5], tf.float32)

# use the placeholder as you would a constant
c = a + b  # short for tf.add(a, b)

with tf.Session() as sess:
	print sess.run(c) # error because a doesn’t have any value

	# feed [1, 2, 3] to placeholder a via the dict {a: [1, 2, 3]}
	# fetch value of c
	print sess.run(c, {a: [1, 2, 3]}) # >> [6. 7. 8.]


# Example 2: feed_dict with variables
a = tf.add(2, 5)
b = tf.mul(a, 3)

with tf.Session() as sess:
	# define a dictionary that says to replace the value of 'a' with 15
	replace_dict = {a: 15}

	# Run the session, passing in 'replace_dict' as the value to 'feed_dict'
	sess.run(b, feed_dict=replace_dict) # >> 45