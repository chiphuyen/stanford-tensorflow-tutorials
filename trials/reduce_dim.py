import tensorflow as tf
import numpy as np

def length(data):
    used = tf.sign(tf.reduce_max(tf.abs(data), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length

if __name__ == "__main__":
    data = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]) # a random tensor of 3 dimensions
    print "shape", data.get_shape()
    l = length(data)
    print l



