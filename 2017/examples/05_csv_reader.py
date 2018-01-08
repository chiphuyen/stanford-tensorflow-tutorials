""" Some people tried to use TextLineReader for the assignment 1
but seem to have problems getting it work, so here is a short 
script demonstrating the use of CSV reader on the heart dataset.
Note that the heart dataset is originally in txt so I first
converted it to csv to take advantage of the already laid out columns.

You can download heart.csv in the data folder.
Author: Chip Huyen
Prepared for the class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import sys
sys.path.append('..')

import tensorflow as tf

DATA_PATH = 'data/heart.csv'
BATCH_SIZE = 2
N_FEATURES = 9

def batch_generator(filenames):
    """ filenames is the list of files you want to read from. 
    In this case, it contains only heart.csv
    """
    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.TextLineReader(skip_header_lines=1) # skip the first line in the file
    _, value = reader.read(filename_queue)

    # record_defaults are the default values in case some of our columns are empty
    # This is also to tell tensorflow the format of our data (the type of the decode result)
    # for this dataset, out of 9 feature columns, 
    # 8 of them are floats (some are integers, but to make our features homogenous, 
    # we consider them floats), and 1 is string (at position 5)
    # the last column corresponds to the lable is an integer

    record_defaults = [[1.0] for _ in range(N_FEATURES)]
    record_defaults[4] = ['']
    record_defaults.append([1])

    # read in the 10 rows of data
    content = tf.decode_csv(value, record_defaults=record_defaults) 

    # convert the 5th column (present/absent) to the binary value 0 and 1
    content[4] = tf.cond(tf.equal(content[4], tf.constant('Present')), lambda: tf.constant(1.0), lambda: tf.constant(0.0))

    # pack all 9 features into a tensor
    features = tf.stack(content[:N_FEATURES])

    # assign the last column to label
    label = content[-1]

    # minimum number elements in the queue after a dequeue, used to ensure 
    # that the samples are sufficiently mixed
    # I think 10 times the BATCH_SIZE is sufficient
    min_after_dequeue = 10 * BATCH_SIZE

    # the maximum number of elements in the queue
    capacity = 20 * BATCH_SIZE

    # shuffle the data to generate BATCH_SIZE sample pairs
    data_batch, label_batch = tf.train.shuffle_batch([features, label], batch_size=BATCH_SIZE, 
                                        capacity=capacity, min_after_dequeue=min_after_dequeue)

    return data_batch, label_batch

def generate_batches(data_batch, label_batch):
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for _ in range(10): # generate 10 batches
            features, labels = sess.run([data_batch, label_batch])
            print(features)
        coord.request_stop()
        coord.join(threads)

def main():
    data_batch, label_batch = batch_generator([DATA_PATH])
    generate_batches(data_batch, label_batch)

if __name__ == '__main__':
    main()
