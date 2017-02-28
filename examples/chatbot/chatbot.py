""" 
Based on Google Translate Tensorflow model 
https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/
"""
from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import time

import tensorflow as tf

from model import ChatBotModel
import config
import data

def _get_random_bucket(train_buckets_scale):
    rand = random.random()
    return min([i for i in xrange(len(train_buckets_scale))
                if train_buckets_scale[i] > rand])

def train():
    """ in train mode, we need to create the backward path
    """
    data_buckets = data.load_data('train_ids.enc', 'train_ids.dec')
    train_bucket_sizes = [len(data_buckets[b]) for b in xrange(len(config.BUCKETS))]
    print("Number of samples in each bucket:\n", train_bucket_sizes)
    train_total_size = sum(train_bucket_sizes)
    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]
    print("Bucket scale:\n", train_buckets_scale)

    model = ChatBotModel(False, config.BATCH_SIZE)
    model.build_graph()

    saver = tf.train.Saver()
    initial_step = 0

    # with tf.Session() as sess:
    #     print('Running session')
    #     sess.run(tf.global_variables_initializer())
    #     ckpt = tf.train.get_checkpoint_state(os.path.dirname(config.CPT_PATH + '/checkpoint'))
    #     if ckpt and ckpt.model_checkpoint_path:
    #         print("Loading parameters for the Chatbot")
    #         saver.restore(sess, ckpt.model_checkpoint_path)
    #     else:
    #         print("Initializing fresh parameters for the Chatbot")
        
    #     iteration = model.global_step.eval()
        
    #     while True:
    #         bucket_id = _get_random_bucket(train_buckets_scale)
    #         # time step x batch size
    #         encoder_inputs, decoder_inputs, decoder_masks = data.get_batch(data_buckets[bucket_id], 
    #                                                                      bucket_id,
    #                                                                      batch_size=config.BATCH_SIZE)
    #         _, step_loss, _ = model.run_step(encoder_inputs, decoder_inputs, decoder_masks, bucket_id, False)

    #         iteration += 1

    #         if iteration % config.SKIP_STEP == 0:
    #             # eval on test set
    #             encoder_inputs, decoder_inputs, decoder_masks = data.get_batch(data_buckets[bucket_id], 
    #                                                                      bucket_id,
    #                                                                      batch_size=config.BATCH_SIZE)
    #             _, step_loss, _ = model.run_step(encoder_inputs, decoder_inputs, decoder_masks, bucket_id, True)
    #         break

def chat():
    """ in test mode, we don't to create the backward path
    """
    model = ChatBotModel(True, batch_size=1)
    model.build_graph()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices={'train', 'chat'},
                        default='train', help="mode. if not specified, it's in the train mode")
    args = parser.parse_args()

    # create checkpoints folder if there isn't one already
    data.make_dir(config.CPT_PATH)

    if args.mode == 'train':
        train()
    elif args.mode == 'chat':
        chat()

if __name__ == '__main__':
    main()