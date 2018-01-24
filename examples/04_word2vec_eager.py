""" starter code for word2vec skip-gram model with NCE loss
Eager execution
CS 20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Chip Huyen (chiphuyen@cs.stanford.edu) & Akshay Agrawal (akshayka@cs.stanford.edu)
Lecture 04
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

import utils
import word2vec_utils

tfe.enable_eager_execution()

# Model hyperparameters
VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128            # dimension of the word embedding vectors
SKIP_WINDOW = 1             # the context window
NUM_SAMPLED = 64            # number of negative examples to sample
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 100000
VISUAL_FLD = 'visualization'
SKIP_STEP = 5000

# Parameters for downloading data
DOWNLOAD_URL = 'http://mattmahoney.net/dc/text8.zip'
EXPECTED_BYTES = 31344016

class Word2Vec(object):
  def __init__(self, vocab_size, embed_size, num_sampled=NUM_SAMPLED):
    self.vocab_size = vocab_size
    self.num_sampled = num_sampled
    self.embed_matrix = tfe.Variable(tf.random_uniform(
                                      [vocab_size, embed_size]))
    self.nce_weight = tfe.Variable(tf.truncated_normal(
                                    [vocab_size, embed_size],
                                    stddev=1.0 / (embed_size ** 0.5)))
    self.nce_bias = tfe.Variable(tf.zeros([vocab_size]))

  def compute_loss(self, center_words, target_words):
    """Computes the forward pass of word2vec with the NCE loss.""" 
    embed = tf.nn.embedding_lookup(self.embed_matrix, center_words)
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=self.nce_weight, 
                                        biases=self.nce_bias, 
                                        labels=target_words, 
                                        inputs=embed, 
                                        num_sampled=self.num_sampled, 
                                        num_classes=self.vocab_size))
    return loss


def gen():
  yield from word2vec_utils.batch_gen(DOWNLOAD_URL, EXPECTED_BYTES,
                                      VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW,
                                      VISUAL_FLD)

def main():
  dataset = tf.data.Dataset.from_generator(gen, (tf.int32, tf.int32),
                              (tf.TensorShape([BATCH_SIZE]),
                              tf.TensorShape([BATCH_SIZE, 1])))
  optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
  model = Word2Vec(vocab_size=VOCAB_SIZE, embed_size=EMBED_SIZE)
  grad_fn = tfe.implicit_value_and_gradients(model.compute_loss)
  total_loss = 0.0  # for average loss in the last SKIP_STEP steps
  num_train_steps = 0
  while num_train_steps < NUM_TRAIN_STEPS:
    for center_words, target_words in tfe.Iterator(dataset):
      if num_train_steps >= NUM_TRAIN_STEPS:
        break
      loss_batch, grads = grad_fn(center_words, target_words)
      total_loss += loss_batch
      optimizer.apply_gradients(grads)
      if (num_train_steps + 1) % SKIP_STEP == 0:
        print('Average loss at step {}: {:5.1f}'.format(
                num_train_steps, total_loss / SKIP_STEP))
        total_loss = 0.0
      num_train_steps += 1


if __name__ == '__main__':
    main()
