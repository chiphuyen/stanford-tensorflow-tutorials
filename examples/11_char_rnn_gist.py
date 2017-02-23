""" A clean, no_frills character-level generative language model.
Created by Danijar Hafner, edited by Chip Huyen
for the class CS 20SI: "TensorFlow for Deep Learning Research"

Based on Andrej Karpathy's blog: 
http://karpathy.github.io/2015/05/21/rnn-effectiveness/
"""
from __future__ import print_function
import os
import time

import tensorflow as tf

DATA_PATH = '../data/arvix_abstracts.txt'
HIDDEN_SIZE = 200
BATCH_SIZE = 64
NUM_STEPS = 50
SKIP_STEP = 40
TEMPRATURE = 0.7
LR = 0.003
LEN_GENERATED = 300

def vocab_encode(text, vocab):
    return [vocab.index(x) + 1 for x in text if x in vocab]

def vocab_decode(array, vocab):
    return ''.join([vocab[x - 1] for x in array])

def read_data(filename, vocab, window=NUM_STEPS, overlap=NUM_STEPS/2):
    for text in open(filename):
        text = vocab_encode(text, vocab)
        for start in range(0, len(text) - window, overlap):
            chunk = text[start: start + window]
            chunk += [0] * (window - len(chunk))
            yield chunk

def read_batch(stream, batch_size=BATCH_SIZE):
    batch = []
    for element in stream:
        batch.append(element)
        if len(batch) == batch_size:
            yield batch
            batch = []
    yield batch

def create_rnn(seq, hidden_size=HIDDEN_SIZE):
    cell = tf.nn.rnn_cell.GRUCell(hidden_size)
    in_state = tf.placeholder_with_default(
            cell.zero_state(tf.shape(seq)[0], tf.float32), [None, hidden_size])
    # this line to calculate the real length of seq
    # all seq are padded to be of the same length which is NUM_STEPS
    length = tf.reduce_sum(tf.reduce_max(tf.sign(seq), 2), 1)
    output, out_state = tf.nn.dynamic_rnn(cell, seq, length, in_state)
    return output, in_state, out_state

def create_model(seq, temp, vocab, hidden=HIDDEN_SIZE):
    seq = tf.one_hot(seq, len(vocab))
    output, in_state, out_state = create_rnn(seq, hidden)
    # fully_connected is syntactic sugar for tf.matmul(w, output) + b
    # it will create w and b for us
    logits = tf.contrib.layers.fully_connected(output, len(vocab), None)
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits[:, :-1], seq[:, 1:]))
    # sample the next character from Maxwell-Boltzmann Distribution with temperature temp
    # it works equally well without tf.exp
    sample = tf.multinomial(tf.exp(logits[:, -1] / temp), 1)[:, 0] 
    return loss, sample, in_state, out_state

def training(vocab, seq, loss, optimizer, global_step, temp, sample, in_state, out_state):
    saver = tf.train.Saver()
    start = time.time()
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('graphs/gist', sess.graph)
        sess.run(tf.global_variables_initializer())
        
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/arvix/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        
        iteration = global_step.eval()
        for batch in read_batch(read_data(DATA_PATH, vocab)):
            batch_loss, _ = sess.run([loss, optimizer], {seq: batch})
            if (iteration + 1) % SKIP_STEP == 0:
                print('Iter {}. \n    Loss {}. Time {}'.format(iteration, batch_loss, time.time() - start))
                online_intference(sess, vocab, seq, sample, temp, in_state, out_state)
                start = time.time()
                saver.save(sess, 'checkpoints/arvix/char-rnn', iteration)
            iteration += 1

def online_intference(sess, vocab, seq, sample, temp, in_state, out_state, seed='T'):
    """ Generate sequence one character at a time, based on the previous character
    """
    sentence = seed
    state = None
    for _ in range(LEN_GENERATED):
        batch = [vocab_encode(sentence[-1], vocab)]
        feed = {seq: batch, temp: TEMPRATURE}
        # for the first decoder step, the state is None
        if state is not None:
            feed.update({in_state: state})
        index, state = sess.run([sample, out_state], feed)
        sentence += vocab_decode(index, vocab)
    print(sentence)

def main():
    vocab = (
            " $%'()+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "\\^_abcdefghijklmnopqrstuvwxyz{|}")
    seq = tf.placeholder(tf.int32, [None, None])
    temp = tf.placeholder(tf.float32)
    loss, sample, in_state, out_state = create_model(seq, temp, vocab)
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    optimizer = tf.train.AdamOptimizer(LR).minimize(loss, global_step=global_step)
    training(vocab, seq, loss, optimizer, global_step, temp, sample, in_state, out_state)
    
if __name__ == '__main__':
    main()