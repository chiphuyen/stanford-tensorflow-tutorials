from __future__ import print_function

import tensorflow as tf

import config

class ChatBotModel(object):
    def __init__(self, forward_only, batch_size):
        """forward_only: if set, we do not construct the backward pass in the model.
        """
        print('Initialize new model')
        self.fw_only = forward_only
        self.batch_size = batch_size
    
    def _create_placeholders(self):
        # Feeds for inputs. It's a list of placeholders
        print('Create placeholders')
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.decoder_masks = []
        for i in xrange(config.BUCKETS[-1][0]):  # Last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                    name='encoder{}'.format(i)))
        for i in xrange(config.BUCKETS[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                    name='decoder{}'.format(i)))
            self.decoder_masks.append(tf.placeholder(tf.float32, shape=[None],
                                                    name='mask{}'.format(i)))

        # Our targets are decoder inputs shifted by one (to ignore <s> symbol)
        self.targets = [self.decoder_inputs[i + 1]
                        for i in xrange(len(self.decoder_inputs) - 1)]

    def _inference(self):
        print('Create inference')
        # If we use sampled softmax, we need an output projection.
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if config.NUM_SAMPLES > 0 and config.NUM_SAMPLES < config.DEC_VOCAB:
            w = tf.get_variable('proj_w', [config.HIDDEN_SIZE, config.DEC_VOCAB])
            b = tf.get_variable('proj_b', [config.DEC_VOCAB])
            self.output_projection = (w, b)

        def sampled_loss(inputs, labels):
            labels = tf.reshape(labels, [-1, 1])
            return tf.nn.sampled_softmax_loss(tf.transpose(w), b, inputs, labels, 
                                              config.NUM_SAMPLES, config.DEC_VOCAB)
        self.softmax_loss_function = sampled_loss

        single_cell = tf.nn.rnn_cell.GRUCell(config.HIDDEN_SIZE)
        self.cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * config.NUM_LAYERS)

    def _create_loss(self):
        print('Create loss')
        def _seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return tf.nn.seq2seq.embedding_attention_seq2seq(
                    encoder_inputs, decoder_inputs, self.cell,
                    num_encoder_symbols=config.ENC_VOCAB,
                    num_decoder_symbols=config.DEC_VOCAB,
                    embedding_size=config.HIDDEN_SIZE,
                    output_projection=self.output_projection,
                    feed_previous=do_decode)

        if self.fw_only:
            self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
                                        self.encoder_inputs, 
                                        self.decoder_inputs, 
                                        self.targets,
                                        self.decoder_masks, 
                                        config.BUCKETS, 
                                        lambda x, y: _seq2seq_f(x, y, True),
                                        softmax_loss_function=self.softmax_loss_function)
            # If we use output projection, we need to project outputs for decoding.
            if self.output_projection:
                for bucket in xrange(len(config.BUCKETS)):
                    self.outputs[bucket] = [tf.matmul(output, 
                                            self.output_projection[0]) + self.output_projection[1]
                                            for output in self.outputs[bucket]]
        else:
            self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(
                                        self.encoder_inputs, 
                                        self.decoder_inputs, 
                                        self.targets,
                                        self.decoder_masks,
                                        config.BUCKETS,
                                        lambda x, y: _seq2seq_f(x, y, False),
                                        softmax_loss_function=self.softmax_loss_function)

    def _creat_optimizer(self):
        print('Create optimizer')
        with tf.variable_scope('training') as scope:
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
            # self.lr = tf.Variable(float(config.LR), trainable=False)
            # self.lr_decay_op = self.lr.assign(self.lr * config.LR_DECAY_FACTOR)
            self.lr = config.LR

            if not self.fw_only:
                self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
                trainables = tf.trainable_variables()
                self.gradient_norms = []
                self.train_ops = []
                for bucket in xrange(len(config.BUCKETS)):
                    print(bucket)
                    clipped_grads, norm = tf.clip_by_global_norm(tf.gradients(self.losses[bucket], 
                                                                 trainables),
                                                                 config.MAX_GRAD_NORM)
                    self.gradient_norms.append(norm)
                    self.train_ops.append(self.optimizer.apply_gradients(zip(clipped_grads, trainables), 
                                                            global_step=self.global_step))


    def _create_summary(self):
        pass

    def build_graph(self):
        self._create_placeholders()
        self._inference()
        self._create_loss()
        self._creat_optimizer()
        self._create_summary()

    def run_step(self, encoder_inputs, decoder_inputs, decoder_masks, bucket_id, forward_only):
        encoder_size, decoder_size = config.BUCKETS[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                            " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                           " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(decoder_masks) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                           " %d != %d." % (len(decoder_masks), decoder_size))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in xrange(encoder_size):
            print(self.encoder_inputs[l].name)
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        # for l in xrange(decoder_size):
        #   input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
        #   input_feed[self.target_weights[l].name] = target_weights[l]

        # # Since our targets are decoder inputs shifted by one, we need one more.
        # last_target = self.decoder_inputs[decoder_size].name
        # input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # # Output feed: depends on whether we do a backward step or not.
        # if not forward_only:
        #   output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
        #                  self.gradient_norms[bucket_id],  # Gradient norm.
        #                  self.losses[bucket_id]]  # Loss for this batch.
        # else:
        #   output_feed = [self.losses[bucket_id]]  # Loss for this batch.
        #   for l in xrange(decoder_size):  # Output logits.
        #     output_feed.append(self.outputs[bucket_id][l])

        # outputs = session.run(output_feed, input_feed)
        # if not forward_only:
        #   return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
        # else:
        #   return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.
        return None, None, None
