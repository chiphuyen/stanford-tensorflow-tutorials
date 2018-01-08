def neural_gpu(features, hparams, name=None):
  """The core Neural GPU."""
  with tf.variable_scope(name, "neural_gpu"):
    inputs = features["inputs"]
    emb_inputs = common_layers.embedding(
        inputs, hparams.vocab_size, hparams.hidden_size)

    def step(state, inp):
      x = tf.nn.dropout(state, 1.0 - hparams.dropout)
      for layer in xrange(hparams.num_hidden_layers):
        x = common_layers.conv_gru(
            x, hparams.kernel_size, hparams.hidden_size, name="cgru_%d" % layer)
      return tf.where(inp == 0, state, x)  # No-op where inp is just padding=0.

    final = tf.foldl(step, tf.transpose(inputs, [1, 0]),
                     initializer=emb_inputs,
                     parallel_iterations=1, swap_memory=True)
    return common_layers.conv(final, hparams.vocab_size, 3, padding="same")


def mixed_curriculum(inputs, hparams):
  """Mixed curriculum: skip short sequences, but only with some probability."""
  with tf.name_scope("mixed_curriculum"):
    inputs_length = tf.to_float(tf.shape(inputs)[1])
    used_length = tf.cond(tf.less(tf.random_uniform([]),
                                  hparams.curriculum_mixing_probability),
                          lambda: tf.constant(0.0),
                          lambda: inputs_length)
    step = tf.to_float(tf.contrib.framework.get_global_step())
    relative_step = step / hparams.curriculum_lengths_per_step
    return used_length - hparams.curriculum_min_length > relative_step


def neural_gpu_curriculum(features, hparams, mode):
  """The Neural GPU model with curriculum."""
  with tf.name_scope("neural_gpu_with_curriculum"):
    inputs = features["inputs"]
    is_training = mode == tf.contrib.learn.ModeKeys.TRAIN
    should_skip = tf.logical_and(is_training, mixed_curriculum(inputs, hparams))
    final_shape = tf.concat([tf.shape(inputs),
                             tf.constant([hparams.vocab_size])], axis=0)
    outputs = tf.cond(should_skip,
                      lambda: tf.zeros(final_shape),
                      lambda: neural_gpu(features, hparams))
    return outputs, should_skip


def basic_params1():
  """A set of basic hyperparameters."""
  return tf.HParams(batch_size=32,
                    num_hidden_layers=4,
                    kernel_size=3,
                    hidden_size=64,
                    vocab_size=256,
                    dropout=0.2,
                    clip_grad_norm=2.0,
                    initializer="orthogonal",
                    initializer_gain=1.5,
                    label_smoothing=0.1,
                    optimizer="Adam",
                    optimizer_adam_epsilon=1e-4,
                    optimizer_momentum_momentum=0.9,
                    max_train_length=512,
                    learning_rate_decay_scheme="none",
                    learning_rate_warmup_steps=100,
                    learning_rate=0.1)


def curriculum_params1():
  """Set of hyperparameters with curriculum settings."""
  hparams = common_hparams.basic_params1()
  hparams.add_hparam("curriculum_mixing_probability", 0.1)
  hparams.add_hparam("curriculum_lengths_per_step", 1000.0)
  hparams.add_hparam("curriculum_min_length", 10)
  return hparams
