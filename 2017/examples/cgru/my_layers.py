def saturating_sigmoid(x):
  """Saturating sigmoid: 1.2 * sigmoid(x) - 0.1 cut to [0, 1]."""
  with tf.name_scope("saturating_sigmoid", [x]):
    y = tf.sigmoid(x)
    return tf.minimum(1.0, tf.maximum(0.0, 1.2 * y - 0.1))


def embedding(x, vocab_size, dense_size, name=None, reuse=None):
  """Embed x of type int64 into dense vectors, reducing to max 4 dimensions."""
  with tf.variable_scope(name, default_name="embedding",
                         values=[x], reuse=reuse):
    embedding_var = tf.get_variable("kernel", [vocab_size, dense_size])
    return tf.gather(embedding_var, x)


def conv_gru(x, kernel_size, filters, padding="same", dilation_rate=1,
             name=None, reuse=None):
  """Convolutional GRU in 1 dimension."""
  # Let's make a shorthand for conv call first.
  def do_conv(args, name, bias_start, padding):
    return tf.layers.conv1d(args, filters, kernel_size,
                padding=padding, dilation_rate=dilation_rate,
                bias_initializer=tf.constant_initializer(bias_start), name=name)
  # Here comes the GRU gate.
  with tf.variable_scope(name, default_name="conv_gru",
                         values=[x], reuse=reuse):
    reset = saturating_sigmoid(do_conv(x, "reset", 1.0, padding))
    gate = saturating_sigmoid(do_conv(x, "gate", 1.0, padding))
    candidate = tf.tanh(do_conv(reset * x, "candidate", 0.0, padding))
    return gate * x + (1 - gate) * candidate
