def examples_queue(data_sources, data_fields_to_features, training,
                   data_items_to_decoders=None, data_items_to_decode=None):
  """Contruct a queue of training or evaluation examples.

  This function will create a reader from files given by data_sources,
  then enqueue the tf.Examples from these files, shuffling if training
  is true, and finally parse these tf.Examples to tensors.

  The dictionary data_fields_to_features for an image dataset can be this:

  data_fields_to_features = {
    'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
    'image/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
    'image/class/label': tf.FixedLenFeature(
        [1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
  }

  and for a simple algorithmic dataset with variable-length data it is this:

  data_fields_to_features = {
    'inputs': tf.VarLenFeature(tf.int64),
    'targets': tf.VarLenFeature(tf.int64),
  }

  The data_items_to_decoders dictionary argument can be left as None if there
  is no decoding to be performed. But, e.g. for images, it should be set so that
  the images are decoded from the features, e.g., like this for MNIST:

  data_items_to_decoders = {
    'image': tfexample_decoder.Image(
      image_key = 'image/encoded',
      format_key = 'image/format',
      shape=[28, 28],
      channels=1),
    'label': tfexample_decoder.Tensor('image/class/label'),
  }

  These arguments are compatible with the use of tf.contrib.slim.data module,
  see there for more documentation.

  Args:
    data_sources: a list or tuple of sources from which the data will be read,
      for example [/path/to/train@128, /path/to/train2*, /tmp/.../train3*]
    data_fields_to_features: a dictionary from data fields in the data sources
      to features, such as tf.VarLenFeature(tf.int64), see above for examples.
    training: a Boolean, whether to read for training or evaluation.
    data_items_to_decoders: a dictionary mapping data items (that will be
      in the returned result) to decoders that will decode them using features
      defined in data_fields_to_features; see above for examples. By default
      (if this is None), we grab the tensor from every feature.
    data_items_to_decode: a subset of data items that will be decoded;
      by default (if this is None), we decode all items.

  Returns:
    A dictionary mapping each data_field to a corresponding 1D int64 tensor
    read from the created queue.

  Raises:
    ValueError: if no files are found with the provided data_prefix or no data
      fields were provided.
  """
  with tf.name_scope("examples_queue"):
    # Read serialized examples using slim parallel_reader.
    _, example_serialized = tf.contrib.slim.parallel_reader.parallel_read(
        data_sources, tf.TFRecordReader, shuffle=training,
        num_readers=4 if training else 1)

    if data_items_to_decoders is None:
      data_items_to_decoders = {
          field: tf.contrib.slim.tfexample_decoder.Tensor(field)
          for field in data_fields_to_features
      }

    decoder = tf.contrib.slim.tfexample_decoder.TFExampleDecoder(
        data_fields_to_features, data_items_to_decoders)

    if data_items_to_decode is None:
      data_items_to_decode = data_items_to_decoders.keys()

    decoded = decoder.decode(example_serialized, items=data_items_to_decode)
    return {field: tensor
            for (field, tensor) in zip(data_items_to_decode, decoded)}


def batch_examples(examples, batch_size, bucket_boundaries=None):
  """Given a queue of examples, create batches of examples with similar lengths.

  We assume that examples is a dictionary with string keys and tensor values,
  possibly coming from a queue, e.g., constructed by examples_queue above.
  Each tensor in examples is assumed to be 1D. We will put tensors of similar
  length into batches togeter. We return a dictionary with the same keys as
  examples, and with values being batches of size batch_size. If elements have
  different lengths, they are padded with 0s. This function is based on
  tf.contrib.training.bucket_by_sequence_length so see there for details.

  For example, if examples is a queue containing [1, 2, 3] and [4], then
  this function with batch_size=2 will return a batch [[1, 2, 3], [4, 0, 0]].

  Args:
    examples: a dictionary with string keys and 1D tensor values.
    batch_size: a python integer or a scalar int32 tensor.
    bucket_boundaries: a list of integers for the boundaries that will be
      used for bucketing; see tf.contrib.training.bucket_by_sequence_length
      for more details; if None, we create a default set of buckets.

  Returns:
    A dictionary with the same keys as examples and with values being batches
    of examples padded with 0s, i.e., [batch_size x length] tensors.
  """
  # Create default buckets if none were provided.
  if bucket_boundaries is None:
    # Small buckets -- go in steps of 8 until 64.
    small_buckets = [8 * (i + 1) for i in xrange(8)]
    # Medium buckets -- go in steps of 32 until 256.
    medium_buckets = [32 * (i + 3) for i in xrange(6)]
    # Large buckets -- go in steps of 128 until maximum of 1024.
    large_buckets = [128 * (i + 3) for i in xrange(6)]
    # By default use the above 20 bucket boundaries (21 queues in total).
    bucket_boundaries = small_buckets + medium_buckets + large_buckets
  with tf.name_scope("batch_examples"):
    # The queue to bucket on will be chosen based on maximum length.
    max_length = 0
    for v in examples.values():  # We assume 0-th dimension is the length.
      max_length = tf.maximum(max_length, tf.shape(v)[0])
    (_, outputs) = tf.contrib.training.bucket_by_sequence_length(
        max_length, examples, batch_size, bucket_boundaries,
        capacity=2 * batch_size, dynamic_pad=True)
    return outputs
