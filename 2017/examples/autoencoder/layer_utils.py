import tensorflow as tf

def get_deconv2d_output_dims(input_dims, filter_dims, stride_dims, padding):
    # Returns the height and width of the output of a deconvolution layer.
    batch_size, input_h, input_w, num_channels_in = input_dims
    filter_h, filter_w, num_channels_out  = filter_dims
    stride_h, stride_w = stride_dims

    # Compute the height in the output, based on the padding.
    if padding == 'SAME':
      out_h = input_h * stride_h
    elif padding == 'VALID':
      out_h = (input_h - 1) * stride_h + filter_h

    # Compute the width in the output, based on the padding.
    if padding == 'SAME':
      out_w = input_w * stride_w
    elif padding == 'VALID':
      out_w = (input_w - 1) * stride_w + filter_w

    return [batch_size, out_h, out_w, num_channels_out]
