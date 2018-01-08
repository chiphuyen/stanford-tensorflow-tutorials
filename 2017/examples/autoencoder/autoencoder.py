import tensorflow as tf

from layers import *

def encoder(input):
    # Create a conv network with 3 conv layers and 1 FC layer
    # Conv 1: filter: [3, 3, 1], stride: [2, 2], relu
    
    # Conv 2: filter: [3, 3, 8], stride: [2, 2], relu
    
    # Conv 3: filter: [3, 3, 8], stride: [2, 2], relu
    
    # FC: output_dim: 100, no non-linearity
    raise NotImplementedError

def decoder(input):
    # Create a deconv network with 1 FC layer and 3 deconv layers
    # FC: output dim: 128, relu
    
    # Reshape to [batch_size, 4, 4, 8]
    
    # Deconv 1: filter: [3, 3, 8], stride: [2, 2], relu
    
    # Deconv 2: filter: [8, 8, 1], stride: [2, 2], padding: valid, relu
    
    # Deconv 3: filter: [7, 7, 1], stride: [1, 1], padding: valid, sigmoid
    raise NotImplementedError

def autoencoder(input_shape):
    # Define place holder with input shape

    # Define variable scope for autoencoder
    with tf.variable_scope('autoencoder') as scope:
        # Pass input to encoder to obtain encoding
        
        # Pass encoding into decoder to obtain reconstructed image
        
        # Return input image (placeholder) and reconstructed image
        pass
