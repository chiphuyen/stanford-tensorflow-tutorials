""" Load VGGNet weights needed for the implementation of the paper 
"A Neural Algorithm of Artistic Style" by Gatys et al. in TensorFlow.

Author: Chip Huyen (huyenn@stanford.edu)
Prepared for the class CS 20SI: "TensorFlow for Deep Learning Research"
For more details, please read the assignment handout:
http://web.stanford.edu/class/cs20si/assignments/a2.pdf
"""

import numpy as np
import tensorflow as tf
import scipy.io

def _weights(vgg_layers, layer, expected_layer_name):
    """ Return the weights and biases already trained by VGG
    """
    W = vgg_layers[0][layer][0][0][2][0][0]
    b = vgg_layers[0][layer][0][0][2][0][1]
    layer_name = vgg_layers[0][layer][0][0][0][0]
    assert layer_name == expected_layer_name
    return W, b.reshape(b.size)

def _conv2d_relu(vgg_layers, prev_layer, layer, layer_name):
    """ Return the Conv2D layer with RELU using the weights, biases from the VGG
    model at 'layer'.
    Inputs:
        vgg_layers: holding all the layers of VGGNet
        prev_layer: the output tensor from the previous layer
        layer: the index to current layer in vgg_layers
        layer_name: the string that is the name of the current layer.
                    It's used to specify variable_scope.

    Output:
        relu applied on the convolution.

    Note that you first need to obtain W and b from vgg-layers using the function
    _weights() defined above.
    W and b returned from _weights() are numpy arrays, so you have
    to convert them to TF tensors using tf.constant.
    Note that you'll have to do apply relu on the convolution.
    Hint for choosing strides size: 
        for small images, you probably don't want to skip any pixel
    """
    with tf.variable_scope(layer_name) as scope:
        W, b = _weights(vgg_layers, layer, layer_name)
        W = tf.constant(W, name='weights')
        b = tf.constant(b, name='bias')
        conv2d = tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv2d + b)

def _avgpool(prev_layer):
    """ Return the average pooling layer. The paper suggests that average pooling
    actually works better than max pooling.
    Input:
        prev_layer: the output tensor from the previous layer

    Output:
        the output of the tf.nn.avg_pool() function.
    Hint for choosing strides and kszie: choose what you feel appropriate
    """
    return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], 
                          padding='SAME', name='avg_pool_' + prev_layer_name)

def load_vgg(path, input_image):
    """ Load VGG into a TensorFlow model.
    Use a dictionary to hold the model instead of using a Python class
    """
    vgg = scipy.io.loadmat(path)
    vgg_layers = vgg['layers']

    graph = {} 
    graph['conv1_1']  = _conv2d_relu(vgg_layers, input_image, 0, 'conv1_1')
    graph['conv1_2']  = _conv2d_relu(vgg_layers, graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = _avgpool(graph['conv1_2'])
    graph['conv2_1']  = _conv2d_relu(vgg_layers, graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2']  = _conv2d_relu(vgg_layers, graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2'] = _avgpool(graph['conv2_2'])
    graph['conv3_1']  = _conv2d_relu(vgg_layers, graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2']  = _conv2d_relu(vgg_layers, graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3']  = _conv2d_relu(vgg_layers, graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4']  = _conv2d_relu(vgg_layers, graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = _avgpool(graph['conv3_4'])
    graph['conv4_1']  = _conv2d_relu(vgg_layers, graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2']  = _conv2d_relu(vgg_layers, graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3']  = _conv2d_relu(vgg_layers, graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4']  = _conv2d_relu(vgg_layers, graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = _avgpool(graph['conv4_4'])
    graph['conv5_1']  = _conv2d_relu(vgg_layers, graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2']  = _conv2d_relu(vgg_layers, graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3']  = _conv2d_relu(vgg_layers, graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4']  = _conv2d_relu(vgg_layers, graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5'] = _avgpool(graph['conv5_4'])
    
    return graph