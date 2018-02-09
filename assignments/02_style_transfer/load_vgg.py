""" Load VGGNet weights needed for the implementation in TensorFlow
of the paper A Neural Algorithm of Artistic Style (Gatys et al., 2016) 

Created by Chip Huyen (chiphuyen@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu

For more details, please read the assignment handout:

"""
import numpy as np
import scipy.io
import tensorflow as tf

import utils

# VGG-19 parameters file
VGG_DOWNLOAD_LINK = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat'
VGG_FILENAME = 'imagenet-vgg-verydeep-19.mat'
EXPECTED_BYTES = 534904783

class VGG(object):
    def __init__(self, input_img):
        utils.download(VGG_DOWNLOAD_LINK, VGG_FILENAME, EXPECTED_BYTES)
        self.vgg_layers = scipy.io.loadmat(VGG_FILENAME)['layers']
        self.input_img = input_img
        self.mean_pixels = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

    def _weights(self, layer_idx, expected_layer_name):
        """ Return the weights and biases at layer_idx already trained by VGG
        """
        W = self.vgg_layers[0][layer_idx][0][0][2][0][0]
        b = self.vgg_layers[0][layer_idx][0][0][2][0][1]
        layer_name = self.vgg_layers[0][layer_idx][0][0][0][0]
        assert layer_name == expected_layer_name
        return W, b.reshape(b.size)

    def conv2d_relu(self, prev_layer, layer_idx, layer_name):
        """ Create a convolution layer with RELU using the weights and
        biases extracted from the VGG model at 'layer_idx'. You should use
        the function _weights() defined above to extract weights and biases.

        _weights() returns numpy arrays, so you have to convert them to TF tensors.

        Don't forget to apply relu to the output from the convolution.
        Inputs:
            prev_layer: the output tensor from the previous layer
            layer_idx: the index to current layer in vgg_layers
            layer_name: the string that is the name of the current layer.
                        It's used to specify variable_scope.
        Hint for choosing strides size: 
            for small images, you probably don't want to skip any pixel
        """
        ###############################
        ## TO DO
        out = None
        ###############################
        setattr(self, layer_name, out)

    def avgpool(self, prev_layer, layer_name):
        """ Create the average pooling layer. The paper suggests that 
        average pooling works better than max pooling.
        
        Input:
            prev_layer: the output tensor from the previous layer
            layer_name: the string that you want to name the layer.
                        It's used to specify variable_scope.

        Hint for choosing strides and kszie: choose what you feel appropriate
        """
        ###############################
        ## TO DO
        out = None
        ###############################
        setattr(self, layer_name, out)

    def load(self):
        self.conv2d_relu(self.input_img, 0, 'conv1_1')
        self.conv2d_relu(self.conv1_1, 2, 'conv1_2')
        self.avgpool(self.conv1_2, 'avgpool1')
        self.conv2d_relu(self.avgpool1, 5, 'conv2_1')
        self.conv2d_relu(self.conv2_1, 7, 'conv2_2')
        self.avgpool(self.conv2_2, 'avgpool2')
        self.conv2d_relu(self.avgpool2, 10, 'conv3_1')
        self.conv2d_relu(self.conv3_1, 12, 'conv3_2')
        self.conv2d_relu(self.conv3_2, 14, 'conv3_3')
        self.conv2d_relu(self.conv3_3, 16, 'conv3_4')
        self.avgpool(self.conv3_4, 'avgpool3')
        self.conv2d_relu(self.avgpool3, 19, 'conv4_1')
        self.conv2d_relu(self.conv4_1, 21, 'conv4_2')
        self.conv2d_relu(self.conv4_2, 23, 'conv4_3')
        self.conv2d_relu(self.conv4_3, 25, 'conv4_4')
        self.avgpool(self.conv4_4, 'avgpool4')
        self.conv2d_relu(self.avgpool4, 28, 'conv5_1')
        self.conv2d_relu(self.conv5_1, 30, 'conv5_2')
        self.conv2d_relu(self.conv5_2, 32, 'conv5_3')
        self.conv2d_relu(self.conv5_3, 34, 'conv5_4')
        self.avgpool(self.conv5_4, 'avgpool5')