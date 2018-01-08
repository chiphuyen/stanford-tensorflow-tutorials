import os
import sys
import tensorflow
import numpy as np

import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist_image_shape = [28, 28, 1]

def load_dataset():
    return input_data.read_data_sets('MNIST_data')

def get_next_batch(dataset, batch_size):
    # dataset should be mnist.(train/val/test)
    batch, _ = dataset.next_batch(batch_size)
    batch_shape = [batch_size] + mnist_image_shape
    return np.reshape(batch, batch_shape)

def visualize(_original, _reconstructions, num_visualize):
    vis_folder = './vis/'
    if not os.path.exists(vis_folder):
          os.makedirs(vis_folder)

    original = _original[:num_visualize]
    reconstructions = _reconstructions[:num_visualize]
    
    count = 1
    for (orig, rec) in zip(original, reconstructions):
        orig = np.reshape(orig, (mnist_image_shape[0],
                                 mnist_image_shape[1]))
        rec = np.reshape(rec, (mnist_image_shape[0],
                               mnist_image_shape[1]))
        f, ax = plt.subplots(1,2)
        ax[0].imshow(orig, cmap='gray')
        ax[1].imshow(rec, cmap='gray')
        plt.savefig(vis_folder + "test_%d.png" % count)
        count += 1
