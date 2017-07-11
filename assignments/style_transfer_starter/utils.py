""" Utils needed for the implementation of the paper "A Neural Algorithm of Artistic Style"
by Gatys et al. in TensorFlow.

Author: Chip Huyen (huyenn@stanford.edu)
Prepared for the class CS 20SI: "TensorFlow for Deep Learning Research"
For more details, please read the assignment handout:
http://web.stanford.edu/class/cs20si/assignments/a2.pdf
"""
from __future__ import print_function

import os

from PIL import Image, ImageOps
import numpy as np
import scipy.misc
from six.moves import urllib

def download(download_link, file_name, expected_bytes):
    """ Download the pretrained VGG-19 model if it's not already downloaded """
    if os.path.exists(file_name):
        print("VGG-19 pre-trained model ready")
        return
    print("Downloading the VGG pre-trained model. This might take a while ...")
    file_name, _ = urllib.request.urlretrieve(download_link, file_name)
    file_stat = os.stat(file_name)
    if file_stat.st_size == expected_bytes:
        print('Successfully downloaded VGG-19 pre-trained model', file_name)
    else:
        raise Exception('File ' + file_name +
                        ' might be corrupted. You should try downloading it with a browser.')

def get_resized_image(img_path, height, width, save=True):
    image = Image.open(img_path)
    # it's because PIL is column major so you have to change place of width and height
    # this is stupid, i know
    image = ImageOps.fit(image, (width, height), Image.ANTIALIAS)
    if save:
        image_dirs = img_path.split('/')
        image_dirs[-1] = 'resized_' + image_dirs[-1]
        out_path = '/'.join(image_dirs)
        if not os.path.exists(out_path):
            image.save(out_path)
    image = np.asarray(image, np.float32)
    return np.expand_dims(image, 0)

def generate_noise_image(content_image, height, width, noise_ratio=0.6):
    noise_image = np.random.uniform(-20, 20, 
                                    (1, height, width, 3)).astype(np.float32)
    return noise_image * noise_ratio + content_image * (1 - noise_ratio)

def save_image(path, image):
    # Output should add back the mean pixels we subtracted at the beginning
    image = image[0] # the image
    image = np.clip(image, 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)

def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass