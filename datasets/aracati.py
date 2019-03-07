# aracati.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import numpy as np
import tensorflow as tf

from scipy.misc import imread, imresize, imsave


class Aracati(object):
    def __init__(self, config):
        self.config = config
        self.idx = 0

        # Load Data
        self.son_data = np.array(sorted(glob.glob("./datasets/aracati/train/input/*.png")))
        self.sat_data = np.array(sorted(glob.glob("./datasets/aracati/train/gt/*.png")))

        self.son_data_val = np.array(sorted(glob.glob("./datasets/aracati/validation/input/*.png")))
        self.sat_data_val = np.array(sorted(glob.glob("./datasets/aracati/validation/gt/*.png")))

        if len(self.son_data) != len(self.sat_data):
            tf.logging.error("Dataset has unequal number of satellite and segmentation training images")
            raise ValueError
        elif len(self.son_data_val) != len(self.sat_data_val):
            tf.logging.error("Dataset has unequal number of satellite and segmentation validation images")
            raise ValueError
        else:
            self.num_images = len(self.son_data)
            self.num_images_val = len(self.son_data_val)

    def next_batch(self, batch_size, is_test=False):
        batch_idxs = self.num_images_val // batch_size if is_test else self.num_images // batch_size
        if self.idx == batch_idxs:
            self.idx = 0

        if is_test:
            yield [self.load_data(file, is_sonar=True) for file in self.son_data_val[self.idx * batch_size:(self.idx + 1) * batch_size]], \
                  [self.load_data(file, is_sonar=False) for file in self.sat_data_val[self.idx * batch_size:(self.idx + 1) * batch_size]]
        else:
            yield [self.load_data(file, is_sonar=True) for file in self.son_data[self.idx * batch_size:(self.idx + 1) * batch_size]], \
                  [self.load_data(file, is_sonar=False) for file in self.sat_data[self.idx * batch_size:(self.idx + 1) * batch_size]]

    @staticmethod
    def load_data(path, width=256, height=128, is_sonar=False):
        data = imread(path).astype(np.float)
        data = imresize(data, [height, width])
        data = data / 255.0

        if np.ndim(data) == 2:
            data = np.expand_dims(data, axis=2)

        if is_sonar:
            return data[:, :, :1]
        else:
            return data[:, :, :3]

    @staticmethod
    def save_data(path, idx, data):
        img = data[0]
        imsave("{}/test_{:05d}.png".format(path, idx), img)
