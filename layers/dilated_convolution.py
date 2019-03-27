# dilated_convolution.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras.layers import Concatenate, Conv2D


def dilated_conv2d(features, filters, kernel_size, padding):
    with K.name_scope("dilated_conv2d"):
        first = Conv2D(filters=filters, kernel_size=kernel_size, dilation_rate=1, padding=padding)(features)
        second = Conv2D(filters=filters, kernel_size=kernel_size, dilation_rate=2, padding=padding)(features)
        third = Conv2D(filters=filters, kernel_size=kernel_size, dilation_rate=4, padding=padding)(features)
        fourth = Conv2D(filters=filters, kernel_size=kernel_size, dilation_rate=8, padding=padding)(features)

        return Concatenate()([first, second, third, fourth])
