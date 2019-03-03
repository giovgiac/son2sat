# style_loss.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.activations import relu
from keras.layers import Conv2D, Dense, MaxPool2D, Reshape


def load_weights(parameters):
    with tf.Session() as session:
        weights = np.load("vgg16.npz")
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            session.run(parameters[i].assign(weights[k]))


def build_vgg(fake, real, layers):
    loss = 0.0
    parameters = []

    with tf.variable_scope("vgg16", reuse=tf.AUTO_REUSE):
        # Preprocess
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=(1, 1, 1, 3))
        fake = tf.image.resize_images(fake, size=[224, 224]) * 255.0 - mean
        real = tf.image.resize_images(real, size=[224, 224]) * 255.0 - mean

        # First Convolution
        with K.name_scope("conv11"):
            conv11 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer='zeros',
                            bias_initializer='zeros')

            # Output
            c11 = conv11(fake)
            c11 = relu(c11)

            # Ground Truth
            c11_gt = conv11(real)
            c11_gt = relu(c11_gt)

            # Add Parameters
            parameters += [conv11.kernel, conv11.bias]

        # Loss
        if "relu11" in layers:
            loss += tf.reduce_mean(tf.squared_difference(c11, c11_gt))

        # Second Convolution
        with K.name_scope("conv12"):
            conv12 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer='zeros',
                            bias_initializer='zeros')

            # Output
            c12 = conv12(c11)
            c12 = relu(c12)

            # Ground Truth
            c12_gt = conv12(c11_gt)
            c12_gt = relu(c12_gt)

            # Add Parameters
            parameters += [conv12.kernel, conv12.bias]

        # Loss
        if "relu12" in layers:
            loss += tf.reduce_mean(tf.squared_difference(c12, c12_gt))

        # First MaxPool
        with K.name_scope("pool1"):
            pool1 = MaxPool2D(pool_size=2, strides=2, padding='same')

            # Output
            p1 = pool1(c12)

            # Ground Truth
            p1_gt = pool1(c12_gt)

        # Third Convolution
        with K.name_scope("conv21"):
            conv21 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer='zeros',
                            bias_initializer='zeros')

            # Output
            c21 = conv21(p1)
            c21 = relu(c21)

            # Ground Truth
            c21_gt = conv21(p1_gt)
            c21_gt = relu(c21_gt)

            # Add Parameters
            parameters += [conv21.kernel, conv21.bias]

        # Loss
        if "relu21" in layers:
            loss += tf.reduce_mean(tf.squared_difference(c21, c21_gt))

        # Fourth Convolution
        with K.name_scope("conv22"):
            conv22 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer='zeros',
                            bias_initializer='zeros')

            # Output
            c22 = conv22(c21)
            c22 = relu(c22)

            # Ground Truth
            c22_gt = conv22(c21_gt)
            c22_gt = relu(c22_gt)

            # Add Parameters
            parameters += [conv22.kernel, conv22.bias]

        # Loss
        if "relu22" in layers:
            loss += tf.reduce_mean(tf.squared_difference(c22, c22_gt))

        # Second MaxPool
        with K.name_scope("pool2"):
            pool2 = MaxPool2D(pool_size=2, strides=2, padding='same')

            # Output
            p2 = pool2(c22)

            # Ground Truth
            p2_gt = pool2(c22_gt)

        # Fifth Convolution
        with K.name_scope("conv31"):
            conv31 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer='zeros',
                            bias_initializer='zeros')

            # Output
            c31 = conv31(p2)
            c31 = relu(c31)

            # Ground Truth
            c31_gt = conv31(p2_gt)
            c31_gt = relu(c31_gt)

            # Add Parameters
            parameters += [conv31.kernel, conv31.bias]

        # Loss
        if "relu31" in layers:
            loss += tf.reduce_mean(tf.squared_difference(c31, c31_gt))

        # Sixth Convolution
        with K.name_scope("conv32"):
            conv32 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer='zeros',
                            bias_initializer='zeros')

            # Output
            c32 = conv32(c31)
            c32 = relu(c32)

            # Ground Truth
            c32_gt = conv32(c31_gt)
            c32_gt = relu(c32_gt)

            # Add Parameters
            parameters += [conv32.kernel, conv32.bias]

        # Loss
        if "relu32" in layers:
            loss += tf.reduce_mean(tf.squared_difference(c32, c32_gt))

        # Seventh Convolution
        with K.name_scope("conv33"):
            conv33 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer='zeros',
                            bias_initializer='zeros')

            # Output
            c33 = conv33(c32)
            c33 = relu(c33)

            # Ground Truth
            c33_gt = conv33(c32_gt)
            c33_gt = relu(c33_gt)

            # Add Parameters
            parameters += [conv33.kernel, conv33.bias]

        # Loss
        if "relu33" in layers:
            loss += tf.reduce_mean(tf.squared_difference(c33, c33_gt))

        # Third MaxPool
        with K.name_scope("pool3"):
            pool3 = MaxPool2D(pool_size=2, strides=2, padding='same')

            # Output
            p3 = pool3(c33)

            # Ground Truth
            p3_gt = pool3(c33_gt)

        # Eighth Convolution
        with K.name_scope("conv41"):
            conv41 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', kernel_initializer='zeros',
                            bias_initializer='zeros')

            # Output
            c41 = conv41(p3)
            c41 = relu(c41)

            # Ground Truth
            c41_gt = conv41(p3_gt)
            c41_gt = relu(c41_gt)

            # Add Parameters
            parameters += [conv41.kernel, conv41.bias]

        # Loss
        if "relu41" in layers:
            loss += tf.reduce_mean(tf.squared_difference(c41, c41_gt))

        # Nineth Convolution
        with K.name_scope("conv42"):
            conv42 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', kernel_initializer='zeros',
                            bias_initializer='zeros')

            # Output
            c42 = conv42(c41)
            c42 = relu(c42)

            # Ground Truth
            c42_gt = conv42(c41_gt)
            c42_gt = relu(c42_gt)

            # Add Parameters
            parameters += [conv42.kernel, conv42.bias]

        # Loss
        if "relu42" in layers:
            loss += tf.reduce_mean(tf.squared_difference(c42, c42_gt))

        # Tenth Convolution
        with K.name_scope("conv43"):
            conv43 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', kernel_initializer='zeros',
                            bias_initializer='zeros')

            # Output
            c43 = conv43(c42)
            c43 = relu(c43)

            # Ground Truth
            c43_gt = conv43(c42_gt)
            c43_gt = relu(c43_gt)

            # Add Parameters
            parameters += [conv43.kernel, conv43.bias]

        # Loss
        if "relu43" in layers:
            loss += tf.reduce_mean(tf.squared_difference(c43, c43_gt))

        # Fourth MaxPool
        with K.name_scope("pool4"):
            pool4 = MaxPool2D(pool_size=2, strides=2, padding='same')

            # Output
            p4 = pool4(c43)

            # Ground Truth
            p4_gt = pool4(c43_gt)

        # Eleventh Convolution
        with K.name_scope("conv51"):
            conv51 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', kernel_initializer='zeros',
                            bias_initializer='zeros')

            # Output
            c51 = conv51(p4)
            c51 = relu(c51)

            # Ground Truth
            c51_gt = conv51(p4_gt)
            c51_gt = relu(c51_gt)

            # Add Parameters
            parameters += [conv51.kernel, conv51.bias]

        # Loss
        if "relu51" in layers:
            loss += tf.reduce_mean(tf.squared_difference(c51, c51_gt))

        # Twelfth Convolution
        with K.name_scope("conv52"):
            conv52 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', kernel_initializer='zeros',
                            bias_initializer='zeros')

            # Output
            c52 = conv52(c51)
            c52 = relu(c52)

            # Ground Truth
            c52_gt = conv52(c51_gt)
            c52_gt = relu(c52_gt)

            # Add Parameters
            parameters += [conv52.kernel, conv52.bias]

        # Loss
        if "relu52" in layers:
            loss += tf.reduce_mean(tf.squared_difference(c52, c52_gt))

        # Thirteenth Convolution
        with K.name_scope("conv53"):
            conv53 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', kernel_initializer='zeros',
                            bias_initializer='zeros')

            # Output
            c53 = conv53(c52)
            c53 = relu(c53)

            # Ground Truth
            c53_gt = conv53(c52_gt)
            c53_gt = relu(c53_gt)

            # Add Parameters
            parameters += [conv53.kernel, conv53.bias]

        # Loss
        if "relu53" in layers:
            loss += tf.reduce_mean(tf.squared_difference(c53, c53_gt))

        # Fifth MaxPool
        with K.name_scope("pool5"):
            pool5 = MaxPool2D(pool_size=2, strides=2, padding='same')

            # Output
            p5 = pool5(c53)

            # Ground Truth
            p5_gt = pool5(c53_gt)

        # Reshape
        shape = int(np.prod(p5.get_shape()[1:]))
        p5_flat = tf.reshape(p5, [-1, shape])
        p5_gt_flat = tf.reshape(p5_gt, [-1, shape])

        # First Dense
        with K.name_scope("fc1"):
            fc1 = Dense(units=4096)

            # Output
            f1 = fc1(p5_flat)
            f1 = relu(f1)

            # Ground Truth
            f1_gt = fc1(p5_gt_flat)
            f1_gt = relu(f1_gt)

            # Add Parameters
            parameters += [fc1.kernel, fc1.bias]

        # Loss
        if "fc1" in layers:
            loss += tf.reduce_mean(tf.squared_difference(f1, f1_gt))

        # Second Dense
        with K.name_scope("fc2"):
            fc2 = Dense(units=4096)

            # Output
            f2 = fc2(f1)
            f2 = relu(f2)

            # Ground Truth
            f2_gt = fc2(f1_gt)
            f2_gt = relu(f2_gt)

            # Add Parameters
            parameters += [fc2.kernel, fc2.bias]

        # Loss
        if "fc2" in layers:
            loss += tf.reduce_mean(tf.squared_difference(f2, f2_gt))

        # Third Dense
        with K.name_scope("fc3"):
            fc3 = Dense(units=1000)

            # Output
            f3 = fc3(f2)

            # Ground Truth
            f3_gt = fc3(f2_gt)

            # Add Parameters
            parameters += [fc3.kernel, fc3.bias]

        if "fc3" in layers:
            loss += tf.reduce_mean(tf.squared_difference(f3, f3_gt))

    # Load Weights
    load_weights(parameters)
    return loss

