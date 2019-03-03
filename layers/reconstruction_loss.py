# reconstruction_loss.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.activations import relu
from keras.layers import Conv2D, Dense, MaxPool2D, Reshape


def gram_matrix(x, nhwc=True):
    assert x.shape.ndims == 4

    shape = tf.shape(x)
    if nhwc:
        flat = tf.reshape(x, shape=(-1, shape[1] * shape[2], shape[3]))
    else:
        flat = tf.reshape(x, shape=(-1, shape[2] * shape[3], shape[1]))

    return tf.matmul(flat, flat, transpose_a=True) / tf.cast(shape[1] * shape[2] * shape[3], dtype=tf.float32)


def load_weights(config, parameters):
    with tf.Session() as session:
        weights = np.load(config.weights_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            session.run(parameters[i].assign(weights[k]))


def style_loss(config, fake, real, layers):
    loss = 0.0
    parameters = []

    # Preprocess
    mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32,
                       shape=[1, 1, 1, 3], name="img_mean")
    fake = tf.image.resize_images(fake, size=[224, 224]) * 255.0 - mean
    real = tf.image.resize_images(real, size=[224, 224]) * 255.0 - mean

    # First Convolution
    w_conv11 = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32, stddev=1e-1),
                           trainable=False, name="weights")
    b_conv11 = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                           trainable=False, name="biases")
    parameters += [w_conv11, b_conv11]

    # Output
    conv11 = tf.nn.conv2d(fake, w_conv11,
                          strides=[1, 1, 1, 1], padding='SAME') + b_conv11
    conv11 = tf.nn.relu(conv11)

    # Ground-Truth
    conv11_gt = tf.nn.conv2d(real, w_conv11,
                             strides=[1, 1, 1, 1], padding='SAME') + b_conv11
    conv11_gt = tf.nn.relu(conv11_gt)

    # Loss
    if "relu11" in layers:
        loss += tf.reduce_sum(tf.square(gram_matrix(conv11 / 255.0) - gram_matrix(conv11_gt / 255.0)))

    # Second Convolution
    w_conv12 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1),
                           trainable=False, name="weights")
    b_conv12 = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                           trainable=False, name="biases")
    parameters += [w_conv12, b_conv12]

    # Output
    conv12 = tf.nn.conv2d(conv11, w_conv12,
                          strides=[1, 1, 1, 1], padding='SAME') + b_conv12
    conv12 = tf.nn.relu(conv12)

    # Ground-Truth
    conv12_gt = tf.nn.conv2d(conv11_gt, w_conv12,
                             strides=[1, 1, 1, 1], padding='SAME') + b_conv12
    conv12_gt = tf.nn.relu(conv12_gt)

    # Loss
    if "relu12" in layers:
        loss += tf.reduce_sum(tf.square(gram_matrix(conv12 / 255.0) - gram_matrix(conv12_gt / 255.0)))

    # First Maxpool
    pool1 = tf.nn.max_pool(conv12, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name="pool1")
    pool1_gt = tf.nn.max_pool(conv12_gt, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name="pool1_gt")

    # Third Convolution
    w_conv21 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=1e-1),
                           trainable=False, name="weights")
    b_conv21 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                           trainable=False, name="biases")
    parameters += [w_conv21, b_conv21]

    # Output
    conv21 = tf.nn.conv2d(pool1, w_conv21,
                          strides=[1, 1, 1, 1], padding='SAME') + b_conv21
    conv21 = tf.nn.relu(conv21)

    # Ground-Truth
    conv21_gt = tf.nn.conv2d(pool1_gt, w_conv21,
                             strides=[1, 1, 1, 1], padding='SAME') + b_conv21
    conv21_gt = tf.nn.relu(conv21_gt)

    # Loss
    if "relu21" in layers:
        loss += tf.reduce_sum(tf.square(gram_matrix(conv21 / 255.0) - gram_matrix(conv21_gt / 255.0)))

    # Fourth Convolution
    w_conv22 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32, stddev=1e-1),
                           trainable=False, name="weights")
    b_conv22 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                           trainable=False, name="biases")
    parameters += [w_conv22, b_conv22]

    # Output
    conv22 = tf.nn.conv2d(conv21, w_conv22,
                          strides=[1, 1, 1, 1], padding='SAME') + b_conv22
    conv22 = tf.nn.relu(conv22)

    # Ground-Truth
    conv22_gt = tf.nn.conv2d(conv21_gt, w_conv22,
                             strides=[1, 1, 1, 1], padding='SAME') + b_conv22
    conv22_gt = tf.nn.relu(conv22_gt)

    # Loss
    if "relu22" in layers:
        loss += tf.reduce_sum(tf.square(gram_matrix(conv22 / 255.0) - gram_matrix(conv22_gt / 255.0)))

    # Second Maxpool
    pool2 = tf.nn.max_pool(conv22, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name="pool2")
    pool2_gt = tf.nn.max_pool(conv22_gt, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name="pool2_gt")

    # Fifth Convolution
    w_conv31 = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32, stddev=1e-1),
                           trainable=False, name="weights")
    b_conv31 = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                           trainable=False, name="biases")
    parameters += [w_conv31, b_conv31]

    # Output
    conv31 = tf.nn.conv2d(pool2, w_conv31,
                          strides=[1, 1, 1, 1], padding='SAME') + b_conv31
    conv31 = tf.nn.relu(conv31)

    # Ground-Truth
    conv31_gt = tf.nn.conv2d(pool2_gt, w_conv31,
                             strides=[1, 1, 1, 1], padding='SAME') + b_conv31
    conv31_gt = tf.nn.relu(conv31_gt)

    # Loss
    if "relu31" in layers:
        loss += tf.reduce_sum(tf.square(gram_matrix(conv31 / 255.0) - gram_matrix(conv31_gt / 255.0)))

    # Sixth Convolution
    w_conv32 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1),
                           trainable=False, name="weights")
    b_conv32 = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                           trainable=False, name="biases")
    parameters += [w_conv32, b_conv32]

    # Output
    conv32 = tf.nn.conv2d(conv31, w_conv32,
                          strides=[1, 1, 1, 1], padding='SAME') + b_conv32
    conv32 = tf.nn.relu(conv32)

    # Ground-Truth
    conv32_gt = tf.nn.conv2d(conv31_gt, w_conv32,
                             strides=[1, 1, 1, 1], padding='SAME') + b_conv32
    conv32_gt = tf.nn.relu(conv32_gt)

    # Loss
    if "relu32" in layers:
        loss += tf.reduce_sum(tf.square(gram_matrix(conv32 / 255.0) - gram_matrix(conv32_gt / 255.0)))

    # Seventh Convolution
    w_conv33 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1),
                           trainable=False, name="weights")
    b_conv33 = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                           trainable=False, name="biases")
    parameters += [w_conv33, b_conv33]

    # Output
    conv33 = tf.nn.conv2d(conv32, w_conv33,
                          strides=[1, 1, 1, 1], padding='SAME') + b_conv33
    conv33 = tf.nn.relu(conv33)

    # Ground-Truth
    conv33_gt = tf.nn.conv2d(conv32_gt, w_conv33,
                             strides=[1, 1, 1, 1], padding='SAME') + b_conv33
    conv33_gt = tf.nn.relu(conv33_gt)

    # Loss
    if "relu33" in layers:
        loss += tf.reduce_sum(tf.square(gram_matrix(conv33 / 255.0) - gram_matrix(conv33_gt / 255.0)))

    # Third Maxpool
    pool3 = tf.nn.max_pool(conv33, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name="pool3")
    pool3_gt = tf.nn.max_pool(conv33_gt, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name="pool3_gt")

    # Eighth Convolution
    w_conv41 = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32, stddev=1e-1),
                           trainable=False, name="weights")
    b_conv41 = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                           trainable=False, name="biases")
    parameters += [w_conv41, b_conv41]

    # Output
    conv41 = tf.nn.conv2d(pool3, w_conv41,
                          strides=[1, 1, 1, 1], padding='SAME') + b_conv41
    conv41 = tf.nn.relu(conv41)

    # Ground-Truth
    conv41_gt = tf.nn.conv2d(pool3_gt, w_conv41,
                             strides=[1, 1, 1, 1], padding='SAME') + b_conv41
    conv41_gt = tf.nn.relu(conv41_gt)

    # Loss
    if "relu41" in layers:
        loss += tf.reduce_sum(tf.square(gram_matrix(conv41 / 255.0) - gram_matrix(conv41_gt / 255.0)))

    # Nineth Convolution
    w_conv42 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),
                           trainable=False, name="weights")
    b_conv42 = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                           trainable=False, name="biases")
    parameters += [w_conv42, b_conv42]

    # Output
    conv42 = tf.nn.conv2d(conv41, w_conv42,
                          strides=[1, 1, 1, 1], padding='SAME') + b_conv42
    conv42 = tf.nn.relu(conv42)

    # Ground-Truth
    conv42_gt = tf.nn.conv2d(conv41_gt, w_conv42,
                             strides=[1, 1, 1, 1], padding='SAME') + b_conv42
    conv42_gt = tf.nn.relu(conv42_gt)

    # Loss
    if "relu42" in layers:
        loss += tf.reduce_sum(tf.square(gram_matrix(conv42 / 255.0) - gram_matrix(conv42_gt / 255.0)))

    # Tenth Convolution
    w_conv43 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),
                           trainable=False, name="weights")
    b_conv43 = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                           trainable=False, name="biases")
    parameters += [w_conv43, b_conv43]

    # Output
    conv43 = tf.nn.conv2d(conv42, w_conv43,
                          strides=[1, 1, 1, 1], padding='SAME') + b_conv43
    conv43 = tf.nn.relu(conv43)

    # Ground-Truth
    conv43_gt = tf.nn.conv2d(conv42_gt, w_conv43,
                             strides=[1, 1, 1, 1], padding='SAME') + b_conv43
    conv43_gt = tf.nn.relu(conv43_gt)

    # Loss
    if "relu43" in layers:
        loss += tf.reduce_sum(tf.square(gram_matrix(conv43 / 255.0) - gram_matrix(conv43_gt / 255.0)))

    # Fourth Maxpool
    pool4 = tf.nn.max_pool(conv43, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name="pool4")
    pool4_gt = tf.nn.max_pool(conv43_gt, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name="pool4_gt")

    # Eleventh Convolution
    w_conv51 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),
                           trainable=False, name="weights")
    b_conv51 = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                           trainable=False, name="biases")
    parameters += [w_conv51, b_conv51]

    # Output
    conv51 = tf.nn.conv2d(pool4, w_conv51,
                          strides=[1, 1, 1, 1], padding='SAME') + b_conv51
    conv51 = tf.nn.relu(conv51)

    # Ground-Truth
    conv51_gt = tf.nn.conv2d(pool4_gt, w_conv51,
                             strides=[1, 1, 1, 1], padding='SAME') + b_conv51
    conv51_gt = tf.nn.relu(conv51_gt)

    # Loss
    if "relu51" in layers:
        loss += tf.reduce_sum(tf.square(gram_matrix(conv51 / 255.0) - gram_matrix(conv51_gt / 255.0)))

    # Twelfth Convolution
    w_conv52 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),
                           trainable=False, name="weights")
    b_conv52 = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                           trainable=False, name="biases")
    parameters += [w_conv52, b_conv52]

    # Output
    conv52 = tf.nn.conv2d(conv51, w_conv52,
                          strides=[1, 1, 1, 1], padding='SAME') + b_conv52
    conv52 = tf.nn.relu(conv52)

    # Ground-Truth
    conv52_gt = tf.nn.conv2d(conv51_gt, w_conv52,
                             strides=[1, 1, 1, 1], padding='SAME') + b_conv52
    conv52_gt = tf.nn.relu(conv52_gt)

    # Loss
    if "relu52" in layers:
        loss += tf.reduce_sum(tf.square(gram_matrix(conv52 / 255.0) - gram_matrix(conv52_gt / 255.0)))

    # Thirteenth Convolution
    w_conv53 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),
                           trainable=False, name="weights")
    b_conv53 = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                           trainable=False, name="biases")
    parameters += [w_conv53, b_conv53]

    # Output
    conv53 = tf.nn.conv2d(conv52, w_conv53,
                          strides=[1, 1, 1, 1], padding='SAME') + b_conv53
    conv53 = tf.nn.relu(conv53)

    # Ground-Truth
    conv53_gt = tf.nn.conv2d(conv52_gt, w_conv53,
                             strides=[1, 1, 1, 1], padding='SAME') + b_conv53
    conv53_gt = tf.nn.relu(conv53_gt)

    # Loss
    if "relu53" in layers:
        loss += tf.reduce_sum(tf.square(gram_matrix(conv53 / 255.0) - gram_matrix(conv53_gt / 255.0)))

    # Fifth Maxpool
    pool5 = tf.nn.max_pool(conv53, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name="pool5")
    pool5_gt = tf.nn.max_pool(conv53_gt, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name="pool5_gt")

    # FC Parameters
    shape = int(np.prod(pool5.get_shape()[1:]))
    pool5_flat = tf.reshape(pool5, [-1, shape])
    pool5_gt_flat = tf.reshape(pool5_gt, [-1, shape])

    # First FC
    w_fc1 = tf.Variable(tf.truncated_normal([shape, 4096], dtype=tf.float32, stddev=1e-1),
                        trainable=False, name="weights")
    b_fc1 = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                        trainable=False, name="biases")
    parameters += [w_fc1, b_fc1]

    # Output
    fc1 = tf.matmul(pool5_flat, w_fc1) + b_fc1
    fc1 = tf.nn.relu(fc1)

    # Ground-Truth
    fc1_gt = tf.matmul(pool5_gt_flat, w_fc1) + b_fc1
    fc1_gt = tf.nn.relu(fc1_gt)

    # Loss
    if "fc1" in layers:
        loss += tf.reduce_sum(tf.square(gram_matrix(fc1 / 255.0) - gram_matrix(fc1_gt / 255.0)))

    # Second FC
    w_fc2 = tf.Variable(tf.truncated_normal([4096, 4096], dtype=tf.float32, stddev=1e-1),
                        trainable=False, name="weights")
    b_fc2 = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                        trainable=False, name="biases")
    parameters += [w_fc2, b_fc2]

    # Output
    fc2 = tf.matmul(fc1, w_fc2) + b_fc2
    fc2 = tf.nn.relu(fc2)

    # Ground-Truth
    fc2_gt = tf.matmul(fc1_gt, w_fc2) + b_fc2
    fc2_gt = tf.nn.relu(fc2_gt)

    # Loss
    if "fc2" in layers:
        loss += tf.reduce_sum(tf.square(gram_matrix(fc2 / 255.0) - gram_matrix(fc2_gt / 255.0)))

    # Third FC
    w_fc3 = tf.Variable(tf.truncated_normal([4096, 1000], dtype=tf.float32, stddev=1e-1),
                        trainable=False, name="weights")
    b_fc3 = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32),
                        trainable=False, name="biases")
    parameters += [w_fc3, b_fc3]

    # Output
    fc3 = tf.matmul(fc2, w_fc3) + b_fc3

    # Ground-Truth
    fc3_gt = tf.matmul(fc2_gt, w_fc3) + b_fc3

    # Loss
    if "fc3" in layers:
        loss += tf.reduce_sum(tf.square(gram_matrix(fc3 / 255.0) - gram_matrix(fc3_gt / 255.0)))

    # Load Weights
    load_weights(config, parameters)
    return loss


def feature_loss(config, fake, real, layers):
    loss = 0.0
    parameters = []

    # Preprocess
    mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32,
                       shape=[1, 1, 1, 3], name="img_mean")
    fake = tf.image.resize_images(fake, size=[224, 224]) * 255.0 - mean
    real = tf.image.resize_images(real, size=[224, 224]) * 255.0 - mean

    # First Convolution
    w_conv11 = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32, stddev=1e-1),
                           trainable=False, name="weights")
    b_conv11 = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                           trainable=False, name="biases")
    parameters += [w_conv11, b_conv11]

    # Output
    conv11 = tf.nn.conv2d(fake, w_conv11,
                          strides=[1, 1, 1, 1], padding='SAME') + b_conv11
    conv11 = tf.nn.relu(conv11)

    # Ground-Truth
    conv11_gt = tf.nn.conv2d(real, w_conv11,
                             strides=[1, 1, 1, 1], padding='SAME') + b_conv11
    conv11_gt = tf.nn.relu(conv11_gt)

    # Loss
    if "relu11" in layers:
        loss += tf.reduce_mean(tf.squared_difference(conv11 / 255.0, conv11_gt / 255.0, name=None))

    # Second Convolution
    w_conv12 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1),
                           trainable=False, name="weights")
    b_conv12 = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                           trainable=False, name="biases")
    parameters += [w_conv12, b_conv12]

    # Output
    conv12 = tf.nn.conv2d(conv11, w_conv12,
                          strides=[1, 1, 1, 1], padding='SAME') + b_conv12
    conv12 = tf.nn.relu(conv12)

    # Ground-Truth
    conv12_gt = tf.nn.conv2d(conv11_gt, w_conv12,
                             strides=[1, 1, 1, 1], padding='SAME') + b_conv12
    conv12_gt = tf.nn.relu(conv12_gt)

    # Loss
    if "relu12" in layers:
        loss += tf.reduce_mean(tf.squared_difference(conv12 / 255.0, conv12_gt / 255.0, name=None))

    # First Maxpool
    pool1 = tf.nn.max_pool(conv12, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name="pool1")
    pool1_gt = tf.nn.max_pool(conv12_gt, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name="pool1_gt")

    # Third Convolution
    w_conv21 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=1e-1),
                           trainable=False, name="weights")
    b_conv21 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                           trainable=False, name="biases")
    parameters += [w_conv21, b_conv21]

    # Output
    conv21 = tf.nn.conv2d(pool1, w_conv21,
                          strides=[1, 1, 1, 1], padding='SAME') + b_conv21
    conv21 = tf.nn.relu(conv21)

    # Ground-Truth
    conv21_gt = tf.nn.conv2d(pool1_gt, w_conv21,
                             strides=[1, 1, 1, 1], padding='SAME') + b_conv21
    conv21_gt = tf.nn.relu(conv21_gt)

    # Loss
    if "relu21" in layers:
        loss += tf.reduce_mean(tf.squared_difference(conv21 / 255.0, conv21_gt / 255.0, name=None))

    # Fourth Convolution
    w_conv22 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32, stddev=1e-1),
                           trainable=False, name="weights")
    b_conv22 = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                           trainable=False, name="biases")
    parameters += [w_conv22, b_conv22]

    # Output
    conv22 = tf.nn.conv2d(conv21, w_conv22,
                          strides=[1, 1, 1, 1], padding='SAME') + b_conv22
    conv22 = tf.nn.relu(conv22)

    # Ground-Truth
    conv22_gt = tf.nn.conv2d(conv21_gt, w_conv22,
                             strides=[1, 1, 1, 1], padding='SAME') + b_conv22
    conv22_gt = tf.nn.relu(conv22_gt)

    # Loss
    if "relu22" in layers:
        loss += tf.reduce_mean(tf.squared_difference(conv22 / 255.0, conv22_gt / 255.0, name=None))

    # Second Maxpool
    pool2 = tf.nn.max_pool(conv22, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name="pool2")
    pool2_gt = tf.nn.max_pool(conv22_gt, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name="pool2_gt")

    # Fifth Convolution
    w_conv31 = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32, stddev=1e-1),
                           trainable=False, name="weights")
    b_conv31 = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                           trainable=False, name="biases")
    parameters += [w_conv31, b_conv31]

    # Output
    conv31 = tf.nn.conv2d(pool2, w_conv31,
                          strides=[1, 1, 1, 1], padding='SAME') + b_conv31
    conv31 = tf.nn.relu(conv31)

    # Ground-Truth
    conv31_gt = tf.nn.conv2d(pool2_gt, w_conv31,
                             strides=[1, 1, 1, 1], padding='SAME') + b_conv31
    conv31_gt = tf.nn.relu(conv31_gt)

    # Loss
    if "relu31" in layers:
        loss += tf.reduce_mean(tf.squared_difference(conv31 / 255.0, conv31_gt / 255.0, name=None))

    # Sixth Convolution
    w_conv32 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1),
                           trainable=False, name="weights")
    b_conv32 = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                           trainable=False, name="biases")
    parameters += [w_conv32, b_conv32]

    # Output
    conv32 = tf.nn.conv2d(conv31, w_conv32,
                          strides=[1, 1, 1, 1], padding='SAME') + b_conv32
    conv32 = tf.nn.relu(conv32)

    # Ground-Truth
    conv32_gt = tf.nn.conv2d(conv31_gt, w_conv32,
                             strides=[1, 1, 1, 1], padding='SAME') + b_conv32
    conv32_gt = tf.nn.relu(conv32_gt)

    # Loss
    if "relu32" in layers:
        loss += tf.reduce_mean(tf.squared_difference(conv32 / 255.0, conv32_gt / 255.0, name=None))

    # Seventh Convolution
    w_conv33 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1),
                           trainable=False, name="weights")
    b_conv33 = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                           trainable=False, name="biases")
    parameters += [w_conv33, b_conv33]

    # Output
    conv33 = tf.nn.conv2d(conv32, w_conv33,
                          strides=[1, 1, 1, 1], padding='SAME') + b_conv33
    conv33 = tf.nn.relu(conv33)

    # Ground-Truth
    conv33_gt = tf.nn.conv2d(conv32_gt, w_conv33,
                             strides=[1, 1, 1, 1], padding='SAME') + b_conv33
    conv33_gt = tf.nn.relu(conv33_gt)

    # Loss
    if "relu33" in layers:
        loss += tf.reduce_mean(tf.squared_difference(conv33 / 255.0, conv33_gt / 255.0, name=None))

    # Third Maxpool
    pool3 = tf.nn.max_pool(conv33, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name="pool3")
    pool3_gt = tf.nn.max_pool(conv33_gt, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name="pool3_gt")

    # Eighth Convolution
    w_conv41 = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32, stddev=1e-1),
                           trainable=False, name="weights")
    b_conv41 = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                           trainable=False, name="biases")
    parameters += [w_conv41, b_conv41]

    # Output
    conv41 = tf.nn.conv2d(pool3, w_conv41,
                          strides=[1, 1, 1, 1], padding='SAME') + b_conv41
    conv41 = tf.nn.relu(conv41)

    # Ground-Truth
    conv41_gt = tf.nn.conv2d(pool3_gt, w_conv41,
                             strides=[1, 1, 1, 1], padding='SAME') + b_conv41
    conv41_gt = tf.nn.relu(conv41_gt)

    # Loss
    if "relu41" in layers:
        loss += tf.reduce_mean(tf.squared_difference(conv41 / 255.0, conv41_gt / 255.0, name=None))

    # Nineth Convolution
    w_conv42 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),
                           trainable=False, name="weights")
    b_conv42 = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                           trainable=False, name="biases")
    parameters += [w_conv42, b_conv42]

    # Output
    conv42 = tf.nn.conv2d(conv41, w_conv42,
                          strides=[1, 1, 1, 1], padding='SAME') + b_conv42
    conv42 = tf.nn.relu(conv42)

    # Ground-Truth
    conv42_gt = tf.nn.conv2d(conv41_gt, w_conv42,
                             strides=[1, 1, 1, 1], padding='SAME') + b_conv42
    conv42_gt = tf.nn.relu(conv42_gt)

    # Loss
    if "relu42" in layers:
        loss += tf.reduce_mean(tf.squared_difference(conv42 / 255.0, conv42_gt / 255.0, name=None))

    # Tenth Convolution
    w_conv43 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),
                           trainable=False, name="weights")
    b_conv43 = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                           trainable=False, name="biases")
    parameters += [w_conv43, b_conv43]

    # Output
    conv43 = tf.nn.conv2d(conv42, w_conv43,
                          strides=[1, 1, 1, 1], padding='SAME') + b_conv43
    conv43 = tf.nn.relu(conv43)

    # Ground-Truth
    conv43_gt = tf.nn.conv2d(conv42_gt, w_conv43,
                             strides=[1, 1, 1, 1], padding='SAME') + b_conv43
    conv43_gt = tf.nn.relu(conv43_gt)

    # Loss
    if "relu43" in layers:
        loss += tf.reduce_mean(tf.squared_difference(conv43 / 255.0, conv43_gt / 255.0, name=None))

    # Fourth Maxpool
    pool4 = tf.nn.max_pool(conv43, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name="pool4")
    pool4_gt = tf.nn.max_pool(conv43_gt, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name="pool4_gt")

    # Eleventh Convolution
    w_conv51 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),
                           trainable=False, name="weights")
    b_conv51 = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                           trainable=False, name="biases")
    parameters += [w_conv51, b_conv51]

    # Output
    conv51 = tf.nn.conv2d(pool4, w_conv51,
                          strides=[1, 1, 1, 1], padding='SAME') + b_conv51
    conv51 = tf.nn.relu(conv51)

    # Ground-Truth
    conv51_gt = tf.nn.conv2d(pool4_gt, w_conv51,
                             strides=[1, 1, 1, 1], padding='SAME') + b_conv51
    conv51_gt = tf.nn.relu(conv51_gt)

    # Loss
    if "relu51" in layers:
        loss += tf.reduce_mean(tf.squared_difference(conv51 / 255.0, conv51_gt / 255.0, name=None))

    # Twelfth Convolution
    w_conv52 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),
                           trainable=False, name="weights")
    b_conv52 = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                           trainable=False, name="biases")
    parameters += [w_conv52, b_conv52]

    # Output
    conv52 = tf.nn.conv2d(conv51, w_conv52,
                          strides=[1, 1, 1, 1], padding='SAME') + b_conv52
    conv52 = tf.nn.relu(conv52)

    # Ground-Truth
    conv52_gt = tf.nn.conv2d(conv51_gt, w_conv52,
                             strides=[1, 1, 1, 1], padding='SAME') + b_conv52
    conv52_gt = tf.nn.relu(conv52_gt)

    # Loss
    if "relu52" in layers:
        loss += tf.reduce_mean(tf.squared_difference(conv52 / 255.0, conv52_gt / 255.0, name=None))

    # Thirteenth Convolution
    w_conv53 = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),
                           trainable=False, name="weights")
    b_conv53 = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                           trainable=False, name="biases")
    parameters += [w_conv53, b_conv53]

    # Output
    conv53 = tf.nn.conv2d(conv52, w_conv53,
                          strides=[1, 1, 1, 1], padding='SAME') + b_conv53
    conv53 = tf.nn.relu(conv53)

    # Ground-Truth
    conv53_gt = tf.nn.conv2d(conv52_gt, w_conv53,
                             strides=[1, 1, 1, 1], padding='SAME') + b_conv53
    conv53_gt = tf.nn.relu(conv53_gt)

    # Loss
    if "relu53" in layers:
        loss += tf.reduce_mean(tf.squared_difference(conv53 / 255.0, conv53_gt / 255.0, name=None))

    # Fifth Maxpool
    pool5 = tf.nn.max_pool(conv53, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name="pool5")
    pool5_gt = tf.nn.max_pool(conv53_gt, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name="pool5_gt")

    # FC Parameters
    shape = int(np.prod(pool5.get_shape()[1:]))
    pool5_flat = tf.reshape(pool5, [-1, shape])
    pool5_gt_flat = tf.reshape(pool5_gt, [-1, shape])

    # First FC
    w_fc1 = tf.Variable(tf.truncated_normal([shape, 4096], dtype=tf.float32, stddev=1e-1),
                        trainable=False, name="weights")
    b_fc1 = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                        trainable=False, name="biases")
    parameters += [w_fc1, b_fc1]

    # Output
    fc1 = tf.matmul(pool5_flat, w_fc1) + b_fc1
    fc1 = tf.nn.relu(fc1)

    # Ground-Truth
    fc1_gt = tf.matmul(pool5_gt_flat, w_fc1) + b_fc1
    fc1_gt = tf.nn.relu(fc1_gt)

    # Loss
    if "fc1" in layers:
        loss += tf.reduce_mean(tf.squared_difference(fc1 / 255.0, fc1_gt / 255.0, name=None))

    # Second FC
    w_fc2 = tf.Variable(tf.truncated_normal([4096, 4096], dtype=tf.float32, stddev=1e-1),
                        trainable=False, name="weights")
    b_fc2 = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                        trainable=False, name="biases")
    parameters += [w_fc2, b_fc2]

    # Output
    fc2 = tf.matmul(fc1, w_fc2) + b_fc2
    fc2 = tf.nn.relu(fc2)

    # Ground-Truth
    fc2_gt = tf.matmul(fc1_gt, w_fc2) + b_fc2
    fc2_gt = tf.nn.relu(fc2_gt)

    # Loss
    if "fc2" in layers:
        loss += tf.reduce_mean(tf.squared_difference(fc2 / 255.0, fc2_gt / 255.0, name=None))

    # Third FC
    w_fc3 = tf.Variable(tf.truncated_normal([4096, 1000], dtype=tf.float32, stddev=1e-1),
                        trainable=False, name="weights")
    b_fc3 = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32),
                        trainable=False, name="biases")
    parameters += [w_fc3, b_fc3]

    # Output
    fc3 = tf.matmul(fc2, w_fc3) + b_fc3

    # Ground-Truth
    fc3_gt = tf.matmul(fc2_gt, w_fc3) + b_fc3

    # Loss
    if "fc3" in layers:
        loss += tf.reduce_mean(tf.squared_difference(fc3 / 255.0, fc3_gt / 255.0, name=None))

    # Load Weights
    load_weights(config, parameters)
    return loss
