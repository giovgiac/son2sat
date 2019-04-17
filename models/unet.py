# unet.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from base.base_model import BaseModel
from keras import backend as K
from keras.activations import sigmoid, relu
from keras.layers import Concatenate, Conv2D, Dense, Dropout, MaxPool2D, ReLU, Reshape, UpSampling2D

from layers.dilated_convolution import dilated_conv2d
from layers.guided_filter import guided_filter
from layers.reconstruction_loss import vgg16, style_loss


class Unet(BaseModel):
    def __init__(self, config, is_evaluating=False):
        super(Unet, self).__init__(config, is_evaluating)

        # Inputs
        self.x = None
        self.y = None

        # Losses
        self.discriminator_loss = None
        self.pixel_loss = None
        self.reconstruction_loss = None

        # Entropies
        self.real_entropy = None
        self.fake_entropy = None
        self.cross_entropy = None
        self.disc_entropy = None

        # Shared Layers
        self.disc_layers = {}

        # Outputs
        self.fn = None
        self.train_step = None

        self.dis_filters = self.config.gen_filters
        self.gen_filters = self.config.gen_filters
        self.input_shape = tf.TensorShape([self.config.image_height,
                                           self.config.image_width,
                                           self.config.input_channels])

        self.output_shape = tf.TensorShape([self.config.image_height,
                                            self.config.image_width,
                                            self.config.output_channels])

        self.build_model(1 if self.is_evaluating else self.config.batch_size)
        self.init_saver()

    def build_discriminator(self, x, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            if not self.disc_layers:
                # First Layer
                self.disc_layers['h0_conv'] = Conv2D(filters=self.dis_filters, kernel_size=(5, 5), strides=(2, 2),
                                                     padding='same')
                self.disc_layers['h0_actv'] = ReLU()

                # Second Layer
                self.disc_layers['h1_conv'] = Conv2D(filters=self.dis_filters * 2, kernel_size=(5, 5), strides=(2, 2),
                                                     padding='same')
                self.disc_layers['h1_actv'] = ReLU()

                # Third Layer
                self.disc_layers['h2_conv'] = Conv2D(filters=self.dis_filters * 4, kernel_size=(5, 5), strides=(2, 2),
                                                     padding='same')
                self.disc_layers['h2_actv'] = ReLU()

                # Fourth Layer
                self.disc_layers['h3_conv'] = Conv2D(filters=self.dis_filters * 8, kernel_size=(5, 5), strides=(1, 1),
                                                     padding='same')
                self.disc_layers['h3_actv'] = ReLU()

                # Fifth Layer
                self.disc_layers['h4_resh'] = Reshape(target_shape=(-1,))
                self.disc_layers['h4_line'] = Dense(units=1)

            # First Convolution
            h0 = self.disc_layers['h0_conv'](x)
            h0 = self.disc_layers['h0_actv'](h0)

            # Second Convolution
            h1 = self.disc_layers['h1_conv'](h0)
            h1 = tf.contrib.layers.batch_norm(h1, decay=0.9, epsilon=1e-5, updates_collections=None, scale=True)
            h1 = self.disc_layers['h1_actv'](h1)

            # Third Convolution
            h2 = self.disc_layers['h2_conv'](h1)
            h2 = tf.contrib.layers.batch_norm(h2, decay=0.9, epsilon=1e-5, updates_collections=None, scale=True)
            h2 = self.disc_layers['h2_actv'](h2)

            # Fourth Convolution
            h3 = self.disc_layers['h3_conv'](h2)
            h3 = tf.contrib.layers.batch_norm(h3, decay=0.9, epsilon=1e-5, updates_collections=None, scale=True)
            h3 = self.disc_layers['h3_actv'](h3)

            # Dense Layer
            h4 = self.disc_layers['h4_resh'](h3)
            h4 = self.disc_layers['h4_line'](h4)

            return h4, sigmoid(h4)

    def build_generator(self, x):
        son = tf.image.grayscale_to_rgb(x)
        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
            with K.name_scope("encode1"):
                # Dilated Convolution
                e1 = dilated_conv2d(features=x, filters=self.gen_filters * 2, kernel_size=3, padding='same')

                # Max Pool
                e1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(e1)
                e1 = ReLU()(e1)

            with K.name_scope("encode2"):
                # Dilated Convolution
                e2 = dilated_conv2d(features=e1, filters=self.gen_filters * 4, kernel_size=3, padding='same')

                # Max Pool
                e2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(e2)
                e2 = ReLU()(e2)
                e2 = tf.contrib.layers.batch_norm(e2, decay=0.9, epsilon=1e-5, updates_collections=None, scale=True)

            with K.name_scope("encode3"):
                # Dilated Convolution
                e3 = dilated_conv2d(features=e2, filters=self.gen_filters * 8, kernel_size=3, padding='same')

                # Max Pool
                e3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(e3)
                e3 = ReLU()(e3)
                e3 = tf.contrib.layers.batch_norm(e3, decay=0.9, epsilon=1e-5, updates_collections=None, scale=True)

            with K.name_scope("encode4"):
                # Dilated Convolution
                e4 = dilated_conv2d(features=e3, filters=self.gen_filters * 8, kernel_size=3, padding='same')

                # Max Pool
                e4 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(e4)
                e4 = ReLU()(e4)
                e4 = tf.contrib.layers.batch_norm(e4, decay=0.9, epsilon=1e-5, updates_collections=None, scale=True)

            with K.name_scope("decode1"):
                # Up Convolution
                d1 = UpSampling2D(size=(2, 2), interpolation='nearest')(e4)
                d1 = Conv2D(filters=self.gen_filters * 8, kernel_size=4, padding='same')(d1)

                d1 = Dropout(rate=0.2)(d1)
                d1 = ReLU()(d1)
                d1 = tf.contrib.layers.batch_norm(d1, decay=0.9, epsilon=1e-5, updates_collections=None, scale=True)

                # Skip Connection
                d1 = Concatenate()([d1, e3])

            with K.name_scope("decode2"):
                # Up Convolution
                d2 = UpSampling2D(size=(2, 2), interpolation='nearest')(d1)
                d2 = Conv2D(filters=self.gen_filters * 4, kernel_size=4, padding='same')(d2)

                d2 = Dropout(rate=0.2)(d2)
                d2 = ReLU()(d2)
                d2 = tf.contrib.layers.batch_norm(d2, decay=0.9, epsilon=1e-5, updates_collections=None, scale=True)

                # Skip Connection
                d2 = Concatenate()([d2, e2])

            with K.name_scope("decode3"):
                # Up Convolution
                d3 = UpSampling2D(size=(2, 2), interpolation='nearest')(d2)
                d3 = Conv2D(filters=self.gen_filters * 2, kernel_size=4, padding='same')(d3)

                d3 = Dropout(rate=0.2)(d3)
                d3 = ReLU()(d3)
                d3 = tf.contrib.layers.batch_norm(d3, decay=0.9, epsilon=1e-5, updates_collections=None, scale=True)

                # Skip Connection
                d3 = Concatenate()([d3, e1])

            with K.name_scope("decode4"):
                # Up Convolution
                d4 = UpSampling2D(size=(2, 2), interpolation='nearest')(d3)
                d4 = Conv2D(filters=self.output_shape.as_list()[-1], kernel_size=4, padding='same')(d4)

                d4 = ReLU()(d4)
                d4 = tf.contrib.layers.batch_norm(d4, decay=0.9, epsilon=1e-5, updates_collections=None, scale=True)

            with K.name_scope("guided1"):
                # First Guided Convolution
                c1 = Conv2D(filters=self.output_shape.as_list()[-1], kernel_size=3, strides=1, padding='same')(d4)
                c1 = ReLU()(c1)
                c1 = tf.contrib.layers.batch_norm(c1, decay=0.9, epsilon=1e-5, updates_collections=None, scale=True)

                # First Guided Filter
                g1 = guided_filter(x=son, y=c1, r=60, eps=1e-4, nhwc=True)

            with K.name_scope("guided2"):
                # Second Guided Convolution
                c2 = Conv2D(filters=self.output_shape.as_list()[-1], kernel_size=3, strides=1, padding='same')(d4)
                c2 = ReLU()(c2)
                c2 = tf.contrib.layers.batch_norm(c2, decay=0.9, epsilon=1e-5, updates_collections=None, scale=True)

                # Second Guided Filter
                g2 = guided_filter(x=son, y=c2, r=60, eps=1e-4, nhwc=True)

            # Guided Concatenation
            gf = Concatenate()([g2 * d4, g1 + d4, d4])

            # Final Convolution
            fn = Conv2D(filters=self.output_shape.as_list()[-1], kernel_size=1, strides=1, padding='same')(gf)
            return relu(fn)

    def build_model(self, batch_size):
        self.x = tf.placeholder(tf.float32, shape=[batch_size] + self.input_shape.as_list(), name="sonar")
        self.y = tf.placeholder(tf.float32, shape=[batch_size] + self.output_shape.as_list(), name="satellite")

        # Network Architecture
        self.fn = self.build_generator(self.x)
        real_logits, real = self.build_discriminator(Concatenate()([self.x, self.y]), reuse=False)
        fake_logits, fake = self.build_discriminator(Concatenate()([self.x, self.fn]), reuse=True)
        conv21, conv21_gt = vgg16(self.config, self.fn, self.y, layers=["relu21"])

        with K.name_scope("loss"):
            self.real_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits,
                                                                                       labels=tf.ones_like(real)))
            self.fake_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits,
                                                                                       labels=tf.zeros_like(fake)))
            self.disc_entropy = self.real_entropy + self.fake_entropy

            self.discriminator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits,
                                                                                             labels=tf.ones_like(fake)))
            self.pixel_loss = 5e1 * tf.reduce_mean(tf.abs(self.fn - self.y))
            self.reconstruction_loss = 2.5e-7 * style_loss(conv21, conv21_gt)
            self.cross_entropy = self.discriminator_loss + self.pixel_loss + self.reconstruction_loss

            disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                discriminator_step = tf.train.AdamOptimizer(self.config.learning_rate, beta1=0.5).minimize(
                    loss=self.disc_entropy,
                    global_step=None,
                    var_list=disc_vars)

            gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
            with tf.control_dependencies([discriminator_step]):
                generator_step = tf.train.AdamOptimizer(self.config.learning_rate, beta1=0.5).minimize(
                    loss=self.cross_entropy,
                    global_step=self.global_step,
                    var_list=gen_vars)

            self.train_step = tf.group(discriminator_step, generator_step)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
