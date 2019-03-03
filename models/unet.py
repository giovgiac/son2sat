# unet.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from base.base_model import BaseModel
from keras import backend as K
from keras.activations import sigmoid, relu
from keras.layers import Concatenate, Conv2D, Conv2DTranspose, Dense, Dropout, LeakyReLU, ReLU, Reshape
from keras.objectives import binary_crossentropy, mean_absolute_error

from layers.guided_filter import guided_filter
from layers.reconstruction_loss import feature_loss, style_loss


class Unet(BaseModel):
    def __init__(self, config, is_evaluating=False):
        super(Unet, self).__init__(config, is_evaluating)

        # Inputs
        self.x = None
        self.y = None

        # Losses
        self.discriminator_loss = None
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
                self.disc_layers['h0_conv'] = Conv2D(filters=self.gen_filters, kernel_size=(5, 5), strides=(2, 2),
                                                     padding='same')
                self.disc_layers['h0_actv'] = LeakyReLU(alpha=0.2)

                # Second Layer
                self.disc_layers['h1_conv'] = Conv2D(filters=self.gen_filters * 2, kernel_size=(5, 5), strides=(2, 2),
                                                     padding='same')
                self.disc_layers['h1_actv'] = LeakyReLU(alpha=0.2)

                # Third Layer
                self.disc_layers['h2_conv'] = Conv2D(filters=self.gen_filters * 4, kernel_size=(5, 5), strides=(2, 2),
                                                     padding='same')
                self.disc_layers['h2_actv'] = LeakyReLU(alpha=0.2)

                # Fourth Layer
                self.disc_layers['h3_conv'] = Conv2D(filters=self.gen_filters * 8, kernel_size=(5, 5), strides=(1, 1),
                                                     padding='same')
                self.disc_layers['h3_actv'] = LeakyReLU(alpha=0.2)

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
        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
            with K.name_scope("encode1"):
                e1 = Conv2D(filters=self.gen_filters, kernel_size=5, strides=2, padding='same')(x)
                e1 = LeakyReLU(alpha=0.2)(e1)

            with K.name_scope("encode2"):
                e2 = Conv2D(filters=self.gen_filters * 2, kernel_size=5, strides=2, padding='same')(e1)
                e2 = tf.contrib.layers.batch_norm(e2, decay=0.9, epsilon=1e-5, updates_collections=None, scale=True)
                e2 = LeakyReLU(alpha=0.2)(e2)

            with K.name_scope("encode3"):
                e3 = Conv2D(filters=self.gen_filters * 4, kernel_size=5, strides=2, padding='same')(e2)
                e3 = tf.contrib.layers.batch_norm(e3, decay=0.9, epsilon=1e-5, updates_collections=None, scale=True)
                e3 = LeakyReLU(alpha=0.2)(e3)

            with K.name_scope("encode4"):
                e4 = Conv2D(filters=self.gen_filters * 8, kernel_size=5, strides=2, padding='same')(e3)
                e4 = tf.contrib.layers.batch_norm(e4, decay=0.9, epsilon=1e-5, updates_collections=None, scale=True)
                e4 = LeakyReLU(alpha=0.2)(e4)

            with K.name_scope("encode5"):
                e5 = Conv2D(filters=self.gen_filters * 8, kernel_size=5, strides=2, padding='same')(e4)
                e5 = tf.contrib.layers.batch_norm(e5, decay=0.9, epsilon=1e-5, updates_collections=None, scale=True)
                e5 = LeakyReLU(alpha=0.2)(e5)

            with K.name_scope("encode6"):
                e6 = Conv2D(filters=self.gen_filters * 8, kernel_size=5, strides=2, padding='same')(e5)
                e6 = tf.contrib.layers.batch_norm(e6, decay=0.9, epsilon=1e-5, updates_collections=None, scale=True)
                e6 = LeakyReLU(alpha=0.2)(e6)

            with K.name_scope("encode7"):
                e7 = Conv2D(filters=self.gen_filters * 8, kernel_size=5, strides=2, padding='same')(e6)
                e7 = tf.contrib.layers.batch_norm(e7, decay=0.9, epsilon=1e-5, updates_collections=None, scale=True)
                e7 = LeakyReLU(alpha=0.2)(e7)

            with K.name_scope("encode8"):
                e8 = Conv2D(filters=self.gen_filters * 8, kernel_size=5, strides=2, padding='same')(e7)
                e8 = tf.contrib.layers.batch_norm(e8, decay=0.9, epsilon=1e-5, updates_collections=None, scale=True)
                e8 = ReLU()(e8)

            with K.name_scope("decode1"):
                d1 = Conv2DTranspose(filters=self.gen_filters * 8, kernel_size=5, strides=2, padding='same')(e8)
                d1 = tf.contrib.layers.batch_norm(d1, decay=0.9, epsilon=1e-5, updates_collections=None, scale=True)
                d1 = Dropout(rate=0.5)(d1)
                d1 = Concatenate()([d1, e7])
                d1 = ReLU()(d1)

            with K.name_scope("decode2"):
                d2 = Conv2DTranspose(filters=self.gen_filters * 8, kernel_size=5, strides=2, padding='same')(d1)
                d2 = tf.contrib.layers.batch_norm(d2, decay=0.9, epsilon=1e-5, updates_collections=None, scale=True)
                d2 = Dropout(rate=0.5)(d2)
                d2 = Concatenate()([d2, e6])
                d2 = ReLU()(d2)

            with K.name_scope("decode3"):
                d3 = Conv2DTranspose(filters=self.gen_filters * 8, kernel_size=5, strides=2, padding='same')(d2)
                d3 = tf.contrib.layers.batch_norm(d3, decay=0.9, epsilon=1e-5, updates_collections=None, scale=True)
                d3 = Dropout(rate=0.5)(d3)
                d3 = Concatenate()([d3, e5])
                d3 = ReLU()(d3)

            with K.name_scope("decode4"):
                d4 = Conv2DTranspose(filters=self.gen_filters * 8, kernel_size=5, strides=2, padding='same')(d3)
                d4 = tf.contrib.layers.batch_norm(d4, decay=0.9, epsilon=1e-5, updates_collections=None, scale=True)
                d4 = Concatenate()([d4, e4])
                d4 = ReLU()(d4)

            with K.name_scope("decode5"):
                d5 = Conv2DTranspose(filters=self.gen_filters * 4, kernel_size=5, strides=2, padding='same')(d4)
                d5 = tf.contrib.layers.batch_norm(d5, decay=0.9, epsilon=1e-5, updates_collections=None, scale=True)
                d5 = Concatenate()([d5, e3])
                d5 = ReLU()(d5)

            with K.name_scope("decode6"):
                d6 = Conv2DTranspose(filters=self.gen_filters * 2, kernel_size=5, strides=2, padding='same')(d5)
                d6 = tf.contrib.layers.batch_norm(d6, decay=0.9, epsilon=1e-5, updates_collections=None, scale=True)
                d6 = Concatenate()([d6, e2])
                d6 = ReLU()(d6)

            with K.name_scope("decode7"):
                d7 = Conv2DTranspose(filters=self.gen_filters, kernel_size=5, strides=2, padding='same')(d6)
                d7 = tf.contrib.layers.batch_norm(d7, decay=0.9, epsilon=1e-5, updates_collections=None, scale=True)
                d7 = Concatenate()([d7, e1])
                d7 = ReLU()(d7)

            with K.name_scope("decode8"):
                d8 = Conv2DTranspose(filters=self.output_shape.as_list()[-1], kernel_size=5, strides=2, padding='same')(d7)

            with K.name_scope("guided1"):
                son = tf.image.grayscale_to_rgb(x)

                # Guided Convolution
                c1 = Conv2D(filters=self.output_shape.as_list()[-1], kernel_size=3, strides=1, padding='same')(d8)
                c1 = tf.contrib.layers.batch_norm(c1, decay=0.9, epsilon=1e-5, updates_collections=None, scale=True)
                c1 = ReLU()(c1)

                # Guided Filter
                g1 = guided_filter(x=c1, y=son, r=20, eps=1e-4, nhwc=True)
                g1 = Concatenate()([g1 * d8, d8])

            # Final Convolution
            fn = Conv2D(filters=self.output_shape.as_list()[-1], kernel_size=1, strides=1, padding='same')(g1)

            return relu(fn)

    def build_model(self, batch_size):
        self.x = tf.placeholder(tf.float32, shape=[batch_size] + self.input_shape.as_list(), name="sonar")
        self.y = tf.placeholder(tf.float32, shape=[batch_size] + self.output_shape.as_list(), name="satellite")

        # Network Architecture
        self.fn = self.build_generator(self.x)
        real_logits, real = self.build_discriminator(Concatenate()([self.x, self.y]), reuse=False)
        fake_logits, fake = self.build_discriminator(Concatenate()([self.x, self.fn]), reuse=True)

        with K.name_scope("loss"):
            self.real_entropy = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones_like(real)))
            self.fake_entropy = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros_like(fake)))
            self.disc_entropy = self.real_entropy + self.fake_entropy

            self.discriminator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.ones_like(fake))
            self.reconstruction_loss = 10e6 * style_loss(self.config, self.fn, self.y, layers=["relu12"])
            self.cross_entropy = tf.reduce_mean(self.discriminator_loss + self.reconstruction_loss)

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
