# unet.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from base.base_model import BaseModel
from keras import backend as K
from keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Conv2DTranspose, Dense, Dropout, \
    LeakyReLU, MaxPool2D, ReLU, Reshape
from keras.objectives import binary_crossentropy, mean_absolute_error

from layers.guided_filter import guided_filter

class Unet(BaseModel):
    def __init__(self, config, is_evaluating=False):
        super(Unet, self).__init__(config, is_evaluating)

        self.x = None
        self.y = None
        self.cross_entropy = None
        self.disc_entropy = None
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

    def build_discriminator(self, x, batch_size):
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            # First Convolution
            h0 = Conv2D(filters=self.gen_filters, kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
            h0 = LeakyReLU(alpha=0.2)(h0)

            # Second Convolution
            h1 = Conv2D(filters=self.gen_filters * 2, kernel_size=(5, 5), strides=(2, 2), padding='same')(h0)
            h1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(h1)
            h1 = LeakyReLU(alpha=0.2)(h1)

            # Third Convolution
            h2 = Conv2D(filters=self.gen_filters * 4, kernel_size=(5, 5), strides=(2, 2), padding='same')(h1)
            h2 = BatchNormalization(momentum=0.9, epsilon=1e-5)(h2)
            h2 = LeakyReLU(alpha=0.2)(h2)

            # Fourth Convolution
            h3 = Conv2D(filters=self.gen_filters * 8, kernel_size=(5, 5), strides=(1, 1), padding='same')(h2)
            h3 = BatchNormalization(momentum=0.9, epsilon=1e-5)(h3)
            h3 = LeakyReLU(alpha=0.2)(h3)

            # Dense Layer
            h4 = Reshape(target_shape=(-1,))(h3)
            h4 = Dense(units=1)(h4)

            return h4, Activation("sigmoid")(h4)

    def build_generator(self, x, batch_size):
        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
            with K.name_scope("Encode1"):
                e1 = Conv2D(filters=self.gen_filters, kernel_size=5, strides=2, padding='same')(x)
                e1 = LeakyReLU(alpha=0.2)(e1)

            with K.name_scope("Encode2"):
                e2 = Conv2D(filters=self.gen_filters * 2, kernel_size=5, strides=2, padding='same')(e1)
                e2 = BatchNormalization(momentum=0.9, epsilon=1e-5)(e2)
                e2 = LeakyReLU(alpha=0.2)(e2)

            with K.name_scope("Encode3"):
                e3 = Conv2D(filters=self.gen_filters * 4, kernel_size=5, strides=2, padding='same')(e2)
                e3 = BatchNormalization(momentum=0.9, epsilon=1e-5)(e3)
                e3 = LeakyReLU(alpha=0.2)(e3)

            with K.name_scope("Encode4"):
                e4 = Conv2D(filters=self.gen_filters * 8, kernel_size=5, strides=2, padding='same')(e3)
                e4 = BatchNormalization(momentum=0.9, epsilon=1e-5)(e4)
                e4 = LeakyReLU(alpha=0.2)(e4)

            with K.name_scope("Encode5"):
                e5 = Conv2D(filters=self.gen_filters * 8, kernel_size=5, strides=2, padding='same')(e4)
                e5 = BatchNormalization(momentum=0.9, epsilon=1e-5)(e5)
                e5 = LeakyReLU(alpha=0.2)(e5)

            with K.name_scope("Encode6"):
                e6 = Conv2D(filters=self.gen_filters * 8, kernel_size=5, strides=2, padding='same')(e5)
                e6 = BatchNormalization(momentum=0.9, epsilon=1e-5)(e6)
                e6 = LeakyReLU(alpha=0.2)(e6)

            with K.name_scope("Encode7"):
                e7 = Conv2D(filters=self.gen_filters * 8, kernel_size=5, strides=2, padding='same')(e6)
                e7 = BatchNormalization(momentum=0.9, epsilon=1e-5)(e7)
                e7 = LeakyReLU(alpha=0.2)(e7)

            with K.name_scope("Encode8"):
                e8 = Conv2D(filters=self.gen_filters * 8, kernel_size=5, strides=2, padding='same')(e7)
                e8 = BatchNormalization(momentum=0.9, epsilon=1e-5)(e8)
                e8 = ReLU()(e8)

            with K.name_scope("Decode1"):
                d1 = Conv2DTranspose(filters=self.gen_filters * 8, kernel_size=5, strides=2, padding='same')(e8)
                d1 = BatchNormalization(momentum=0.9, epsilon=1e-5)(d1)
                d1 = Dropout(rate=0.5)(d1)
                d1 = Concatenate()([d1, e7])
                d1 = ReLU()(d1)

            with K.name_scope("Decode2"):
                d2 = Conv2DTranspose(filters=self.gen_filters * 8, kernel_size=5, strides=2, padding='same')(d1)
                d2 = BatchNormalization(momentum=0.9, epsilon=1e-5)(d2)
                d2 = Dropout(rate=0.5)(d2)
                d2 = Concatenate()([d2, e6])
                d2 = ReLU()(d2)

            with K.name_scope("Decode3"):
                d3 = Conv2DTranspose(filters=self.gen_filters * 8, kernel_size=5, strides=2, padding='same')(d2)
                d3 = BatchNormalization(momentum=0.9, epsilon=1e-5)(d3)
                d3 = Dropout(rate=0.5)(d3)
                d3 = Concatenate()([d3, e5])
                d3 = ReLU()(d3)

            with K.name_scope("Decode4"):
                d4 = Conv2DTranspose(filters=self.gen_filters * 8, kernel_size=5, strides=2, padding='same')(d3)
                d4 = BatchNormalization(momentum=0.9, epsilon=1e-5)(d4)
                d4 = Concatenate()([d4, e4])
                d4 = ReLU()(d4)

            with K.name_scope("Decode5"):
                d5 = Conv2DTranspose(filters=self.gen_filters * 4, kernel_size=5, strides=2, padding='same')(d4)
                d5 = BatchNormalization(momentum=0.9, epsilon=1e-5)(d5)
                d5 = Concatenate()([d5, e3])
                d5 = ReLU()(d5)

            with K.name_scope("Decode6"):
                d6 = Conv2DTranspose(filters=self.gen_filters * 2, kernel_size=5, strides=2, padding='same')(d5)
                d6 = BatchNormalization(momentum=0.9, epsilon=1e-5)(d6)
                d6 = Concatenate()([d6, e2])
                d6 = ReLU()(d6)

            with K.name_scope("Decode7"):
                d7 = Conv2DTranspose(filters=self.gen_filters, kernel_size=5, strides=2, padding='same')(d6)
                d7 = BatchNormalization(momentum=0.9, epsilon=1e-5)(d7)
                d7 = Concatenate()([d7, e1])
                d7 = ReLU()(d7)

            with K.name_scope("Decode8"):
                d8 = Conv2DTranspose(filters=self.output_shape.as_list()[-1], kernel_size=5, strides=2, padding='same')(d7)
                d8 = ReLU()(d8)

            with K.name_scope("Guided"):
                son = tf.image.grayscale_to_rgb(x)

                g1 = guided_filter(x=d8, y=son, r=20, eps=1e-4)
                g1 = Concatenate()([g1 * d8, d8])

            # Final Convolution
            fn = Conv2D(filters=self.output_shape.as_list()[-1], kernel_size=1, strides=1, padding='same')(g1)

            return ReLU()(fn)

    def build_model(self, batch_size):
        self.x = tf.placeholder(tf.float32, shape=[batch_size] + self.input_shape.as_list())
        self.y = tf.placeholder(tf.float32, shape=[batch_size] + self.output_shape.as_list())

        # Network Architecture
        self.fn = self.build_generator(self.x, batch_size)
        real_logits, real = self.build_discriminator(Concatenate()([self.x, self.y]), batch_size)
        fake_logits, fake = self.build_discriminator(Concatenate()([self.x, self.fn]), batch_size)

        with K.name_scope("loss"):
            real_loss = tf.reduce_mean(binary_crossentropy(tf.ones_like(real), real_logits))
            fake_loss = tf.reduce_mean(binary_crossentropy(tf.zeros_like(fake), fake_logits))

            self.cross_entropy = tf.reduce_mean(binary_crossentropy(tf.ones_like(fake), fake_logits)) + \
                                 100.0 * tf.reduce_mean(mean_absolute_error(self.y, self.fn))
            self.disc_entropy = real_loss + fake_loss

            disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                discriminator_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.disc_entropy,
                                                                                                None,
                                                                                                disc_vars)

            gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
            with tf.control_dependencies([discriminator_step]):
                generator_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cross_entropy,
                                                                                            self.global_step,
                                                                                            gen_vars)

            self.train_step = tf.group(discriminator_step, generator_step)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)