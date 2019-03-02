# base_model.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class BaseModel(object):
    def __init__(self, config, is_evaluating=False):
        self.config = config
        self.epoch = None
        self.global_step = None
        self.increment_epoch = None
        self.is_evaluating = is_evaluating
        self.saver = None

        self.init_epoch()
        self.init_step()

    def init_epoch(self):
        with tf.variable_scope("current_epoch"):
            self.epoch = tf.get_variable(name="current_epoch", dtype=tf.int32, shape=(), trainable=False,
                                         initializer=tf.zeros_initializer())
            self.increment_epoch = tf.assign(self.epoch, self.epoch + 1)

    def init_step(self):
        with tf.variable_scope("global_step"):
            self.global_step = tf.get_variable(name="global_step", shape=(), trainable=False,
                                               initializer=tf.zeros_initializer())

    def load(self, session):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            tf.logging.info("Loading Model Checkpoint {}".format(latest_checkpoint))
            self.saver.restore(session, latest_checkpoint)
            tf.logging.info("Model Loaded")

    def save(self, session):
        tf.logging.info("Saving Model")
        self.saver.save(session, self.config.checkpoint_dir, self.global_step)
        tf.logging.info("Model Saved")

    def build_model(self, batch_size):
        raise NotImplementedError

    def init_saver(self):
        raise NotImplementedError
