# base_trainer.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class BaseTrainer(object):
    def __init__(self, config, data, logger, model, session):
        self.config = config
        self.data = data
        self.logger = logger
        self.model = model
        self.session = session

        self.session.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

    def train(self):
        for epoch in range(self.model.epoch.eval(self.session), self.config.num_epochs + 1, 1):
            self.train_epoch()
            self.validate_epoch()
            self.session.run(self.model.increment_epoch)

    def evaluate_data(self, data, save_fn):
        raise NotImplementedError

    def train_epoch(self):
        raise NotImplementedError

    def train_step(self):
        raise NotImplementedError

    def validate_epoch(self):
        raise NotImplementedError

    def validate_step(self):
        raise NotImplementedError
