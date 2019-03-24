# trainer.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random as rand
import tensorflow as tf

from base.base_trainer import BaseTrainer
from keras import backend as K
from tqdm import tqdm


class Trainer(BaseTrainer):
    def __init__(self, config, data, logger, model, session):
        super(Trainer, self).__init__(config, data, logger, model, session)

    def evaluate_data(self, data, save_fn):
        loop = tqdm(range(len(data)))
        idx = 0

        for _ in loop:
            loop.set_description("Evaluating Image [{}/{}]".format(idx, len(data)))

            feed_dict = {self.model.x: data[idx], K.learning_phase(): 0}
            result = self.session.run(self.model.fn, feed_dict=feed_dict)

            save_fn(self.config.evaluate_dir, idx, result)
            idx += 1

    def train_epoch(self):
        loop = tqdm(range(self.data.num_images // self.config.batch_size))
        loop.set_description("Training Epoch [{}/{}]".format(self.model.epoch.eval(self.session),
                                                             self.config.num_epochs))

        err_list = []
        r_err_list = []
        f_err_list = []
        d_err_list = []
        r_loss_list = []
        d_loss_list = []
        for _ in loop:
            err, r_err, f_err, d_err, r_loss, d_loss = self.train_step()

            # Append Data
            err_list.append(err)
            r_err_list.append(r_err)
            f_err_list.append(f_err)
            d_err_list.append(d_err)
            r_loss_list.append(r_loss)
            d_loss_list.append(d_loss)

            self.data.idx += 1
        self.data.idx = 0

        it = self.model.global_step.eval(self.session)
        generator_dict = {
            "discriminator_loss": np.mean(d_loss_list),
            "loss": np.mean(err_list),
            "reconstruction_loss": np.mean(r_loss_list),
        }
        discriminator_dict = {
            "fake_entropy": np.mean(f_err_list),
            "entropy": np.mean(d_err_list),
            "real_entropy": np.mean(r_err_list),
        }

        self.logger.summarize(it, summarizer="train", scope="generative", summaries_dict=generator_dict)
        self.logger.summarize(it, summarizer="train", scope="discriminative", summaries_dict=discriminator_dict)

    def train_step(self):
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size, is_validation=False))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, K.learning_phase(): 1}

        _, err, r_err, f_err, d_err, r_loss, d_loss = self.session.run([self.model.train_step,
                                                                        self.model.cross_entropy,
                                                                        self.model.real_entropy,
                                                                        self.model.fake_entropy,
                                                                        self.model.disc_entropy,
                                                                        self.model.reconstruction_loss,
                                                                        self.model.discriminator_loss],
                                                                       feed_dict=feed_dict)
        return err, r_err, f_err, d_err, r_loss, d_loss

    def validate_epoch(self):
        loop = tqdm(range(self.data.num_images_val // self.config.batch_size))
        loop.set_description("Validating Epoch {}".format(self.model.epoch.eval(self.session)))

        err_list = []
        r_loss_list = []
        d_loss_list = []
        fn_list = []
        y_list = []
        x_list = []
        for _ in loop:
            err, r_loss, d_loss, fn, y, x = self.validate_step()

            # Append Data
            err_list.append(err)
            r_loss_list.append(r_loss)
            d_loss_list.append(d_loss)
            fn_list.append(fn)
            y_list.append(y)
            x_list.append(x)

            self.data.idx += 1
        self.data.idx = 0

        batch = rand.choice(range(len(fn_list)))
        it = self.model.global_step.eval(self.session)
        data_dict = {
            "discriminator_loss": np.mean(d_loss_list),
            "loss": np.mean(err_list),
            "reconstruction_loss": np.mean(r_loss_list),
        }
        image_dict = {
            "fake": fn_list[batch],
            "input": x_list[batch],
            "real": y_list[batch],
        }

        self.logger.summarize(it, summarizer="validation", scope="generative", summaries_dict=data_dict)
        self.logger.summarize(it, summarizer="validation", scope="", summaries_dict=image_dict)
        self.model.save(self.session)

    def validate_step(self):
        batch_x_val, batch_y_val = next(self.data.next_batch(self.config.batch_size, is_validation=True))
        feed_dict = {self.model.x: batch_x_val, self.model.y: batch_y_val, K.learning_phase(): 0}

        err, r_loss, d_loss, fn, y, x = self.session.run([self.model.cross_entropy,
                                                          self.model.reconstruction_loss,
                                                          self.model.discriminator_loss,
                                                          self.model.fn,
                                                          self.model.y,
                                                          self.model.x],
                                                         feed_dict=feed_dict)
        return err, r_loss, d_loss, fn, y, x
