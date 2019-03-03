# logger.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf


class Logger(object):
    def __init__(self, config, session):
        self.config = config
        self.session = session
        self.summary_placeholders = {}
        self.summary_ops = {}

        train_path = os.path.join(self.config.summary_dir, "train")
        print("tensorboard --logdir='{}'".format(os.path.abspath(self.config.summary_dir)))

        self.summary_writer = tf.summary.FileWriter(train_path, self.session.graph)
        self.summary_writer_val = tf.summary.FileWriter(os.path.join(self.config.summary_dir, "validation"))

    def summarize(self, step, summarizer="train", scope="", summaries_dict=None):
        summary_writer = self.summary_writer if summarizer == "train" else self.summary_writer_val

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            if summaries_dict is not None:
                summary_list = []
                for tag, value in summaries_dict.items():
                    if tag not in self.summary_ops:
                        if len(value.shape) <= 1:
                            self.summary_placeholders[tag] = tf.placeholder("float32",
                                                                            value.shape,
                                                                            name=None)
                        else:
                            self.summary_placeholders[tag] = tf.placeholder("float32",
                                                                            [None] + list(value.shape[1:]),
                                                                            name=None)
                        if len(value.shape) <= 1:
                            self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])
                        else:
                            self.summary_ops[tag] = tf.summary.image(tag, self.summary_placeholders[tag], max_outputs=6)

                    summary_list.append(
                        self.session.run(self.summary_ops[tag], {self.summary_placeholders[tag]: value}))

                for summary in summary_list:
                    summary_writer.add_summary(summary, step)
                summary_writer.flush()
