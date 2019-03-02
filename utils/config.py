# config.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import time

# Configuration Entries
tf.flags.DEFINE_string("mode", "train", "Choose one from: evaluate, restore or train.")
tf.flags.DEFINE_string("name", "auto",
                       "Name of the folder to store the files of the experiment. "
                       "Default value is auto, which is an algorithm that automatically generates a name.")
tf.flags.DEFINE_string("checkpoint_dir", "", "")
tf.flags.DEFINE_string("evaluate_dir", "", "")
tf.flags.DEFINE_string("presentation_dir", "", "")
tf.flags.DEFINE_string("summary_dir", "", "")
tf.flags.DEFINE_integer("batch_size", 16, "Batch size to use for the network.")
tf.flags.DEFINE_integer("gen_filters", 64, "Parameter that scales the size of the network.")
tf.flags.DEFINE_integer("input_channels", 1, "Number of channels in the input images.")
tf.flags.DEFINE_integer("image_height", 256, "Height of the images to run through the network.")
tf.flags.DEFINE_integer("image_width", 512, "Width of the images to run through the network.")
tf.flags.DEFINE_integer("max_to_keep", 1, "Maximum number of checkpoints to keep.")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of epochs to execute the network for.")
tf.flags.DEFINE_integer("output_channels", 3, "Number of channels in the output images.")
tf.flags.DEFINE_float("learning_rate", 0.0002, "Initial learning rate for the optimizer.")

FLAGS = tf.flags.FLAGS


def process_config():
    config = FLAGS
    if config.mode == "evaluate" and config.name == "auto":
        tf.logging.error("Cannot automatically choose a name for an evaluation")
        raise NameError
    elif config.mode == "restore" and config.name == "auto":
        tf.logging.error("Cannot automatically choose a name for a restoration")
        raise NameError
    elif config.name == "auto":
        config.name = time.strftime("%Y-%m-%d_%H:%M")

    config.checkpoint_dir = os.path.join("./executions", config.name, "checkpoint/")
    config.presentation_dir = os.path.join("./executions", config.name, "presentations/")
    config.evaluate_dir = os.path.join("./executions", config.name, "results/")
    config.summary_dir = os.path.join("./executions", config.name, "summary/")

    return config
