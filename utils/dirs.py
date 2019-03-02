# dirs.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf


def create_dirs(config, dirs):
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
            else:
                if config.mode == "train":
                    tf.logging.error("Cannot train because directory already exists")
                    raise RuntimeError
        return 0
    except Exception as err:
        tf.logging.error("Error creating directories: {0}".format(err))
        raise RuntimeError
