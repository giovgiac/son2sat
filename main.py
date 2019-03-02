# main.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import tensorflow as tf

from datasets.aracati import Aracati
from keras import backend as K
from models.unet import Unet
from trainers.trainer import Trainer
from utils.logger import Logger

from utils.config import process_config
from utils.dirs import create_dirs


def main(unused_argv):
    tf.logging.set_verbosity(3)

    try:
        config = process_config()

    except:
        exit(0)

    create_dirs(config, [config.checkpoint_dir, config.evaluate_dir, config.presentation_dir, config.summary_dir])

    session = tf.Session()
    K.set_session(session)

    if config.mode == "evaluate":
        model = Unet(config, is_evaluating=True)
        trainer = Trainer(config, None, None, model, session)

        son_data = [Aracati.load_data(file, is_sonar=True) for file in
                    sorted(glob.glob("./datasets/aracati/test/input/*.png"))]
        son_data = [son_data[i:i+1] for i in range(len(son_data))]

        model.load(session)
        trainer.evaluate_data(son_data, Aracati.save_data)

    else:
        data = Aracati(config)
        model = Unet(config)
        logger = Logger(config, session)
        trainer = Trainer(config, data, logger, model, session)

        if config.mode == "restore":
            model.load(session)

        trainer.train()


if __name__ == "__main__":
    tf.app.run()
