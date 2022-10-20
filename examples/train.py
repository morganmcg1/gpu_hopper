import random
import numpy as np
import tensorflow as tf
from fastcore.script import *

import wandb
from wandb.keras import WandbCallback

from configs.utils import import_config

@call_parse
def main(
        wandb_run_id:Param("wandb run id from", str)=None,
        config_path:Param("config file to use", str)="configs/baseline.py"
        ):

    # Get the config from wandb
    config = import_config(config_path="configs/baseline.py")

    # Resume the run
    wandb.init(
        id=wandb_run_id,
        resume="must",
        entity=config.wandb.entity,
        project=config.wandb.project
    )

    # Start training

    # Get the data
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, y_train = x_train[::5], y_train[::5]  # Subset data for a faster demo
    x_test, y_test = x_test[::20], y_test[::20]
    # labels = [str(digit) for digit in range(np.max(y_train) + 1)]

    # Build a model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(config.model.layer_1, activation=config.model.activation_1),
        tf.keras.layers.Dropout(config.model.dropout),
        tf.keras.layers.Dense(config.model.layer_2, activation=config.model.activation_2)
        ])

    model.compile(optimizer=config.train.optimizer,
                    loss=config.train.loss_func,
                    metrics=[config.train.metric]
                    )

    # WandbCallback auto-saves all metrics from model.fit(), plus predictions on validation_data
    logging_callback = WandbCallback(log_evaluation=True)

    model.fit(x=x_train, y=y_train,
                        epochs=config.train.epoch,
                        batch_size=config.train.batch_size,
                        validation_data=(x_test, y_test),
                        callbacks=[logging_callback]
                        )

    # wandb.finish()