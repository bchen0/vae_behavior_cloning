import os

import architectures.vae_with_prediction as vae_model

import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, Dense, Flatten


def create_nature_cnn(num_actions, l2):
    inputs = keras.Input(shape=(84, 84, 4))
    x = Conv2D(32, 8, activation="relu", strides=4,
               kernel_regularizer=regularizers.l2(l2))(inputs)
    x = Conv2D(64, 4, activation="relu", strides=2,
               kernel_regularizer=regularizers.l2(l2))(x)
    x = Conv2D(64, 3, activation="relu", strides=1,
               kernel_regularizer=regularizers.l2(l2))(x)
    x = Flatten()(x)
    x = Dense(512, activation="relu",
              kernel_regularizer=regularizers.l2(l2))(x)
    x = Dense(num_actions, activation="softmax",
              kernel_regularizer=regularizers.l2(l2))(x)
    nature_cnn = Model(inputs, x, name="nature_cnn")
    return nature_cnn


def train_nature_cnn(train_data, val_data, epoch_size, num_actions,
                     lr=0.01, l2=0.001, epochs=50, batch_size=32,
                     log_dir=None, decay_type=None):
    nature_cnn = create_nature_cnn(num_actions, l2)
    if decay_type is None:
        opt = keras.optimizers.Adam(learning_rate=lr)
    elif decay_type == 'cos':
        lr_schedule = keras.experimental.CosineDecay(lr, epochs * epoch_size)
        opt = keras.optimizers.Adam(learning_rate=lr_schedule)
    nature_cnn.compile(optimizer=opt, loss='categorical_crossentropy',
                       metrics=['categorical_accuracy'])
    nature_cnn.summary()

    if log_dir is not None:
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            os.path.join(log_dir,
                         r'checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5')
        )
        csv_callback = keras.callbacks.CSVLogger(
            os.path.join(log_dir, 'training_log.csv'))

        train_params = {
            'callbacks': [tensorboard_callback,
                          checkpoint_callback,
                          csv_callback]
        }
    else:
        train_params = {}

    nature_cnn.fit(train_data, validation_data=val_data,
                   epochs=epochs, batch_size=batch_size, **train_params)
    nature_cnn.save(os.path.join(log_dir, 'model'))
    return nature_cnn


def train_only_predictor(train_data, val_data, dim, epoch_size,
                         num_actions, lr=0.01, l2=0.001, hidden_layers=1,
                         epochs=50, batch_size=32, log_dir=None,
                         decay_type=None):
    """
    Train just the predictor, given the encoder.
    """
    # Create the predictor
    model = vae_model.create_predictor(dim, num_actions, l2=l2,
                                       hidden_layers=hidden_layers)

    if decay_type is None:
        opt = keras.optimizers.Adam(learning_rate=lr)
    elif decay_type == 'cos':
        lr_schedule = keras.experimental.CosineDecay(lr, epochs * epoch_size)
        opt = keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    if log_dir is not None:
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            os.path.join(log_dir,
                         r'checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5')
        )
        csv_callback = keras.callbacks.CSVLogger(
            os.path.join(log_dir, 'training_log.csv'))

        train_params = {
            'callbacks': [tensorboard_callback,
                          checkpoint_callback,
                          csv_callback]
        }
    else:
        train_params = {}

    model.fit(train_data, validation_data=val_data,
              epochs=epochs, batch_size=batch_size, **train_params)
    model.save(os.path.join(log_dir, 'model'))
    return model
