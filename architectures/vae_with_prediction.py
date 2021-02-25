import math
import os


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, Reshape, Flatten, \
    Conv2DTranspose


# Code to create an encoder, decoder, and predictor, which are used by the VAE.
class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def create_encoder(latent_dim, l2=0.001, extra_conv_layers=0):
    encoder_inputs = keras.Input(shape=(84, 84, 4))
    x = Conv2D(32, 3, activation="relu", strides=2, padding="same",
               kernel_regularizer=regularizers.l2(l2))(encoder_inputs)
    x = Conv2D(64, 3, activation="relu", strides=2, padding="same",
               kernel_regularizer=regularizers.l2(l2))(x)
    for _ in range(extra_conv_layers):
        x = Conv2D(64, 3, activation="relu", strides=1, padding="same",
                   kernel_regularizer=regularizers.l2(l2))(x)
    x = Flatten()(x)
    x = Dense(latent_dim, activation="relu",
              kernel_regularizer=regularizers.l2(l2))(x)
    z_mean = Dense(latent_dim, name="z_mean",
                   kernel_regularizer=regularizers.l2(l2))(x)
    z_log_var = Dense(latent_dim, name="z_log_var",
                      kernel_regularizer=regularizers.l2(l2))(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()
    return encoder


def create_decoder(latent_dim, l2=0.001, extra_conv_layers=0):
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = Dense(21 * 21 * 64, activation="relu",
              kernel_regularizer=regularizers.l2(l2))(latent_inputs)
    x = Reshape((21, 21, 64))(x)

    for _ in range(extra_conv_layers):
        x = Conv2DTranspose(64, 3, activation="relu", strides=1,
                            padding="same",
                            kernel_regularizer=regularizers.l2(l2))(x)

    x = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same",
                        kernel_regularizer=regularizers.l2(l2))(x)
    decoder_outputs = Conv2DTranspose(4, 3, activation="sigmoid", strides=2,
                                      padding="same",
                                      kernel_regularizer=regularizers.l2(
                                          l2))(x)

    decoder = Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()
    return decoder


def create_predictor(latent_dim, num_categories, l2=0.001, hidden_layers=None):
    latent_inputs = keras.Input(shape=(latent_dim,))

    if hidden_layers is not None and hidden_layers > 0:
        for _ in range(hidden_layers):
            x = Dense(latent_dim, activation="relu",
                      kernel_regularizer=regularizers.l2(l2))(latent_inputs)
        x = Dense(num_categories, activation='softmax',
                  kernel_regularizer=regularizers.l2(l2))(x)
    else:
        x = Dense(num_categories, activation='softmax',
                  kernel_regularizer=regularizers.l2(l2))(latent_inputs)
    predictor = Model(latent_inputs, x, name="predictor")
    predictor.summary()
    return predictor


# --------------- VAE code ---------------


class VAE_w_Prediction(Model):
    def __init__(self, encoder, decoder, predictor, predictor_wt=1,
                 kl_div_wt=1, vae_wt=1, **kwargs):
        super(VAE_w_Prediction, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.predictor = predictor
        self.predictor_wt = predictor_wt
        self.kl_div_wt = kl_div_wt
        self.vae_wt = vae_wt

    def train_step(self, data):
        image, labels = data

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(
                image)
            reconstruction = self.decoder(z)
            predictions = self.predictor(z)

            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(image, reconstruction)
            )
            reconstruction_loss *= 84 * 84

            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5

            cce = tf.keras.losses.CategoricalCrossentropy()
            prediction_loss = cce(labels, predictions)
            reg_encoder_loss = tf.add_n(self.encoder.losses)
            reg_decoder_loss = tf.add_n(self.decoder.losses)
            reg_predictor_loss = tf.add_n(self.predictor.losses)
            regularization_loss = reg_encoder_loss + reg_decoder_loss + \
                                  reg_predictor_loss

            total_loss = (reconstruction_loss + self.kl_div_wt * kl_loss) * \
                         self.vae_wt + prediction_loss * self.predictor_wt + \
                         regularization_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.compiled_metrics.update_state(labels, predictions)
        output = {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "prediction_loss": prediction_loss,
            "regularization_predictor_loss": reg_predictor_loss,
            "regularization_encoder_loss": reg_encoder_loss,
            "regularization_decoder_loss": reg_decoder_loss,
            "regularization_loss_total": regularization_loss
        }
        #         output.update({m.name: m.result() for m in self.metrics[:-1]})
        output.update({m.name: m.result() for m in self.metrics})
        return output

    def test_step(self, data):
        # Need test_step to run validation test sets. See
        # https://github.com/keras-team/keras-io/issues/38
        # This is pretty much the same as train_step, excluding the gradient
        # tape.
        image, labels = data

        z_mean, z_log_var, z = self.encoder(image)
        reconstruction = self.decoder(z)
        predictions = self.predictor(z)

        reconstruction_loss = tf.reduce_mean(
            keras.losses.binary_crossentropy(image, reconstruction)
        )
        reconstruction_loss *= 84 * 84  # this is the total lozs based on
        # our output pixel shape (84,84)
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5

        cce = tf.keras.losses.CategoricalCrossentropy()
        prediction_loss = cce(labels, predictions)

        reg_encoder_loss = tf.add_n(self.encoder.losses)
        reg_decoder_loss = tf.add_n(self.decoder.losses)
        reg_predictor_loss = tf.add_n(self.predictor.losses)
        regularization_loss = reg_encoder_loss + reg_decoder_loss + \
                              reg_predictor_loss

        total_loss = (reconstruction_loss + self.kl_div_wt * kl_loss) * \
                     self.vae_wt + prediction_loss * self.predictor_wt + \
                     regularization_loss

        self.compiled_metrics.update_state(labels, predictions)

        output = {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "prediction_loss": prediction_loss,
            "regularization_predictor_loss": reg_predictor_loss,
            "regularization_encoder_loss": reg_encoder_loss,
            "regularization_decoder_loss": reg_decoder_loss,
            "regularization_loss_total": regularization_loss
        }
        output.update({m.name: m.result() for m in self.metrics})
        return output

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction


class VAE(Model):
    def __init__(self, encoder, decoder, kl_div_wt=1, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.kl_div_wt = kl_div_wt

    def train_step(self, data):
        image, labels = data

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(image)
            reconstruction = self.decoder(z)

            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(image, reconstruction)
            )
            reconstruction_loss *= 84 * 84

            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5

            reg_encoder_loss = tf.add_n(self.encoder.losses)
            reg_decoder_loss = tf.add_n(self.decoder.losses)
            regularization_loss = reg_encoder_loss + reg_decoder_loss

            total_loss = reconstruction_loss + self.kl_div_wt * kl_loss + \
                         regularization_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        output = {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "regularization_encoder_loss": reg_encoder_loss,
            "regularization_decoder_loss": reg_decoder_loss,
            "regularization_loss_total": regularization_loss
        }
        output.update({m.name: m.result() for m in self.metrics})
        return output

    def test_step(self, data):
        # Need test_step to run validation test sets. See
        # https://github.com/keras-team/keras-io/issues/38
        # This is pretty much the same as train_step, excluding the gradient
        # tape.
        image, labels = data

        z_mean, z_log_var, z = self.encoder(image)
        reconstruction = self.decoder(z)

        reconstruction_loss = tf.reduce_mean(
            keras.losses.binary_crossentropy(image, reconstruction)
        )
        reconstruction_loss *= 84 * 84  # this is the total lozs based on
        # our output pixel shape (84,84)
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5

        reg_encoder_loss = tf.add_n(self.encoder.losses)
        reg_decoder_loss = tf.add_n(self.decoder.losses)
        regularization_loss = reg_encoder_loss + reg_decoder_loss

        total_loss = reconstruction_loss + self.kl_div_wt * kl_loss + \
                     regularization_loss

        output = {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "regularization_encoder_loss": reg_encoder_loss,
            "regularization_decoder_loss": reg_decoder_loss,
            "regularization_loss_total": regularization_loss
        }
        output.update({m.name: m.result() for m in self.metrics})
        return output

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction


# --------------- A function for training multiple VAEs ---------------

def train_multiple_vaes(train_data, val_data, latent_dim_list, lr_list,
                        vae_l2_reg_list,
                        log_dir, save_dir,
                        num_actions,
                        epochs=20, batch_size=32, predictor_wt=1000,
                        vae_wt=1, lr_decay=None, lr_decay_steps=None,
                        lr_decay_staircase=True,
                        predict_hidden_layers=None,
                        cosine_decay_steps=None,
                        extra_conv_layers=0,
                        kl_div_wt=1,
                        num_predictors=1):
    """
    Creates the VAE and the underlying encoder, decoder, and predictor.

    :param train_data: list [train_frames, train_actions_onehot] or
     data generator
    :param val_data: list [val_frames, val_actions_onehot] or data generator
    :param latent_dim_list: list of the # of latent dimensions to loop over
    :param lr_list: list of the initial learning rates to loop over
    :param vae_l2_reg_list: list of l2 regularization penalties to loop over
    :param log_dir: Directory to save tensorboard logs.
    :param save_dir: Directory to save final model
    :param num_actions: Number of allowable actions.
    :param version_name: used only for naming the save file.
    :param epochs: Number of epochs to run for.
    :param batch_size: Batch size to use. Not used if training data is provided
    as a data generator.
    :param predictor_wt: Weight to place on the prediction loss (versus the
    reconstruction loss). See VAE_w_Prediction.
    :param lr_decay: Learning rate decay. This cannot be set at the same time
    as cosine_decay_steps.
    :param lr_decay_steps: Number of decay steps - see
    https://www.tensorflow.org/api_docs/python/tf/keras/
    optimizers/schedules/ExponentialDecay
    :param lr_decay_staircase: See https://www.tensorflow.org/api_docs/python/
    tf/keras/optimizers/schedules/ExponentialDecay
    :param predict_hidden_layers: Number of hidden layers in the predictor.
    :param cosine_decay_steps: See https://www.tensorflow.org/api_docs/
    python/tf/keras/experimental/CosineDecay
    :param extra_conv_layers: Number of extra convolutional layers in the
    encoder and decoder.
    :return: None. Runs models and saves them down where specified.
    """
    if isinstance(train_data, keras.utils.Sequence) and batch_size is not None:
        print("If training using a data generator, the batch size should be"
              "specified there. The batch size inputted here will not do "
              "anything.")
    encoder_path_dict = {}
    decoder_path_dict = {}
    predictor_path_dict = {}

    for lr in lr_list:
        for l2 in vae_l2_reg_list:
            for latent_dim in latent_dim_list:
                name = '{}fts_lr=1e{:.2}, l2={}'. \
                    format(latent_dim, math.log10(lr),
                           '1e{:.2}'.format(math.log10(l2)) if l2 > 0 else
                           'None')

                LOGDIR = os.path.join(log_dir, name)
                print("Starting {} - Tensorboard logs located at {}".format(
                    name, LOGDIR
                ))

                encoder = create_encoder(
                    latent_dim, l2=l2, extra_conv_layers=extra_conv_layers
                )
                decoder = create_decoder(
                    latent_dim, l2=l2, extra_conv_layers=extra_conv_layers
                )
                if num_predictors == 1:
                    predictor = create_predictor(
                        latent_dim, l2=l2,
                        num_categories=num_actions,
                        hidden_layers=predict_hidden_layers
                    )

                    vae = VAE_w_Prediction(encoder, decoder, predictor,
                                           predictor_wt=predictor_wt,
                                           kl_div_wt=kl_div_wt,
                                           vae_wt=vae_wt)
                elif num_predictors == 0:
                    vae = VAE(encoder, decoder, kl_div_wt=kl_div_wt)
                else:
                    raise NotImplementedError

                if lr_decay is not None:
                    assert lr_decay_steps is not None
                    assert cosine_decay_steps is None
                    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                        lr, lr_decay_steps, lr_decay,
                        staircase=lr_decay_staircase
                    )
                    opt = keras.optimizers.Adam(learning_rate=lr_schedule)
                elif cosine_decay_steps is not None:
                    lr_schedule = \
                        keras.experimental.CosineDecay(lr,
                                                       cosine_decay_steps)
                    opt = keras.optimizers.Adam(learning_rate=lr_schedule)
                else:
                    opt = keras.optimizers.Adam(learning_rate=lr)

                vae.compile(optimizer=opt,
                            metrics=['categorical_accuracy'])

                tensorboard_callback = keras.callbacks.TensorBoard(
                    log_dir=LOGDIR)
                checkpoint_callback = keras.callbacks.ModelCheckpoint(
                    os.path.join(LOGDIR,
                                 r'checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5')
                )
                csv_callback = keras.callbacks.CSVLogger(
                    os.path.join(LOGDIR, 'training_log.csv'))

                train_params = {
                    'epochs': epochs,
                    # 'batch_size': batch_size,
                    'callbacks': [tensorboard_callback,
                                  checkpoint_callback,
                                  csv_callback]
                }

                if isinstance(train_data, keras.utils.Sequence):
                    assert isinstance(val_data, keras.utils.Sequence)
                    history = vae.fit(
                        train_data,
                        validation_data=val_data,
                        **train_params
                    )
                else:
                    history = vae.fit(
                        x=np.array(train_data[0]),
                        y=np.array(train_data[1]),
                        validation_data=(np.array(val_data[0]),
                                         np.array(val_data[1])),
                        batch_size=batch_size,
                        **train_params
                    )
                encoder.save(os.path.join(save_dir, 'enc_' + name))
                decoder.save(os.path.join(save_dir, 'dec_' + name))

                encoder_path_dict[name] = os.path.join(save_dir,
                                                       'enc_' + name)
                decoder_path_dict[name] = os.path.join(save_dir,
                                                       'dec_' + name)
                if num_predictors == 1:
                    predictor.save(os.path.join(save_dir, 'pred_' + name))
                    predictor_path_dict[name] = os.path.join(save_dir,
                                                             'pred1_' + name)
    return encoder_path_dict, decoder_path_dict, predictor_path_dict
