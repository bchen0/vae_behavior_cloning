import math
import os
import sys

from tensorflow import keras

import architectures.vae_with_prediction as vae_code
import architectures.training as training
import data_loaders as loaders


def run_experiments(env_name, save_folder):
    num_actions = {
        'MsPacmanNoFrameskip-v4': 9,
        'SeaquestNoFrameskip-v4': 18,
        'SpaceInvadersNoFrameskip-v4': 6,
        'PongNoFrameskip-v4': 6,
        'QbertNoFrameskip-v4': 6,
        'EnduroNoFrameskip-v4': 9,
        'BreakoutNoFrameskip-v4': 4,
        'BeamRiderNoFrameskip-v4': 9
    }[env_name]

    data_path = os.path.join('gameplay', 'trajectories', 'rand_steps=0',
                             env_name)
    data_path_75 = os.path.join('gameplay', 'trajectories', 'rand_steps=75',
                                env_name)
    data_path_100 = os.path.join('gameplay', 'trajectories', 'rand_steps=75',
                                env_name)

    train_loader = loaders.create_data_loader(
        [data_path, data_path_75, data_path_100],
        min_traj=[0, 10000, 20000],
        max_traj=[159, 10039, 20039],
        num_actions=num_actions,
        batch_size=32
    )
    val_loader = loaders.create_data_loader(
        [data_path, data_path_75, data_path_100],
        min_traj=[160, 10040, 20040],
        max_traj=[199, 10049, 20049],
        num_actions=num_actions,
        batch_size=32
    )

    epochs = 20
    num_hidden_layers = 1
    extra_conv_layers = 1

    # Train Combined VAE
    vae_code.train_multiple_vaes(
        train_loader,
        val_loader,
        latent_dim_list=[32],
        lr_list=[1e-3],
        vae_l2_reg_list=[1e-2],
        num_actions=num_actions,
        log_dir=os.path.join(
            save_folder, 'combinedvae', 'logs', env_name[:-14]
        ),
        save_dir=os.path.join(save_folder, 'combinedvae', env_name[:-14]),
        epochs=epochs,
        predict_hidden_layers=num_hidden_layers,
        cosine_decay_steps=train_loader.__len__()*epochs,
        extra_conv_layers=extra_conv_layers
    )

    # Train Nature CNN
    lr = 1e-3
    l2 = 1e-4
    training.train_nature_cnn(
        train_loader, val_loader, epoch_size=train_loader.__len__(),
        num_actions=num_actions, lr=lr, l2=l2, epochs=20, batch_size=32,
        log_dir=os.path.join(
            save_folder,
            'cnn',
            env_name[:-14],
            'lr=1e{:.2}_l2=1e{:.2}'.format(
                env_name[:-14], math.log10(lr), math.log10(l2)
            )),
        decay_type='cos'
    )

    # Train Separate VAE/Predictor - VAE portion:
    vae_standalone_lr = 1e-4
    vae_standalone_l2 = 1e-6
    encoder_paths, decoder_paths, predictor_paths = vae_code.train_multiple_vaes(
        train_loader,
        val_loader,
        latent_dim_list=[32],
        lr_list=[vae_standalone_lr],
        vae_l2_reg_list=[vae_standalone_l2],
        num_actions=num_actions,
        log_dir=os.path.join(save_folder, 'separate_vae_predictor', 'predictor',
                             'logs', env_name[:-14]),
        save_dir=os.path.join(save_folder, 'separate_vae', 'predictor',
                              env_name[:-14]),
        epochs=epochs,
        predict_hidden_layers=num_hidden_layers,
        cosine_decay_steps=train_loader.__len__()*epochs,
        extra_conv_layers=extra_conv_layers,
        num_predictors=0
    )

    # New loaders that preprocess frames with the VAE that was just trained:
    predictor_lr = 1e-2
    predictor_l2 = 1e-6

    assert len(encoder_paths.values()) == 1
    encoder_standalone = keras.models.load_model(
        list(encoder_paths.values())[0]
    )

    def encoding_fcn(x):
        return encoder_standalone(x)[2]

    train_loader = loaders.create_data_loader(
        [data_path, data_path_75, data_path_100],
        min_traj=[0, 10000, 20000],
        max_traj=[159, 10039, 20039],
        num_actions=num_actions,
        batch_size=32,
        preprocess_fcn=encoding_fcn
    )
    val_loader = loaders.create_data_loader(
        [data_path, data_path_75, data_path_100],
        min_traj=[160, 10040, 20040],
        max_traj=[199, 10049, 20049],
        num_actions=num_actions,
        batch_size=32,
        preprocess_fcn=encoding_fcn
    )

    training.train_only_predictor(
        train_loader, val_loader, dim=32, epoch_size=train_loader.__len__(),
        num_actions=num_actions, lr=predictor_lr, l2=predictor_l2,
        hidden_layers=num_hidden_layers, epochs=epochs, batch_size=32,
        decay_type='cos',
        log_dir=os.path.join(
            save_folder,
            'separate_vae_predictor',
            'predictor',
            env_name[:-14],
            'lr=1e{:.2}_l2=1e{:.2}'.format(
                math.log10(predictor_lr), math.log10(predictor_l2)
            )
        )
    )


def main():
    save_folder = sys.argv[1]
    env = sys.argv[2]
    run_experiments(env, save_folder=save_folder)


if __name__ == '__main__':
    main()
