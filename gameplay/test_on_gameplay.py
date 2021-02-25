import os
import sys

from tensorflow import keras
import test_rl_against_bc as rl_v_bc


ENVS = ['MsPacmanNoFrameskip-v4', 'BeamRiderNoFrameskip-v4',
        'SeaquestNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4',
        'BreakoutNoFrameskip-v4', 'EnduroNoFrameskip-v4',
        'QbertNoFrameskip-v4', 'PongNoFrameskip-v4']

RL_MODEL_PATHS = {
    env: os.path.join(
        'dqn_models',
        '{}_10000000.zip'.format(env.replace('NoFrameskip-v4', ''))
    ) for env in ENVS}

ENCODER_PREDICTOR_PATHS = {
    'CombinedVAE': {
        env: [
            os.path.join('trained_imitator_models', 'vae_w_random_init',
                         env.replace('NoFrameskip-v4', ''),
                         'enc_32fts_lr=1e-3.0, l2=1e-2.0'),
            os.path.join('trained_imitator_models', 'vae_w_random_init',
                         env.replace('NoFrameskip-v4', ''),
                         'pred_32fts_lr=1e-3.0, l2=1e-2.0')]
        for env in ENVS
    },
    'NatureCNN': {
        env: [None,
              os.path.join('trained_imitator_models',
                           'nature_cnn_random_init',
                           env.replace('NoFrameskip-v4', ''))]
        for env in ENVS
    },
    'SeparateVAE': {
        env: [
            os.path.join('trained_imitator_models', 'separate_vae_predictor',
                         'vae', env.replace('NoFrameskip-v4', ''),
                         'enc_32fts_lr=1e-4.0, l2=1e-6.0'),
            os.path.join('trained_imitator_models', 'separate_vae_predictor',
                         'predictor', env.replace('NoFrameskip-v4', ''))]
        for env in ENVS
    }
}


def run_rl_vs_bc_models(env, initial_random_steps,
                        base_save_folder='gameplay_results',
                        model_to_use='rl',
                        num_episodes=200):
    """
    Testing code comparing actions chosen by the RL agents versus various
    imitators.
    """
    if base_save_folder is not None:
        save_path = os.path.join(base_save_folder,
                                 '{}_follow_{}_randsteps={}'.format(
                                     env[:-14], model_to_use,
                                     initial_random_steps
                                 ))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        save_path = None

    rl_model_path = RL_MODEL_PATHS[env]
    encoder_paths = {k: v[env][0] for k, v in ENCODER_PREDICTOR_PATHS.items()}
    encoders = {
        k: None if v is None else keras.models.load_model(v) for k, v in
        encoder_paths.items()
    }
    predictor_paths = {k: v[env][1] for k, v in ENCODER_PREDICTOR_PATHS.items()}
    predictors = {k: keras.models.load_model(v) for k, v in
                  predictor_paths.items()}

    results = rl_v_bc.run_and_save_traj_data(
        rl_model_path=rl_model_path,
        encoder_dict=encoders,
        predictor_dict=predictors,
        seed_list=range(10 ** 6, 10 ** 6 + num_episodes),
        save_folder=save_path,
        model_to_use=model_to_use,
        sticky_action_prob=0.25,
        env_name=env,
        initial_random_steps=initial_random_steps,
        video_path=None
    )
    return results


def run_interpolatory_results(env):
    base_folder = os.path.join('gameplay_results', 'interpolatory')
    initial_random_steps = 0
    num_episodes = 200
    print('Starting Interpolatory {}, rand_steps={}'.format(
        env, initial_random_steps)
    )
    run_rl_vs_bc_models(env, initial_random_steps=initial_random_steps,
                        base_save_folder=base_folder,
                        num_episodes=num_episodes)

    initial_random_steps = 75
    num_episodes = 50
    print('Starting Interpolatory {}, rand_steps={}'.format(
        env, initial_random_steps)
    )
    run_rl_vs_bc_models(env, initial_random_steps=initial_random_steps,
                        base_save_folder=base_folder,
                        num_episodes=num_episodes)

    initial_random_steps = 100
    num_episodes = 50
    print('Starting Interpolatory {}, rand_steps={}'.format(
        env, initial_random_steps)
    )
    run_rl_vs_bc_models(env, initial_random_steps=initial_random_steps,
                        base_save_folder=base_folder,
                        num_episodes=num_episodes)


def run_extrapolatory_results(env):
    base_folder = os.path.join('gameplay_results', 'extrapolatory')
    initial_random_steps = 150
    num_episodes = 300
    print('Starting Interpolatory {}, rand_steps={}'.format(
        env, initial_random_steps)
    )
    run_rl_vs_bc_models(env, initial_random_steps=initial_random_steps,
                        base_save_folder=base_folder,
                        num_episodes=num_episodes)


if __name__ == '__main__':
    env = sys.argv[1]
    run_interpolatory_results(env)
    run_extrapolatory_results(env)
