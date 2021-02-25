import os
import sys

import numpy as np

import gym
from stable_baselines import DQN
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack
import expert_traj

DQN_MODEL_FOLDER = r'dqn_models'
SAVE_PATH = r'trajectories'


def get_dqn_model_path(env_name):
    return os.path.join(
        DQN_MODEL_FOLDER,
        '{}_10000000.zip'.format(env_name.replace('NoFrameskip-v4', ''))
    )


def generate_trajectories(
    env_name, model_path, save_path, n_episodes=1, atari=True, suffix='',
    seed=None, sticky_action_prob=0., random_initial_steps=0
):
    # Make environment
    if atari:
        # Need to provide seed to make_atari_env.
        if seed is None:
            seed = np.random.randint(0, 1e9)
        env = make_atari_env(env_name, num_env=1, seed=seed)
        env = VecFrameStack(env, n_stack=4)
    else:
        env = gym.make(env_name)
    model = DQN.load(model_path)
    model.set_env(env)

    path = os.path.join(save_path, env_name)
    if not os.path.isdir(path):
        os.makedirs(path)
    image_folder = '{}-recorded_images-{}'.format(env_name, suffix)
    save_path = os.path.join(path, '{}_{}'.format(env_name, suffix))

    print("Using modified generate_expert_traj function to save down"
          "just the last frame.")

    seed = seed + 1000  # In case there are weird interactions between the
    # seed used in the environment and the seed used to determine sticky
    # actions. Seems very unlikely, but can't hurt to do this.
    expert_traj.generate_expert_traj(
        model, save_path=save_path, n_episodes=n_episodes,
        image_folder=image_folder, sticky_action_prob=sticky_action_prob,
        seed=seed, random_initial_steps=random_initial_steps
    )


def generate_trajectories_multiple(n_episodes_per_folder, num_folders,
                                   env_name, model_path, save_path,
                                   atari=True, seed_list=None,
                                   sticky_action_prob=0.,
                                   random_initial_steps=0):
    if seed_list is not None:
        assert len(seed_list) == num_folders
    for i in range(0, num_folders):
        print('Starting batch of trajectories: {}'.format(i))
        if seed_list is None:
            seed = None
        else:
            seed = seed_list[i]
        if n_episodes_per_folder == 1:
            suffix = '{}'.format(seed)
        else:
            suffix = '{}_to_{}'.format(i*n_episodes_per_folder,
                                       (i+1)*n_episodes_per_folder-1)
        generate_trajectories(
            env_name, model_path, save_path, n_episodes=n_episodes_per_folder,
            atari=atari, suffix=suffix, seed=seed,
            sticky_action_prob=sticky_action_prob,
            random_initial_steps=random_initial_steps
        )


if __name__ == '__main__':
    env = sys.argv[1]
    model_path = get_dqn_model_path(env)
    generate_trajectories_multiple(
        1, 200, env, model_path=model_path,
        save_path=os.path.join(SAVE_PATH, 'rand_steps=0'),
        atari=True, seed_list=range(200), sticky_action_prob=0.25
    )
    generate_trajectories_multiple(
        1, 50, env, model_path=model_path,
        save_path=os.path.join(SAVE_PATH, 'rand_steps=75'),
        atari=True, seed_list=range(10000, 10050), sticky_action_prob=0.25
    )
    generate_trajectories_multiple(
        1, 50, env, model_path=model_path,
        save_path=os.path.join(SAVE_PATH, 'rand_steps=100'),
        atari=True, seed_list=range(20000, 20050), sticky_action_prob=0.25
    )
