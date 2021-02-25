"""
Taken from stable baselines but slightly altered to only save down the last
frame. Previously, a stack of four frames would be saved down as an image in
four channels.
"""

import os
import warnings
from typing import Dict

import cv2  # pytype:disable=import-error
import numpy as np
from gym import spaces

from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.vec_env import VecEnv, VecFrameStack
from stable_baselines.common.base_class import _UnvecWrapper


def generate_expert_traj(model, save_path=None, env=None, n_timesteps=0,
                         n_episodes=100, image_folder='recorded_images',
                         sticky_action_prob=0, seed=None,
                         random_initial_steps=0):
    """
    Train expert controller (if needed) and record expert trajectories.

    .. note::

        only Box and Discrete spaces are supported for now.

    :param model: (RL model or callable) The expert model, if it needs to be
        trained, then you need to pass ``n_timesteps > 0``.
    :param save_path: (str) Path without the extension where the expert dataset
        will be saved (ex: 'expert_cartpole' -> creates 'expert_cartpole.npz').
        If not specified, it will not save, and just return the generated expert
        trajectories. This parameter must be specified for image-based
        environments.
    :param env: (gym.Env) The environment, if not defined then it tries to use
        the model environment.
    :param n_timesteps: (int) Number of training timesteps
    :param n_episodes: (int) Number of trajectories (episodes) to record
    :param image_folder: (str) When using images, folder that will be used to
        record images.
    :param sticky_action_prob: (float) Probability of repeating previous action.
    :param seed: (int) seed to initialize environment with
    :param random_initial_steps: (int) number of random initial steps to start
        each trajectory with.
    :return: (dict) the generated expert trajectories.
    """

    # Retrieve the environment using the RL model
    if env is None and isinstance(model, BaseRLModel):
        env = model.get_env()

    assert env is not None, "You must set the env in the model or pass it to " \
                            "the function."

    is_vec_env = False
    if isinstance(env, VecEnv) and not isinstance(env, _UnvecWrapper):
        is_vec_env = True
        if env.num_envs > 1:
            warnings.warn("You are using multiple envs, only the data from the"
                          " first one will be recorded.")

    # Sanity check
    assert (isinstance(env.observation_space, spaces.Box) or
            isinstance(env.observation_space, spaces.Discrete)), \
        "Observation space type not supported"

    assert (isinstance(env.action_space, spaces.Box) or
            isinstance(env.action_space, spaces.Discrete)), \
        "Action space type not supported"

    # Check if we need to record images
    obs_space = env.observation_space
    record_images = len(obs_space.shape) == 3 \
        and obs_space.shape[-1] in [1, 3, 4] \
        and obs_space.dtype == np.uint8
    if record_images and save_path is None:
        warnings.warn("Observations are images but no save path was "
                      "specified, so will save in numpy archive; "
                      "this can lead to higher memory usage.")
        record_images = False

    if not record_images and len(obs_space.shape) == 3 \
            and obs_space.dtype == np.uint8:
        warnings.warn("The observations looks like images (shape = {}) "
                      "but the number of channel > 4, so it will be saved "
                      "in the numpy archive which can lead to high memory "
                      "usage".format(obs_space.shape))

    # image_ext = 'jpg'
    image_ext = 'png' # Lossless
    if record_images:
        # We save images as jpg or png, that have only 3/4 color channels
        if isinstance(env, VecFrameStack) and env.n_stack == 4:
            image_ext = 'png'

        folder_path = os.path.dirname(save_path)
        image_folder = os.path.join(folder_path, image_folder)
        os.makedirs(image_folder, exist_ok=True)
        print("=" * 10)
        print("Images will be recorded to {}/".format(image_folder))
        print("Image shape: {}".format(obs_space.shape))
        print("=" * 10)

    if n_timesteps > 0 and isinstance(model, BaseRLModel):
        model.learn(n_timesteps)

    taken_actions = []
    observations = []
    rewards = []
    episode_returns = np.zeros((n_episodes,))
    episode_starts = []
    model_selected_actions = []
    repeated = []

    ep_idx = 0
    obs = env.reset()
    episode_starts.append(True)
    reward_sum = 0.0
    idx = 0
    # state and mask for recurrent policies
    state, mask = None, None

    if is_vec_env:
        mask = [True for _ in range(env.num_envs)]

    prev_action = None
    np_rng = np.random.RandomState(seed)

    # If random_initial_steps > 0, take random_initial_steps random actions
    # at the start of each game.
    random_steps = random_initial_steps > 0

    while ep_idx < n_episodes:
        if random_steps:
            for _ in range(random_initial_steps):
                obs, _, _, _ = env.step(env.action_space.sample())
            # Set random steps = False until the next episode
            random_steps = False

        if record_images:
            image_path = os.path.join(image_folder, "{}.{}".format(idx,
                                                                   image_ext))
            obs_ = obs[0] if is_vec_env else obs
            # Convert from RGB to BGR
            # which is the format OpenCV expect
            # if obs_.shape[-1] == 3:
            #     obs_ = cv2.cvtColor(obs_, cv2.COLOR_RGB2BGR)

            # Only grab the last frame from the framestack
            obs_ = obs_[..., -1][..., np.newaxis]
            cv2.imwrite(image_path, obs_)
            observations.append(image_path)
        else:
            observations.append(obs)

        # Choose an action
        if isinstance(model, BaseRLModel):
            model_selected_action, state = model.predict(obs, state=state,
                                                         mask=mask)
        else:
            model_selected_action = model(obs)

        # With probability sticky_action_prob, the agent repeats the last
        # action, ignoring the chosen action above. Note that this
        # is implemented in the gym environment, but I do not know how that
        # interacts with all the stable-baselines code on top of it, so I am
        # doing it here. This also has the advantage in allowing us to
        # distinguish between repeated actions and normal actions.
        if prev_action is None or np_rng.uniform() > sticky_action_prob:
            taken_action = model_selected_action
            repeated.append(False)
        else:
            taken_action = prev_action
            repeated.append(True)

        obs, reward, done, _ = env.step(taken_action)
        prev_action = taken_action

        # Use only first env
        if is_vec_env:
            mask = [done[0] for _ in range(env.num_envs)]
            model_selected_action = np.array([model_selected_action[0]])
            taken_action = np.array([taken_action[0]])
            reward = np.array([reward[0]])
            done = np.array([done[0]])

        taken_actions.append(taken_action)
        model_selected_actions.append(model_selected_action)
        rewards.append(reward)
        episode_starts.append(done)
        reward_sum += reward
        idx += 1
        if done:
            if not is_vec_env:
                obs = env.reset()
                # Reset the state in case of a recurrent policy
                state = None

            episode_returns[ep_idx] = reward_sum
            reward_sum = 0.0
            ep_idx += 1

            # Once done, set random_steps = True so that next sample starts off
            # with random actions.
            if random_initial_steps > 0:
                random_steps = True

    if isinstance(env.observation_space, spaces.Box) and not record_images:
        observations = np.concatenate(observations).\
            reshape((-1,) + env.observation_space.shape)
    elif isinstance(env.observation_space, spaces.Discrete):
        observations = np.array(observations).reshape((-1, 1))
    elif record_images:
        observations = np.array(observations)

    if isinstance(env.action_space, spaces.Box):
        model_selected_actions = np.concatenate(model_selected_actions).\
            reshape((-1,) + env.action_space.shape)
        taken_actions = np.concatenate(taken_actions).\
            reshape((-1,) + env.action_space.shape)
    elif isinstance(env.action_space, spaces.Discrete):
        model_selected_actions = np.array(model_selected_actions).\
            reshape((-1, 1))
        taken_actions = np.array(taken_actions).reshape((-1, 1))

    rewards = np.array(rewards)
    episode_starts = np.array(episode_starts[:-1])
    repeated = np.array(repeated)

    assert len(observations) == len(taken_actions)
    assert len(taken_actions) == len(model_selected_actions)

    numpy_dict = {
        'model selected actions': model_selected_actions,
        'taken actions': taken_actions,
        'obs': observations,
        'rewards': rewards,
        'episode_returns': episode_returns,
        'episode_starts': episode_starts,
        'repeated': repeated
    }  # type: Dict[str, np.ndarray]

    for key, val in numpy_dict.items():
        print(key, val.shape)

    if save_path is not None:
        np.savez(save_path, **numpy_dict)

    env.close()

    return numpy_dict
