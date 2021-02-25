import os

import cv2
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines import DQN
import numpy as np
import tensorflow as tf

tf.enable_eager_execution()


def compare_expert_to_bc(rl_model_path, encoder_dict, predictor_dict, seed=0,
                         model_to_use='rl', render=False, video_path=None,
                         sticky_action_prob=0,
                         env_name='MsPacmanNoFrameskip-v4',
                         initial_random_steps=0):
    env, rl_model = setup_rl_model(rl_model_path, seed, env_name=env_name)

    if video_path is not None:
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        video_path = os.path.join(video_path, 'seed={}.mp4'.format(seed))
    return run_rl_and_bc_model(env, rl_model, encoder_dict, predictor_dict,
                               model_to_use=model_to_use, render=render,
                               video_path=video_path,
                               sticky_action_prob=sticky_action_prob,
                               initial_random_steps=initial_random_steps)


def setup_rl_model(model_path, seed, env_name='MsPacmanNoFrameskip-v4'):
    env = make_atari_env(env_name, num_env=1, seed=seed)
    env = VecFrameStack(env, n_stack=4)
    model = DQN.load(model_path)
    model.set_env(env)
    return env, model


def run_rl_and_bc_model(env, rl_model, encoder_dict, predictor_dict,
                        model_to_use='rl', render=False, video_path=None,
                        sticky_action_prob=0,
                        initial_random_steps=0):
    bc_models = encoder_dict.keys()

    obs = env.reset()
    done = False
    rwd_sum = 0
    num_steps = 0

    state = None
    mask = [True]
    idx = 0
    rl_actions = []
    bc_actions = {name: [] for name in bc_models}
    rwd_hist = []
    taken_actions = []
    repeated = []

    if render:
        frame_data = env.render()
    if video_path is not None:
        layers = 1
        height, width = obs[0, ..., -1].shape
        video = cv2.VideoWriter(video_path, 0, 4, (width, height), False)

    prev_action = None
    for _ in range(initial_random_steps):
        obs, reward, _, _ = env.step([env.action_space.sample()])

    while not done:
        # RL Agent action
        rl_action, state = rl_model.predict(obs, state=state, mask=mask)
        rl_actions.append(rl_action)

        # Behavior cloned action
        obs_decimal = (obs / 255.).astype(np.float32)
        for encoder_name, encoder in encoder_dict.items():
            if encoder is not None:
                encoded_features = encoder(obs_decimal)[2]
            else:
                encoded_features = obs_decimal
            bc_action = np.argmax(
                predictor_dict[encoder_name](encoded_features).numpy(), axis=-1
            )
            bc_actions[encoder_name].append(bc_action)

        if prev_action is None or np.random.uniform() > sticky_action_prob:
            if model_to_use == 'rl':
                taken_action = rl_action
            else:
                taken_action = bc_actions[model_to_use][-1]
            repeated.append(False)
        else:
            taken_action = prev_action
            repeated.append(True)
        obs, reward, done, _ = env.step(taken_action)
        prev_action = taken_action

        rwd_hist.append(reward)
        taken_actions.append(taken_action)
        num_steps += 1
        mask = [done[0]]
        idx += 1

        if render:
            env.render()
        if video_path is not None:
            t = obs[0, ..., -1][..., np.newaxis]
            video.write(obs[0, ..., -1][..., np.newaxis])

    env.close()
    if video_path is not None:
        cv2.destroyAllWindows()
        video.release()

    if num_steps > 0:
        rwd_hist = np.concatenate(rwd_hist)
        bc_actions = {k: np.concatenate(v) for k, v in bc_actions.items()}
        rl_actions = np.concatenate(rl_actions)
    return rwd_sum, num_steps, rl_actions, bc_actions, rwd_hist, np.array(
        repeated)


def run_and_save_traj_data(rl_model_path, encoder_dict, predictor_dict,
                           seed_list, save_folder, model_to_use='rl',
                           sticky_action_prob=0, video_path=None,
                           env_name='MsPacmanNoFrameskip-v4',
                           initial_random_steps=0):
    summary_stats = {k: {
        'Percent Accuracy': [],
        'Number of Steps': [],
        'Reward': []
    } for k in encoder_dict.keys()}
    summary_stats_save_path = 'summary_stats.npz'

    for i in range(len(seed_list)):
        seed = seed_list[i]
        rwd_sum, num_steps, rl_actions, bc_actions, rwd_hist, repeated = \
            compare_expert_to_bc(rl_model_path, encoder_dict, predictor_dict,
                                 seed, model_to_use=model_to_use,
                                 video_path=video_path,
                                 sticky_action_prob=sticky_action_prob,
                                 env_name=env_name,
                                 initial_random_steps=initial_random_steps)
        if num_steps == 0:
            continue

        # Trajectory level statistics
        results = {'RL Actions': rl_actions,
                   'BC Actions': bc_actions,
                   'Number of Steps': num_steps,
                   'Rewards': rwd_hist}
        file_name = 'results_following_{}_seed={}.npz'.format(
            model_to_use, seed
        )
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        np.savez(os.path.join(save_folder, file_name), **results)

        # Summary statistics for entire trajectory
        for k, v in bc_actions.items():
            summary_stats[k]['Percent Accuracy'].append(
                (rl_actions == v).sum() / len(rl_actions)
            )
            summary_stats[k]['Number of Steps'].append(
                len(rl_actions)
            )
            summary_stats[k]['Reward'].append(rwd_sum)
    np.savez(os.path.join(save_folder, summary_stats_save_path),
             **summary_stats)
    return summary_stats
