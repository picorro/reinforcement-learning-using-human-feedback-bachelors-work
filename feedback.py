import gym
from gym.utils.play import play
import numpy as np
import os
import h5py
import pickle
from d3rlpy.dataset import MDPDataset

class Feedback:
    def playEnv(self, env, recording_name='', record=False):
        key_to_action = {}

        if env.spec.id == "CartPole-v0" or env.spec.id == "CartPole-v1":
            key_to_action = {
                (ord('a'),): 0,  # LEFT
                (ord('d'),): 1,  # RIGHT
            }
        elif env.spec.id == "LunarLander-v2":
            key_to_action = {
                (ord('s'),): 0,  # S: Do nothing
                (ord('a'),): 1,  # A: Fire left engine
                (ord('w'),): 2,  # W: Fire main engine
                (ord('d'),): 3,  # D: Fire right engine
            }

        # Determine environment's properties
        is_discrete_action = isinstance(env.action_space, gym.spaces.Discrete)

        # Initialize lists to store data
        states = []
        actions = []
        rewards = []
        next_states = []
        terminals = []
        episode_terminals = []

        def save_data_callback(obs_t, obs_tp1, action, rew, done, info):
            states.append(obs_t)
            actions.append(action)
            rewards.append(rew)
            next_states.append(obs_tp1)
            terminals.append(done)

        play(env, keys_to_action=key_to_action, callback=save_data_callback)

        if record:
            pathExists = os.path.exists(f'./datasets/play/{env.spec.id}')
            if not pathExists:
                os.makedirs(f'datasets/play/{env.spec.id}')

            data = {
                'observations': np.array(states),
                'actions': np.array(actions),
                'rewards': np.array(rewards),
                'next_states': np.array(next_states),
                'terminals': np.array(terminals),
                'discrete_action': isinstance(env.action_space, gym.spaces.Discrete)
            }

            with open(f"./datasets/play/{env.spec.id}/{recording_name}.pkl", "wb") as f:
                pickle.dump(data, f)
    
    def load_dataset_from_pickle(self, file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        dataset = MDPDataset(
            observations=data['observations'],
            actions=data['actions'],
            rewards=data['rewards'],
            terminals=data['terminals'],
            discrete_action=data['discrete_action']
        )

        return dataset
