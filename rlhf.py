# import sys
# import gymnasium as gym
# sys.modules["gym"] = gym

# from stable_baselines3 import A2C

# def main():
#    env = gym.make("LunarLander-v2")
#    model = A2C("MlpPolicy", env)
#    model.learn(total_timesteps=int(1000))

#    model.save("a2c_lunar")

#    del model

#    model = A2C.load("a2c_lunar")

#    render_env = gym.make("LunarLander-v2", render_mode="human")
#    observation, info = render_env.reset()
#    for _ in range(100):
#       action, _ = model.predict(observation)
#       observation, reward, terminated, truncated, info = render_env.step(action)

#       if terminated or truncated:
#          observation, info = render_env.reset()
#    env.close()

# if __name__ == "__main__":
#     main()

import sys
import gymnasium as gym
sys.modules["gym"] = gym

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class HumanFeedbackEnv(gym.Wrapper):
    def __init__(self, env, human_reward_fn):
        super().__init__(env)
        self.human_reward_fn = human_reward_fn

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = self.human_reward_fn(obs)
        return obs, reward, done, info

class Policy(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        return self.layers(x)

class RLHF:
    def __init__(self, env, human_reward_fn, input_dim, hidden_dim, output_dim):
        self.env = HumanFeedbackEnv(env, human_reward_fn)
        self.policy = Policy(input_dim, hidden_dim, output_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.001)

    def train(self, num_epochs, batch_size, num_samples):
        for epoch in range(num_epochs):
            batch_obs = []
            batch_actions = []
            batch_weights = []

            for _ in range(batch_size):
                obs = self.env.reset()
                done = False
                episode_reward = 0
                weights = []
                actions = []

                while not done:
                    action_probs = torch.softmax(self.policy(obs), dim=-1)
                    action = np.random.choice(self.env.action_space.n, p=action_probs.detach().numpy())
                    actions.append(action)

                    obs, reward, done, info = self.env.step(action)
                    episode_reward += reward

                    human_reward = self.env.human_reward_fn(obs)
                    weight = np.exp(human_reward) / (np.exp(human_reward) + np.exp(-human_reward))
                    weights.append(weight)

                batch_obs.append(np.array(obs))
                batch_actions.append(np.array(actions))
                batch_weights.append(np.array(weights))

            batch_obs = np.array(batch_obs)
            batch_actions = np.array(batch_actions)
            batch_weights = np.array(batch_weights)

            for i in range(num_samples):
                indices = np.random.choice(batch_size, size=batch_size, replace=True)
                obs_batch = batch_obs[indices]
                action_batch = batch_actions[indices]
                weight_batch = batch_weights[indices]

                action_probs = torch.softmax(self.policy(obs_batch), dim=-1)
                log_probs = torch.log(action_probs.gather(-1, torch.tensor(action_batch[:, np.newaxis]))).squeeze(-1)
                loss = -(log_probs * torch.tensor(weight_batch, dtype=torch.float32)).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"Epoch {epoch}: loss = {loss.item()}, mean reward = {episode_reward}")

    def act(self, obs):
        action_probs = torch.softmax(self.policy(obs), dim=-1)
        action = np.random.choice(self.env.action_space.n, p=action_probs.detach().numpy())
        return action