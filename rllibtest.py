import os
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from datetime import datetime

start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# Configuration for the PPOTrainer
config = {
    "env": "LunarLander-v2",
    "num_workers": 1,
    "rollout_fragment_length": 200,
    "train_batch_size": 2000,
    "batch_mode": "complete_episodes",
    "num_sgd_iter": 30,
    "sgd_minibatch_size": 128,
    "model": {
        "fcnet_hiddens": [100, 100],
    },
}

# TensorBoard output directory
tensorboard_output = "tensorboard_output"

# Initialize Ray
ray.init()

# Training with TensorBoard logging
analysis = tune.run(
    PPOTrainer,
    config=config,
    stop={"training_iteration": 10},
    local_dir=tensorboard_output,
    checkpoint_at_end=True,
    checkpoint_freq=1,
)

# Print the best performing checkpoint
print(
    "Best performing checkpoint:",
    analysis.get_best_checkpoint(metric="episode_reward_mean", mode="max"),
)

# Cleanup
ray.shutdown()
