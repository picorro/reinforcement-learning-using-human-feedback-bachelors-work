import d3rlpy

from d3rlpy.datasets import get_cartpole # CartPole-v0 dataset
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from sklearn.model_selection import train_test_split
from datetime import datetime
import argparse
import os
import sys

startTime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

parser = argparse.ArgumentParser(description="Parsing module.")
parser.add_argument("-a", "--algo", type=str, required=True, help="Algorithm: --dqn")
parser.add_argument("-l", "--load", type=str, help="Model file location")
args = parser.parse_args()


dataset, env = get_cartpole()


algorithmName = args.algo
algo = None

if args.load != None:
    if algorithmName == 'dqn':
        algo = d3rlpy.algos.DQN()
        algo.build_with_dataset(dataset)
        algo.load_model(f'./trained_models/{algorithmName}/{args.load}.pt')
        
        evaluate_scorer = evaluate_on_environment(env, render=True)
        rewards = evaluate_scorer(algo)
        sys.exit()

    

train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

algo = d3rlpy.algos.DQN()
algo.build_with_dataset(dataset)
td_error = td_error_scorer(algo, test_episodes)

# train offline
algo.fit(dataset, n_steps=1000, n_steps_per_epoch=1000)

evaluate_scorer = evaluate_on_environment(env, render=True)
rewards = evaluate_scorer(algo)

# Save trained model
pathExists = os.path.exists(f'./trained_models/{algorithmName}')
if not pathExists:
    os.makedirs(f'trained_models/{algorithmName}')

algo.save_model(f'./trained_models/{algorithmName}/{startTime}.pt')

