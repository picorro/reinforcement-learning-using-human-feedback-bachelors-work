import d3rlpy
from d3rlpy.datasets import get_cartpole # CartPole-v0 dataset
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from d3rlpy.dataset import MDPDataset
from sklearn.model_selection import train_test_split
from datetime import datetime
import argparse
import os
import sys
import gym
from feedback import Feedback
import torch

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# print(torch.cuda.is_available())
# print(torch.cuda.device_count())


start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

parser = argparse.ArgumentParser(description="Parsing module.")
parser.add_argument("-a", "--algorithm_name", type=str, required=True, help="Algorithm: --dqn")
parser.add_argument("-e", "--environment", type=str, required=True, help="Environment: --CartPole-v0")
parser.add_argument("-m", "--mode", type=str, help="Program mode: --demo --dataset")
parser.add_argument("-s", "--steps", type=int, help="Step count for any operations the program might do related to model training or dataset generation")
parser.add_argument("-tm", "--trained_model", type=str, help="Trained model file name yyyy-mm-dd-HH-MM-SS")
parser.add_argument("-d", "--dataset", type=str, help="Dataset file name yyyy-mm-dd-HH-MM-SS")
parser.add_argument("-g", "--gpu", type=str2bool, help="Use GPU? True/False", default=False)


args = parser.parse_args()

algorithm_name = args.algorithm_name
algorithm = None

dataset = env = None


print(args.gpu)
if args.dataset != None:
    dataset = Feedback.load_dataset_from_pickle(f"./datasets/{args.dataset}")
    #dataset = MDPDataset.load(f"./datasets/{args.dataset}")

if args.environment == "CartPole-v0":
    dataset, env = get_cartpole()
elif args.environment == "LunarLander-v2":
    env = gym.make("LunarLander-v2")
elif args.environment == "CarRacing-v1" or args.environment == "CarRacing-v2":
    env = gym.make("CarRacing-v2")
else:
    print("Could not find environment! Exiting...")
    sys.exit()
    
if algorithm_name == "dqn":
        algorithm = d3rlpy.algos.DQN(use_gpu=args.gpu)
elif algorithm_name == "cql":
    algorithm = d3rlpy.algos.DiscreteCQL(use_gpu=args.gpu)
elif algorithm_name == "sac":
    algorithm = d3rlpy.algos.DiscreteSAC(use_gpu=args.gpu)
else:
    algorithm = d3rlpy.algos.DQN(use_gpu=args.gpu)

if args.mode == "demo":
    algorithm.build_with_dataset(dataset)
    algorithm.load_model(f"./trained_models/{algorithm_name}/{args.load}.pt")
    
    evaluate_scorer = evaluate_on_environment(env, render=True)
    rewards = evaluate_scorer(algorithm)
    sys.exit()
elif args.mode == "dataset":
    algorithm.build_with_env(env)
    algorithm.load_model(f"./trained_models/{algorithm_name}/{args.trained_model}.pt")
    buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=args.steps, env=env)
    algorithm.collect(env, buffer, n_steps=args.steps)
    dataset = buffer.to_mdp_dataset()

    pathExists = os.path.exists(f'./datasets/{algorithm_name}')
    if not pathExists:
        os.makedirs(f'datasets/{algorithm_name}')
    dataset.dump(f"./datasets/{algorithm_name}/{start_time}.h5")
    sys.exit()
elif args.mode == "play":
    Feedback.playEnv(env, recording_name=start_time, record=True)
    sys.exit()


#train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

#algorithm.build_with_dataset(dataset)
#algorithm.build_with_env(env)
#td_error = td_error_scorer(algorithm, test_episodes)

# train offline
#algorithm.fit(dataset, n_steps=args.steps, n_steps_per_epoch=1000)

# train online

# experience replay buffer
buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=args.steps, env=env)
explorer = d3rlpy.online.explorers.ConstantEpsilonGreedy(0.3)

tensorboard_log_dir = f"tensorboard_logs/{algorithm_name}/{start_time}"
if not os.path.exists(tensorboard_log_dir):
    os.makedirs(tensorboard_log_dir)

algorithm.fit_online(
    env,
    buffer,
    explorer,
    n_steps=args.steps, 
    n_steps_per_epoch=1000,
    update_start_step=1000,
    tensorboard_dir=tensorboard_log_dir
)

evaluate_scorer = evaluate_on_environment(env, render=True)
rewards = evaluate_scorer(algorithm)

# Save trained model
pathExists = os.path.exists(f'./trained_models/{algorithm_name}')
if not pathExists:
    os.makedirs(f'trained_models/{algorithm_name}')

algorithm.save_model(f'./trained_models/{algorithm_name}/{start_time}.pt')

