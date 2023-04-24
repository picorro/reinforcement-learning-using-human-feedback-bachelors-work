import d3rlpy
from d3rlpy.datasets import get_cartpole  # CartPole-v0 dataset
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from d3rlpy.preprocessing.scalers import *
from d3rlpy.dataset import MDPDataset
from sklearn.model_selection import train_test_split
from datetime import datetime
import argparse
import os
import sys
import gym
from feedback import Feedback
from intervention import get_intervention_step_array
import numpy as np
import pickle
import cv2
import warnings
from utils import combine_tensorboard_logs

# Ignore some warnings that stem from Tkinter library
warnings.filterwarnings("ignore", message="Exception in thread*")
warnings.filterwarnings("ignore", message=".*render method is deprecated*")


def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# Argument Parsing
parser = argparse.ArgumentParser(description="Parsing module.")
parser.add_argument(
    "-a", "--algorithm_name", type=str, default="no_algorithm", help="Algorithm: --dqn"
)
parser.add_argument(
    "-e",
    "--environment",
    type=str,
    required=True,
    help="Environment: --CartPole-v0 --LunarLander-v2",
)
parser.add_argument("-m", "--mode", type=str, help="Program mode: --demo --dataset")
parser.add_argument("-p", "--parameters", type=str, help="Parameter path")
parser.add_argument(
    "-s",
    "--steps",
    type=int,
    help="Step count for any operations the program might do related to model training or dataset generation",
)
parser.add_argument(
    "-i",
    "--interventions",
    type=int,
    help="How many human interventions should the algorithm execute?",
    default=10,
)
parser.add_argument(
    "-tm",
    "--trained_model",
    type=str,
    help="Trained model file name yyyy-mm-dd-HH-MM-SS",
)
parser.add_argument(
    "-d", "--dataset", type=str, help="Dataset file name yyyy-mm-dd-HH-MM-SS"
)
parser.add_argument(
    "-g", "--gpu", type=str2bool, help="Use GPU? True/False", default=False
)
parser.add_argument("-nm", "--name", type=str, help="Name", default="")


args = parser.parse_args()

start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + args.name

# vds = [
#     "videos/trajectories/2023-04-18-12-25-19/0/0.05.mp4",
#     "videos/trajectories/2023-04-18-12-25-19/0/0.1.mp4",
#     "videos/trajectories/2023-04-18-12-25-19/0/0.2.mp4",
#     "videos/trajectories/2023-04-18-12-25-19/0/0.3.mp4",
#     "videos/trajectories/2023-04-18-12-25-19/0/0.5.mp4",
# ]
# feedback = Feedback.request_human_feedback_from_videos(vds, 900, 450)
# print(feedback)
# sys.exit()

algorithm_name = args.algorithm_name
algorithm = None

dataset = env = None

if args.dataset != None:
    print(f"./datasets/{args.dataset}")
    dataset = Feedback.load_dataset_from_pickle(f"./datasets/{args.dataset}.pkl")
    # dataset = MDPDataset.load(f"./datasets/{args.dataset}")

if args.environment == "CartPole-v0":
    dataset, env = get_cartpole()
elif args.environment == "LunarLanderContinuous-v2":
    env = gym.make("LunarLanderContinuous-v2")
elif args.environment == "LunarLander-v2":
    env = gym.make("LunarLander-v2")
elif args.environment == "CarRacing-v1" or args.environment == "CarRacing-v2":
    env = gym.make("CarRacing-v2")
else:
    print("Could not find environment! Exiting...")
    sys.exit()

if algorithm_name == "dqn":
    algorithm = d3rlpy.algos.DQN(
        learning_rate=2.5e-4,
        optim_factory=d3rlpy.models.optimizers.RMSpropFactory(),
        q_func_factory="mean",
        scaler="pixel",
        target_update_interval=10000 // 4,
        n_frames=4,
        batch_size=32,
        use_gpu=args.gpu,
    )
elif algorithm_name == "cql":
    algorithm = d3rlpy.algos.DiscreteCQL(use_gpu=args.gpu)
elif algorithm_name == "cqlcontinuous":
    algorithm = d3rlpy.algos.CQL(use_gpu=args.gpu)
elif algorithm_name == "sac":
    algorithm = d3rlpy.algos.DiscreteSAC(use_gpu=args.gpu)
elif algorithm_name == "saccontinuous":
    algorithm = d3rlpy.algos.SAC(use_gpu=args.gpu)
elif algorithm_name == "nfq":
    algorithm = d3rlpy.algos.NFQ(batch_size=256, n_steps=1024, use_gpu=args.gpu)
elif algorithm_name == "ddqn":
    algorithm = d3rlpy.algos.DoubleDQN(batch_size=256, n_steps=1024, use_gpu=args.gpu)
elif algorithm_name == "td3plusbc":
    algorithm = d3rlpy.algos.TD3PlusBC(
        batch_size=256,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        n_steps=1024,
        use_gpu=args.gpu,
        scaler=MinMaxScaler(),
    )
else:
    algorithm = None

if args.trained_model != None:
    algorithm.build_with_env(env)
    algorithm.load_model(f"./trained_models/{algorithm_name}/{args.trained_model}.pt")

if args.mode == "demo":
    algorithm.fit_online(
        env,
        n_steps=1000,
        n_steps_per_epoch=1000,
        update_start_step=1000,
    )

    evaluate_scorer = evaluate_on_environment(env, render=True)
    rewards = evaluate_scorer(algorithm)
    sys.exit()

elif args.mode == "dataset":
    buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=args.steps, env=env)
    algorithm.collect(env, buffer, n_steps=args.steps)
    dataset = buffer.to_mdp_dataset()

    pathExists = os.path.exists(f"./datasets/{algorithm_name}")
    if not pathExists:
        os.makedirs(f"datasets/{algorithm_name}")
    dataset.dump(f"./datasets/{algorithm_name}/{start_time}.h5")
    sys.exit()

elif args.mode == "play":
    Feedback.playEnv(env, recording_name=start_time, record=True)
    sys.exit()

elif args.mode == "baseline1":
    # Prepare environment
    algorithm.build_with_env(env)
    # Replay buffer
    buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=1000000, env=env)
    explorer = d3rlpy.online.explorers.LinearDecayEpsilonGreedy(
        start_epsilon=1.0, end_epsilon=0.1, duration=1000000
    )

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
        tensorboard_dir=tensorboard_log_dir,
    )

    # Show a replay using fully trained model
    evaluate_scorer = evaluate_on_environment(env, render=True)
    rewards = evaluate_scorer(algorithm)

    # Save trained model
    pathExists = os.path.exists(f"./trained_models/{algorithm_name}")
    if not pathExists:
        os.makedirs(f"trained_models/{algorithm_name}")

    algorithm.save_model(f"./trained_models/{algorithm_name}/{start_time}.pt")
    sys.exit()


# train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

# algorithm.build_with_dataset(dataset)
# algorithm.build_with_env(env)
# td_error = td_error_scorer(algorithm, test_episodes)

# train offline
# algorithm.fit(dataset, n_steps=args.steps, n_steps_per_epoch=1000)


# # 3rd baseline

# offline_training_steps = int(args.steps / args.interventions)

# tensorboard_log_dir = f"tensorboard_logs/{algorithm_name}/{start_time}"
# if not os.path.exists(tensorboard_log_dir):
#     os.makedirs(tensorboard_log_dir)


# # initial offline
# train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

# algorithm.build_with_dataset(dataset)
# td_error = td_error_scorer(algorithm, test_episodes)

# algorithm.fit(
#     dataset,
#     n_steps=offline_training_steps,
#     n_steps_per_epoch=1000,
#     tensorboard_dir=f"{tensorboard_log_dir}-offline",
# )
# algorithm.save_model(
#     f"./trained_models/{algorithm_name}/{start_time}-initialoffline.pt"
# )

# # initial online

# buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=args.steps, env=env)
# explorer = d3rlpy.online.explorers.ConstantEpsilonGreedy(0.3)

# algorithm.fit_online(
#     env,
#     buffer,
#     explorer,
#     n_steps=1000,
#     n_steps_per_epoch=1000,
#     update_start_step=1000,
#     tensorboard_dir=f"{tensorboard_log_dir}-online",
#     save_interval=10,
# )

# algorithm.save_model(f"./trained_models/{algorithm_name}/{start_time}-initialonline.pt")

# interventions = get_intervention_step_array(
#     args.steps - offline_training_steps, args.interventions
# )
# interventions = np.flip(interventions)


# def update_rewards(rewards, feedback, best_reward, worst_reward):
#     feedback_dict = {int(rank): idx for idx, rank in enumerate(feedback)}

#     for rank, idx in feedback_dict.items():
#         reward_modification = (
#             best_reward - (rank - 1) * (best_reward - worst_reward) / 2
#         )
#         rewards[idx] += reward_modification

#     return rewards


# best_extra_reward = 5
# worst_extra_reward = -5


# for idx in range(0, args.interventions):
#     # generate trajectories

#     trajectory_epsilons = [0.05, 0.1, 0.2, 0.3, 0.5]
#     video_names: str = []

#     for epsilon in trajectory_epsilons:
#         states = []
#         actions = []
#         rewards = []
#         next_states = []
#         terminals = []

#         fourcc = cv2.VideoWriter_fourcc(*"mp4v")

#         pathExists = os.path.exists(f"videos/trajectories/{start_time}/{idx}")
#         if not pathExists:
#             os.makedirs(f"videos/trajectories/{start_time}/{idx}")

#         video_name = f"videos/trajectories/{start_time}/{idx}/{epsilon}.mp4"
#         video = cv2.VideoWriter(video_name, fourcc, float(50), (600, 400))
#         video_names.append(video_name)

#         observation = env.reset()
#         while True:
#             if np.random.random() < epsilon:
#                 action = env.action_space.sample()
#             else:
#                 action = algorithm.predict([observation])[0]

#             states.append(observation)
#             actions.append(action)

#             observation, reward, done, info = env.step(action)

#             rewards.append(reward)
#             next_states.append(observation)
#             terminals.append(done)
#             video.write(env.render(mode="rgb_array"))

#             if done:
#                 break

#         # save dataset

#         video.release()

#         pathExists = os.path.exists(f"./datasets/trajectories/{start_time}/{idx}")
#         if not pathExists:
#             os.makedirs(f"datasets/trajectories/{start_time}/{idx}")

#         data = {
#             "observations": np.array(states),
#             "actions": np.array(actions),
#             "rewards": np.array(rewards),
#             "next_states": np.array(next_states),
#             "terminals": np.array(terminals),
#             "discrete_action": isinstance(env.action_space, gym.spaces.Discrete),
#         }

#         with open(
#             f"./datasets/trajectories/{start_time}/{idx}/{epsilon}.pkl", "wb"
#         ) as f:
#             pickle.dump(data, f)

#     # evaluate trajectories
#     feedback = Feedback.request_human_feedback_from_videos(video_names, 900, 450)
#     if feedback is None:
#         print(
#             f"The window has exited without returning feedback! Ignoring the evaluation. Continuing the algorithm."
#         )

#     # edit datasets based on evaluation
#     for i, video_name in enumerate(video_names):
#         epsilon = trajectory_epsilons[i]

#         with open(
#             f"./datasets/trajectories/{start_time}/{idx}/{epsilon}.pkl", "rb"
#         ) as f:
#             data = pickle.load(f)

#         print(f"Fields in the pickle file (epsilon {epsilon}):", data.keys())
#         # Update the "rewards" based on human feedback
#         updated_rewards = update_rewards(
#             data["rewards"], feedback, best_extra_reward, worst_extra_reward
#         )
#         # Save the modified "rewards" to the pickle file
#         data["rewards"] = updated_rewards
#         with open(
#             f"./datasets/trajectories/{start_time}/{idx}/{epsilon}.pkl", "wb"
#         ) as f:
#             pickle.dump(data, f)

#     # offline training with data

#     dataset_folder = f"./datasets/trajectories/{start_time}/{idx}"

#     combined_data = {
#         "observations": [],
#         "actions": [],
#         "rewards": [],
#         "next_states": [],
#         "terminals": [],
#         "discrete_action": isinstance(env.action_space, gym.spaces.Discrete),
#     }

#     for epsilon in trajectory_epsilons:
#         with open(
#             f"./datasets/trajectories/{start_time}/{idx}/{epsilon}.pkl", "rb"
#         ) as f:
#             data = pickle.load(f)
#             combined_data["observations"].extend(data["observations"])
#             combined_data["actions"].extend(data["actions"])
#             combined_data["rewards"].extend(data["rewards"])
#             combined_data["next_states"].extend(data["next_states"])
#             combined_data["terminals"].extend(data["terminals"])

#     # Convert the lists to numpy arrays
#     for key in ["observations", "actions", "rewards", "next_states", "terminals"]:
#         combined_data[key] = np.array(combined_data[key])

#     with open(
#         f"./datasets/trajectories/{start_time}/{idx}/combined_data.pkl", "wb"
#     ) as f:
#         pickle.dump(combined_data, f)

#     combined_dataset = Feedback.load_dataset_from_pickle(
#         f"./datasets/trajectories/{start_time}/{idx}/combined_data.pkl"
#     )

#     # train offline

#     algorithm.fit(
#         dataset,
#         n_steps=offline_training_steps,
#         n_steps_per_epoch=1000,
#         tensorboard_dir=f"{tensorboard_log_dir}-offline",
#     )
#     algorithm.save_model(
#         f"./trained_models/{algorithm_name}/{start_time}{idx}offline.pt"
#     )

#     # train online

#     algorithm.fit_online(
#         env,
#         buffer,
#         explorer,
#         n_steps=offline_training_steps,
#         n_steps_per_epoch=1000,
#         update_start_step=1000,
#         tensorboard_dir=f"{tensorboard_log_dir}-online",
#         save_interval=10,
#     )

#     algorithm.save_model(
#         f"./trained_models/{algorithm_name}/{start_time}{idx}online.pt"
#     )


# # combine online training logs

# combine_tensorboard_logs(f"{tensorboard_log_dir}-online/runs")

# # anything else that might need to be done

# 0.3 epsilon batch size changed
# 0.15 epsilon
# 0.5 epsilon -> 0.05 linear decay
# 0.75 epsilon -> 0.01 linear decay
# nfq default linear decay
# nfq constant 0.3
# ddqn constant 0.3
# dqqn linear decay default

# ddqn constant 0.1
# ddqn constant 0.3
# ddqn constant 0.5
