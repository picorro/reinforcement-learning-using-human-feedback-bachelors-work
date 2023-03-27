import sys
import gymnasium as gym
sys.modules["gym"] = gym
from utils import *
from model_keras import *
import math
from humancritic_tensorflow import *
import argparse
import random
from video import *
import json
import os

# Args

parser = argparse.ArgumentParser()
parser.add_argument('--load_model_datetime', type=str, default="",
					help='datetime load rl_model / hc_model')
parser.add_argument('--test', type=bool, default=False,
					help='only test a random model or a pretrained one with the use of --load_model_datetime')
args = parser.parse_args()


def save_hyperparameters(hyperparameters, datetime_str):
    with open(f"hyperparameters_{datetime_str}.json", "w") as outfile:
        json.dump(hyperparameters, outfile)


def load_hyperparameters(datetime_str):
    with open(f"hyperparameters_{datetime_str}.json", "r") as infile:
        return json.load(infile)
    
def get_hyperparameters_dict(eps, eps_discount_factor, eps_freq, train_freq, hc_ask_human_freq_episodes, hc_train_freq, hc_loss_mean, hc_loss_mean_freq, hc_loss_mean_c, hc_trijectory_interval, hc_tricectory_c, idx, ask_type):
    hyperparameters_dict = {}
    hyperparameters_dict['eps'] = eps
    hyperparameters_dict['eps_discount_factor'] = eps_discount_factor
    hyperparameters_dict['eps_freq'] = eps_freq
    hyperparameters_dict['train_freq'] = train_freq
    hyperparameters_dict['hc_ask_human_freq_episodes'] = hc_ask_human_freq_episodes
    hyperparameters_dict['hc_train_freq'] = hc_train_freq
    hyperparameters_dict['hc_loss_mean'] = hc_loss_mean
    hyperparameters_dict['hc_loss_mean_freq'] = hc_loss_mean_freq
    hyperparameters_dict['hc_loss_mean_c'] = hc_loss_mean_c
    hyperparameters_dict['hc_trijectory_interval'] = hc_trijectory_interval
    hyperparameters_dict['hc_tricectory_c'] = hc_tricectory_c
    hyperparameters_dict['idx'] = idx
    hyperparameters_dict['ask_types'] = ask_types
    return hyperparameters_dict
    
env_name = "CartPole-v1"
env = gym.make(env_name, render_mode="rgb_array")
obs, _ = env.reset()

action_is_box = type(env.action_space) == gym.spaces.box.Box
if action_is_box:
	action_space_n = np.sum(env.action_space.shape)
else:
	action_space_n = env.action_space.n

programStartTime = datetime.now()
datetime_str = programStartTime.strftime("%Y-%m-%d_%H-%M-%S")
rl_model = RLModel(len(obs),env.action_space.n,datetime_str,layer_sizes=[len(obs)**3,len(obs)**3,len(obs)**3])
mini_batch = MiniBatch(len(obs),env.action_space.n,batch_size=50)
hc_model = HumanCritic(len(obs),env.action_space.n,datetime_str)

eps = 0.5
eps_discount_factor = 0.00008
eps_freq = 100
train_freq = 1

ask_types = ["ask_human", "ask_total_reward"]
ask_type = ask_types[1]

hc_ask_human_freq_episodes = 1
hc_train_freq = 1
hc_loss_mean = 1
hc_loss_mean_freq = 10
hc_loss_mean_c = 1

hc_trijectory_interval = 1
hc_tricectory_c = 0
trijectory = None
trijectory_seed = np.random.randint(low=0, high=2**31 - 1, dtype=np.int32)
trijectory_env_name = env_name

total_env_reward = 0

save_freq = 100

sess=None
mean_reward = 0.0

run_id = 1
idx = 1


if args.load_model_datetime != "":
	datetime_str = args.load_model_datetime
	rl_model.load(datetime_str)
	hc_model.load(datetime_str)
	loaded_hyperparameters = load_hyperparameters(datetime_str)
	eps = loaded_hyperparameters["eps"]
	eps_discount_factor = loaded_hyperparameters["eps_discount_factor"]
	eps_freq = loaded_hyperparameters["eps_freq"]
	train_freq = loaded_hyperparameters["train_freq"]
	hc_ask_human_freq_episodes = loaded_hyperparameters["hc_ask_human_freq_episodes"]
	hc_train_freq = loaded_hyperparameters["hc_train_freq"]
	hc_loss_mean = loaded_hyperparameters["hc_loss_mean"]
	hc_loss_mean_freq = loaded_hyperparameters["hc_loss_mean_freq"]
	hc_loss_mean_c = loaded_hyperparameters["hc_loss_mean_c"]
	hc_trijectory_interval = loaded_hyperparameters["hc_trijectory_interval"]
	hc_tricectory_c = loaded_hyperparameters["hc_tricectory_c"]
	idx = loaded_hyperparameters["idx"]

if args.test == True:
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')

	pathExists = os.path.exists('videos')
	if not pathExists:
		os.makedirs('videos')

	video = cv2.VideoWriter('videos/test'+".mp4", fourcc, float(30), (600, 400))

	idx = 0
	obs, _ = env.reset()
	total_reward = 0
	while idx < 1000:
	
		x = np.reshape(obs,[1,-1])
		pred = rl_model.run(x,sess)

		action = np.argmax(pred)
		obs, reward, terminated, truncated, info = env.step(action)
		done = truncated or terminated
		total_reward += reward
		video.write(env.render())
		if done:
			seed = random.randint(0, 2**32 - 1)
			env = set_env_seed(env,seed)
			obs, _ = env.reset()
			print(f'{total_reward=}')
			total_reward = 0
		idx += 1
	env.close()
	video.release()
	HumanFeedback.playVideo("./videos/test.mp4", 600, 500)
	
else:
	while True:
		if eps <= 0:
			break

		frame = 0
		done=False
		trij_total_reward = 0
		total_reward = 0
		max_a,min_a = 0,100

		avg_loss = 0.0
		run_start = idx

		print("[ Episode:",run_id," Mean-Reward:", mean_reward, "Episode reward:", total_env_reward, " MeanHCLoss:",hc_loss_mean/hc_loss_mean_c," Epsilon:",eps,"]")
		mean_reward = 0.0
		total_env_reward = 0

		# Ask_X
		if run_id > 0 and run_id % hc_ask_human_freq_episodes == 0:
			if ask_type == ask_types[0]:
				hc_model.ask_human()
			elif ask_type == ask_types[1]:
				hc_model.ask_total_reward()

		# Trijectory
		if run_id % hc_trijectory_interval == 0:
			trijectory_seed = random.randint(0, 2**32 - 1)
			trijectory_env_name = env_name
			trijectory = []
			trij_obs_list = []

			env = set_env_seed(env,trijectory_seed)


		# Save
		if run_id % save_freq == 0:
			hc_model.save()
			rl_model.save()
			hyperparameters = get_hyperparameters_dict(eps, eps_discount_factor, eps_freq, train_freq, hc_ask_human_freq_episodes, hc_train_freq, hc_loss_mean, hc_loss_mean_freq, hc_loss_mean_c, hc_trijectory_interval, hc_tricectory_c, idx, ask_type)
			save_hyperparameters(hyperparameters, datetime_str)
			print("Saved")

		#HC_Train
		if run_id % hc_train_freq == 0:
			hc_loss = hc_model.train()
			hc_loss_mean += hc_loss
			hc_loss_mean_c += 1
			if hc_loss_mean_c > hc_loss_mean_freq:
				hc_loss_mean_c = 1.0
				hc_loss_mean = hc_loss


		# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		# pathExists = os.path.exists('videos')
		# if not pathExists:
		# 	os.makedirs('videos')
		# video = cv2.VideoWriter('videos/test'+".mp4", fourcc, float(30), (600, 400))

		obs, _ = env.reset()
		while done == False:
			x = np.reshape(obs,[1,-1])
			pred = rl_model.run(x,sess)

			if np.random.uniform() < eps:
				action = np.random.randint(env.action_space.n)
			else:
				action = np.argmax(pred)

			old_obs = obs.copy()
			observation, env_reward, terminated, truncated, info = env.step(action)
			total_env_reward += env_reward
			done = truncated or terminated
			reward = hc_model.predict(np.reshape(obs,[1,-1]))[0]
			trij_total_reward += reward
			total_reward += reward

			if trijectory != None:
				trijectory.append([old_obs.copy(),obs.copy(),action,done])

			mini_batch.add_sample(old_obs.copy(),obs.copy(),_,action,done=done)

			if idx % train_freq == 0:
				rl_model.train(mini_batch.get_batch_hc(rl_model,sess,hc_model),sess)

			# video.write(env.render())
			if done:
				mean_reward += total_reward
				eps = eps * math.exp(-eps_discount_factor * run_id)
				#eps = eps - eps_discount_factor
				if trijectory != None:
					hc_model.add_trijactory(trijectory_env_name,trijectory_seed,trij_total_reward,trijectory)
					trijectory = None
				# video.release()
				# HumanFeedback.playVideo('./videos/test.mp4', 600, 400)
				env.close()

			idx += 1
		run_id+=1
if args.test == False:
	print("Training finished!")
else:
	print("Replay finished!")
