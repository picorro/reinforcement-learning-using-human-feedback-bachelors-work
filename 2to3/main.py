import sys
import gymnasium as gym
sys.modules["gym"] = gym
from utils import *
from model_keras import *
import math
from humancritic_tensorflow import *
import argparse
import random

# Args

parser = argparse.ArgumentParser()
parser.add_argument('--load_model_datetime', type=str, default="",
					help='datetime load rl_model / hc_model')
parser.add_argument('--test', type=bool, default=False,
					help='only test a random model or a pretrained one with the use of --load_model_datetime')
parser.add_argument('--ask_type', type=str, default="ask_human",
					help='ask_types: ask_human, ask_total_reward')
args = parser.parse_args()

def render(env,recorde=False):

	mean_reward = 0.0
	mean_traj_reward = 0.0
	max_run_time = 0.0
	min_run_time = 1e+10
	mean_run_time = 0.0
	for i in range(5):
		total_reward = 0.0
		traj_total_reward = 0.0

		idx = 0
		done=False
		obs = env.reset()
		while done == False:
			env.render()
			x = np.reshape(obs,[1,-1])
			pred = rl_model.run(x,None)
			action = np.argmax(pred)

			obs,_,done,info = env.step(action)
			total_reward += _
			traj_total_reward += hc_model.predict(obs.reshape([1,-1]))


			idx += 1
			if done or idx > 300:
				
				if idx > max_run_time:
					max_run_time = idx
				elif idx < min_run_time:
					min_run_time = idx
				mean_run_time += idx

				mean_reward += total_reward
				mean_traj_reward += traj_total_reward
				break
	print("[ RunLength =",5," MeanReward =",mean_reward / 5.0, "MeantrajReward =",mean_traj_reward/5.0,\
			" MeanRunTime =",mean_run_time / 5.0, " MaxRunTime =",max_run_time," MinRunTime =",min_run_time,"]")


env_name = "LunarLander-v2"
env = gym.make(env_name, render_mode="rgb_array")
obs, _ = env.reset()

action_is_box = type(env.action_space) == gym.spaces.box.Box
if action_is_box:
	action_space_n = np.sum(env.action_space.shape)
else:
	action_space_n = env.action_space.n

now = datetime.now()
datetime_str = now.strftime("%Y-%m-%d_%H-%M-%S")
rl_model = RLModel(len(obs),env.action_space.n,datetime_str,layer_sizes=[len(obs)**3,len(obs)**3,len(obs)**3])
mini_batch = MiniBatch(len(obs),env.action_space.n,batch_size=50)
hc_model = HumanCritic(len(obs),env.action_space.n,datetime_str)

if args.load_model_datetime != "":
	datetime_str = args.load_model_datetime
	rl_model.load(datetime_str)
	hc_model.load(datetime_str)

eps = 1.0
eps_discount_factor = 0.0005
eps_freq = 100
render_freq = 11
train_freq = 1

ask_types = ["ask_human","ask_total_reward"]
ask_type = args.ask_type

hc_ask_human_freq_episodes = 100000
hc_train_freq = 1
hc_loss_mean = 1
hc_loss_mean_freq = 10
hc_loss_mean_c = 1

hc_trijectory_interval = 1
hc_tricectory_c = 0
trijectory = None
trijectory_seed = np.random.randint(low=0, high=2**31 - 1, dtype=np.int32)
trijectory_env_name = env_name

save_freq = 100

#with tf.Session() as sess:
#	sess.run(tf.global_variables_initializer())
sess=None
mean_reward = 0.0
run_time = 1

run_id = 1
idx = 1

while True:
	frame = 0
	done=False
	trij_total_reward = 0
	total_reward = 0
	max_a,min_a = 0,100

	avg_loss = 0.0
	run_start = idx

	action_strength = np.zeros([env.action_space.n],dtype=np.int32)

	# Status update
	if run_id % run_time == 0:
		print("[ Episode:",run_id," Mean-Reward:",mean_reward/float(run_time), " MeanHCLoss:",hc_loss_mean/hc_loss_mean_c," Epsilon:",eps,"]")
		mean_reward = 0.0

	# Ask_X
	if run_id > 0 and run_id % hc_ask_human_freq_episodes == 0:
		if ask_type == ask_types[0]:
			hc_model.ask_human()
		elif ask_type == ask_types[1]:
			hc_model.ask_total_reward()

	# Trijectory
	if run_id % hc_trijectory_interval == 0 and run_id % render_freq != 0:
		trijectory_seed = random.randint(0, 2**32 - 1)
		trijectory_env_name = env_name
		trijectory = []
		trij_obs_list = []

		env = set_env_seed(env,trijectory_seed)


	# Save
	if run_id % save_freq == 0:
		hc_model.save()
		rl_model.save()
		print("Saved")

	# HC_Train
	if run_id % hc_train_freq == 0:
		hc_loss = hc_model.train()
		hc_loss_mean += hc_loss
		hc_loss_mean_c += 1
		if hc_loss_mean_c > hc_loss_mean_freq:
			hc_loss_mean_c = 1.0
			hc_loss_mean = hc_loss

	obs, abc = env.reset()
	while done == False:

		x = np.reshape(obs,[1,-1])
		pred = rl_model.run(x,sess)

		if np.random.uniform() < eps:
			action = np.random.randint(env.action_space.n)
		else:
			action = np.argmax(pred)
			action_strength[action] += 1

		old_obs = obs.copy()
		observation, reward, terminated, truncated, info = env.step(action)
		done = truncated or terminated
		reward = hc_model.predict(np.reshape(obs,[1,-1]))[0]
		trij_total_reward += reward
		total_reward += reward

		if trijectory != None:
			trijectory.append([old_obs.copy(),obs.copy(),action,done])

		mini_batch.add_sample(old_obs.copy(),obs.copy(),_,action,done=done)

		if idx % train_freq == 0:
			rl_model.train(mini_batch.get_batch_hc(rl_model,sess,hc_model),sess)

		eps = 0.1+(1.0-0.1)*math.exp(-eps_discount_factor*idx)


		if done or idx - run_start > 300:
			mean_reward += total_reward

			if trijectory != None:
				hc_model.add_trijactory(trijectory_env_name,trijectory_seed,trij_total_reward,trijectory)
				trijectory = None
			env.close()
			break

		idx += 1
	run_id+=1