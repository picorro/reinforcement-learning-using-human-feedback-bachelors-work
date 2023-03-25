import sys
import gymnasium as gym
sys.modules["gym"] = gym
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import math
import tensorflow as tf
import tf_slim as slim
import cv2
from time import sleep
from datetime import datetime
import os
from tkVideoPlayer import TkinterVideo
from tkinter import *
from video import * 

def set_env_seed(env,seed):
	env.np_random = np.random.RandomState(seed)
	env.action_space.seed(seed)
	return env

class HumanCritic(object):
	LAYER_COUNT = 3
	NEURON_SIZE = 64
	BATCH_SIZE = 10
	LEARNING_RATE = 0.00025

	def __init__(self,obs_size,action_size,datetime_str):
		self.trijectories = []
		self.preferences = []

		self.obs_size = obs_size
		self.action_size = action_size
		self.NEURON_SIZE = obs_size**3
		self.datetime_str = datetime_str

		self.init_tf()


	def init_tf(self):
		tf.compat.v1.reset_default_graph()
		self.graph =tf.Graph()
		with self.graph.as_default():
			self.initializer = tf.compat.v1.truncated_normal_initializer(stddev=0.3)

			self.input_o0 = tf.compat.v1.placeholder(shape=[None,self.obs_size],dtype=tf.float32)
			self.input_o1 = tf.compat.v1.placeholder(shape=[None,self.obs_size],dtype=tf.float32)
			self.preference_distribution = tf.compat.v1.placeholder(shape=[2],dtype=tf.float32)
			self.model_o0 = self.create_model(self.input_o0)
			self.model_o1 = self.create_model(self.input_o1,reuse=True)
			self.batch_sizes = tf.compat.v1.placeholder(shape=[2],dtype=tf.float32)
			#'''
			self.model_o0_sum = tf.exp(tf.divide(tf.reduce_sum(self.model_o0),self.batch_sizes[0]))
			self.model_o1_sum = tf.exp(tf.divide(tf.reduce_sum(self.model_o1),self.batch_sizes[1]))
			#self.model_o1_sum = tf.exp(tf.reduce_sum(self.model_o1))
			self.p_o0_o1 = tf.divide(self.model_o0_sum,tf.add(self.model_o0_sum,self.model_o1_sum))
			self.p_o1_o0 = tf.divide(self.model_o1_sum,tf.add(self.model_o1_sum,self.model_o0_sum))
			self.loss = -tf.add(tf.multiply(self.preference_distribution[0],tf.compat.v1.log(self.p_o0_o1)), \
					tf.multiply(self.preference_distribution[1],tf.compat.v1.log(self.p_o1_o0)))

			'''
			self.model_o0_sum = tf.exp(tf.reduce_sum(self.model_o0))
			self.model_o1_sum = tf.exp(tf.reduce_sum(self.model_o1))
			self.p_o0_o1 = tf.add(1e-5,tf.divide(self.model_o0_sum,tf.add(1e-5,tf.add(self.model_o0_sum,self.model_o1_sum))))
			self.p_o1_o0 = tf.add(1e-5,tf.divide(self.model_o1_sum,tf.add(1e-5,tf.add(self.model_o1_sum,self.model_o0_sum))))
			self.loss = tf.add(1e-5,-tf.add(tf.multiply(self.preference_distribution[0],tf.log(self.p_o0_o1)), \
					tf.multiply(self.preference_distribution[1],tf.log(self.p_o1_o0))))
			#'''
			self.train_step = tf.compat.v1.train.AdamOptimizer(learning_rate=self.LEARNING_RATE).minimize(self.loss)
			self.sess = tf.compat.v1.Session()
			self.sess.run(tf.compat.v1.global_variables_initializer())

			self.saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
			self.checkpoint_path = "./human_critic"+ self.datetime_str 


	def load(self,datetime_str):
		with self.graph.as_default():
			pathstr = os.path.join(os.getcwd(), "human_critic" + datetime_str + ".ckpt.meta")
			if not os.path.exists(pathstr):
				print("File does not exist:", pathstr)
				return
			ckpt = tf.train.get_checkpoint_state(pathstr)
			print(ckpt)
			self.saver.restore(self.sess,ckpt.model_checkpoint_path)

	def save(self):
		print(self.checkpoint_path + str(os.path.isdir(os.path.dirname(self.checkpoint_path))))
		if not os.path.exists(os.path.dirname(self.checkpoint_path)):
			os.makedirs(self.checkpoint_path)
		with self.graph.as_default():
			self.saver.save(self.sess,self.checkpoint_path + ".ckpt")

	def _create_model(self,input_data,reuse=False):
		with self.graph.as_default():
			with tf.compat.v1.variable_scope("hp_model"):
				model = input_data
				# Programatically define Layers
				for i in range(self.LAYER_COUNT):
					layer = slim.fully_connected(model,self.NEURON_SIZE,activation_fn=tf.nn.relu,scope="hp_model_"+str(i),
						reuse=reuse,weights_initializer=self.initializer)
					model = layer

				layer = slim.fully_connected(model,1,scope="output",\
					reuse=reuse,weights_initializer=self.initializer)

				#'''
				model = layer 
				layer = tf.nn.batch_normalization(model,tf.constant(0.0,shape=[1]),\
					tf.constant(1.0,shape=[1]),None,None,1e-5)
				#'''
				return layer

	def create_model(self,input_data,reuse=False):
		with self.graph.as_default():
			with tf.compat.v1.variable_scope("hp_model"):
				model = input_data
				# Programatically define Layers
				for i in range(self.LAYER_COUNT):
					layer = tf.compat.v1.layers.dense(model,self.NEURON_SIZE,activation=tf.nn.relu,kernel_initializer=self.initializer,
							use_bias=True,name="hp_model_"+str(i),trainable=True,reuse=reuse)
					model = layer

				layer = tf.compat.v1.layers.dense(model,1,name="output",\
					reuse=reuse,kernel_initializer=self.initializer,trainable=True,use_bias=False)

				#'''
				model = layer 
				layer = tf.nn.batch_normalization(model,mean=0,variance=5**2,offset=0,\
					scale=1.0,variance_epsilon=1e-5,name="batch_norm_layer")
				#'''
				return layer

	def train(self):
		if len(self.preferences) < 5:
			return 0

		batch_size = min(len(self.preferences),self.BATCH_SIZE)
		r = np.asarray(list(range(len(self.preferences))))
		np.random.shuffle(r)

		mean_loss = 0.0
		min_loss = 1e+10
		max_loss = -1e+10
		with self.graph.as_default():
			for i in r[:batch_size]:
				o0,o1,preference = self.preferences[i]

				pref_dist = np.zeros([2],dtype=np.float32)
				if preference < 2:
					pref_dist[preference] = 1.0
				else:
					pref_dist[:] = 0.5

				with self.graph.as_default():

					feed_dict = {self.input_o0: o0, self.input_o1: o1,self.preference_distribution: pref_dist,self.batch_sizes: [len(o0),len(o1)]}
					m0,m1,loss,_ = self.sess.run([self.model_o0_sum,self.model_o1_sum,self.loss,self.train_step],feed_dict=feed_dict)
					
					if loss > max_loss:
						max_loss = loss
					elif loss < min_loss:
						min_loss = loss
					#print loss, self.predict([o0[0]])

				mean_loss += np.sum([loss])
			print("[ Loss: mean =",mean_loss," max =",max_loss," min =",min_loss,"]")
		return mean_loss / len(r)

	def predict(self,obs):
		with self.graph.as_default():
			feed_dict = {self.input_o1: obs}
			return self.sess.run(self.model_o1,feed_dict=feed_dict)

	def add_preference(self,o0,o1,preference):
		self.preferences.append([o0,o1,preference])

	def add_trijactory(self,trijectory_env_name,trijectory_seed,total_reward,trijectory):
		self.trijectories.append([trijectory_env_name,trijectory_seed,total_reward,trijectory])

	def _ask_human(self,env,rl_model,sess):
		pref = []
		preference = -1	# 1=0; 2=1; 3=equal; -1=undecided

		for i in range(2):
			done=False
			obs = env.reset()
			obs_list = []
			idx = 0
			while not done:
				env.render()

				x = np.reshape(obs,[1,-1])
				pred = rl_model.run(x,sess)

				action = np.argmax(pred)

				obs,_,done,info = env.step(action)
				obs_list.append(obs)

				idx += 1

				if idx > 300:
					break

				if done:
					break

			pref.append(obs_list)

		str_preference = input("Pref(1>,2>|3=): ")
		if str_preference == "1":
			preference = 0
		elif str_preference == "2":
			preference = 1
		elif preference == "3":
			preference = 2
		else:
			preference = -1
			return

		self.add_preference(pref[0],pref[1],preference)

	def ask_human(self):
		if len(self.trijectories) < 2:
			return

		r = np.asarray(list(range(len(self.trijectories))))
		np.random.shuffle(r)
		t = [self.trijectories[r[0]],self.trijectories[r[1]]]
		

		fourcc = cv2.VideoWriter_fourcc(*'mp4v')

		pathExists = os.path.exists('videos')
		if not pathExists:
			os.makedirs('videos')

		# Prepare environments
		envs = []
		videos = []
		for i in range(len(t)):
			env_name,seed,_,trijectory = t[i]
			env = gym.make(env_name, render_mode="rgb_array")
			env = set_env_seed(env,seed)
			env.reset()
			envs.append(env)
			videos.append(cv2.VideoWriter('videos/' + str(i)+".mp4", fourcc, float(30), (600, 400)))

		# Play environments
		print("Waiting for evaluation, open the Tk() window")
		env_idxs = np.zeros([2],dtype=np.int32)
		preference = -1
		for i in range(300):
			for i in range(len(t)):
				videos[i].write(envs[i].render())

				env_name,seed,_,trijectory = t[i]
				obs,future_obs,action,done = trijectory[env_idxs[i]]
				envs[i].step(action)
				env_idxs[i]+=1
				# Reset envirnment
				if done or env_idxs[i] >= len(trijectory):					
					envs[i] = set_env_seed(envs[i],seed)
					envs[i].reset()
					env_idxs[i] = 0
		for video in videos:
			video.release()

		preference = HumanFeedback.requestHumanFeedbackFromVideos("./videos/0.mp4", "./videos/1.mp4", 1200, 500)

		if preference != -1:
			# Generate observation list
			observations = []
			for i in range(len(t)):
				env_name,seed,_,trijectory = t[i]
				o = []

				for j in range(len(trijectory)):
					o.append(trijectory[j][1])

				observations.append(o)

			self.add_preference(observations[0],observations[1],preference)

		for i in range(len(envs)):
			envs[i].close()

		if preference == 0:
			print(1)
		elif preference == 1:
			print(2)
		elif preference != -1:
			print("neutral")
		else:
			print("no oppinion")

	def ask_total_reward(self):

		if len(self.trijectories) < 2:
			return

		r = np.asarray(list(range(len(self.trijectories))))
		np.random.shuffle(r)
		t = [self.trijectories[r[0]],self.trijectories[r[1]]]

		if t[0][2] > t[1][2]:
			preference = 0
		elif t[0][2] < t[1][2]:
			preference = 1
		else:
			preference = 2

		if preference != -1:
			# Generate observation list
			os = []
			for i in range(len(t)):
				env_name,seed,total_reward,trijectory = t[i]
				o = []

				for j in range(len(trijectory)):
					o.append(trijectory[j][1])

				os.append(o)

			self.add_preference(os[0],os[1],preference)

		'''
		if preference == 0:
			print "Preference:",1
		elif preference == 1:
			print  "Preference:",2
		elif preference != -1:
			print  "Preference:","neutral"
		else:
			print  "Preference:","no oppinion"
		'''

	def play_video(self,path0,path1):
		ret_val = -1
		paths = [path0,path1]
		cap = [cv2.VideoCapture(path0),cv2.VideoCapture(path1)]
		while(True):
			if ret_val != -1:
				break
			for i in range(len(cap)):
				ret,frame = cap[i].read()
				if not ret:
					cap[i].release()
					cap[i] = cv2.VideoCapture(paths[i])
				else:
					cv2.imshow("frame"+str(i),frame)
					key = cv2.waitKey(1)
					if key & 0xFF == ord('1'):
						ret_val= 0
						break
					elif key & 0xFF == ord('2'):
						ret_val= 1
						break
					elif key & 0xFF == ord('0'):
						ret_val= 2
						break

		for i in range(len(cap)):
			cap[i].release()
		return ret_val