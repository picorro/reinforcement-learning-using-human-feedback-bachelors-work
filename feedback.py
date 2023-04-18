import gym
from gym.utils.play import play
import numpy as np
import os
import pickle
from d3rlpy.dataset import MDPDataset
import gym
from tkVideoPlayer import TkinterVideo
import numpy as np
from tkVideoPlayer import TkinterVideo
from tkvideo import tkvideo
from typing import List
import re
from mkinter import *


class Feedback:
    def playEnv(self, env, recording_name="", record=False):
        key_to_action = {}

        if env.spec.id == "CartPole-v0" or env.spec.id == "CartPole-v1":
            key_to_action = {
                (ord("a"),): 0,  # LEFT
                (ord("d"),): 1,  # RIGHT
            }
        elif env.spec.id == "LunarLander-v2":
            key_to_action = {
                (ord("s"),): 0,  # S: Do nothing
                (ord("a"),): 1,  # A: Fire left engine
                (ord("w"),): 2,  # W: Fire main engine
                (ord("d"),): 3,  # D: Fire right engine
            }

        # Determine environment's properties
        is_discrete_action = isinstance(env.action_space, gym.spaces.Discrete)

        # Initialize lists to store data
        states = []
        actions = []
        rewards = []
        next_states = []
        terminals = []

        def save_data_callback(obs_t, obs_tp1, action, rew, done, info):
            states.append(obs_t)
            actions.append(action)
            rewards.append(rew)
            next_states.append(obs_tp1)
            terminals.append(done)

        play(env, keys_to_action=key_to_action, callback=save_data_callback)

        if record:
            pathExists = os.path.exists(f"./datasets/play/{env.spec.id}")
            if not pathExists:
                os.makedirs(f"datasets/play/{env.spec.id}")

            data = {
                "observations": np.array(states),
                "actions": np.array(actions),
                "rewards": np.array(rewards),
                "next_states": np.array(next_states),
                "terminals": np.array(terminals),
                "discrete_action": isinstance(env.action_space, gym.spaces.Discrete),
            }

            with open(f"./datasets/play/{env.spec.id}/{recording_name}.pkl", "wb") as f:
                pickle.dump(data, f)

    def load_dataset_from_pickle(file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        dataset = MDPDataset(
            observations=data["observations"],
            actions=data["actions"],
            rewards=data["rewards"],
            terminals=data["terminals"],
            discrete_action=data["discrete_action"],
        )

        return dataset

    def request_human_feedback_from_videos(videos, winWidth: int, winHeight: int):
        video_width = 300
        video_height = 200
        win = Tk()
        win.title("Human Evaluation")

        canvas = Canvas(win, width=winWidth, height=winHeight - 50)
        canvas.pack()

        video_players: List[tkvideo] = []
        for idx, video in enumerate(videos):
            label = Label(win, width=video_width, height=video_height)
            x = video_width * ((idx) % 3)
            y = video_height * (int)(idx / 3)
            label.place(x=x, y=y)
            temp_player = tkvideo(
                video, label, loop=1, size=(video_width, video_height)
            )
            video_players.append(temp_player)

        description = Label(win, text="Evaluation (example: 12345)")
        description.pack(side=LEFT, pady=30)
        feedback_field = Entry(win, width=40)
        feedback_field.pack(side=LEFT, pady=30)
        submit_button = Button(win, text="Submit")
        submit_button.pack(side=LEFT, padx=10, pady=30)

        feedback = None

        def submit_feedback():
            nonlocal feedback
            feedback = feedback_field.get()
            pattern = r"^(?!.*([12345]).*\1)[12345]{5}$"
            if re.match(pattern, feedback):
                print("Feedback submitted:", feedback)
                win.destroy()
            else:
                print(
                    "Invalid feedback. Please enter exactly 5 unique digits [12345] in your selected order."
                )

        submit_button.config(command=submit_feedback)

        def playVideos():
            for video_player in video_players:
                video_player.play()

        playVideos()
        win.mainloop()

        return feedback
