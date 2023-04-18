import gym
from gym.utils.play import play
import numpy as np
import os
import pickle
from d3rlpy.dataset import MDPDataset
import gym
from tkVideoPlayer import TkinterVideo
import cv2
import os
import numpy as np
from tkVideoPlayer import TkinterVideo
import vlc
from typing import List
import sys
import tkinter as tk
from tkinter import ttk
from tkinter import Canvas

vlc_install_path = "C:\\Program Files\\VideoLAN\\VLC"  # Update this path according to your VLC installation path
os.environ["VLC_PLUGIN_PATH"] = os.path.join(vlc_install_path, "plugins")
sys.path.append(vlc_install_path)


class VLCVideoPlayer(ttk.Frame):
    def __init__(self, master=None, video_path=None, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.master = master
        self.video_path = video_path
        self.instance = vlc.Instance()
        self.player = self.instance.media_player_new()
        self.create_ui()
        self.event_manager = self.player.event_manager()
        self.event_manager.event_attach(
            vlc.EventType.MediaPlayerEndReached, self.end_reached
        )
        self.event_manager.event_attach(
            vlc.EventType.MediaPlayerTimeChanged, self.on_time_changed
        )

    def create_ui(self):
        self.Canvas = tk.Canvas(self, bg="black", width=600, height=400)
        self.Canvas.grid(row=0, column=0, padx=5, pady=5)
        self.player.set_hwnd(self.Canvas.winfo_id())

    def end_reached(self, event):
        self.after(100, self.restart)

    def on_time_changed(self, event):
        length = self.player.get_length()
        current_time = self.player.get_time()
        if length > 0 and current_time >= length:
            self.restart()

    def restart(self):
        self.player.stop()
        self.player.set_time(0)
        self.player.play()

    def play(self, event=None):
        media = self.instance.media_new(self.video_path)
        media.get_mrl()
        self.player.set_media(media)
        self.player.play()


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
        win = tk.Tk()
        win.geometry(f"{winWidth}x{winHeight}")
        win.title("Video Player")
        win.config(bg="black")

        video_players: List[VLCVideoPlayer] = []
        for idx, video in enumerate(videos):
            temp_player = VLCVideoPlayer(master=win)
            temp_player.video_path = video
            video_players.append(temp_player)

        def playVideos():
            for idx, video_player in enumerate(video_players):
                video_player.grid(row=idx // 3, column=idx % 3, padx=5, pady=5)
                video_player.play()

        playVideos()
        win.mainloop()

        # preference = -1

        # # Callback function to update preference variable
        # def set_preference(value):
        #     nonlocal preference
        #     preference = value
        #     win.destroy()

        # win.mainloop()

        # # Return user's preference
        # return preference
