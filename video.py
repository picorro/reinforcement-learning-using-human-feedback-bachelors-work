import gym
from tkVideoPlayer import TkinterVideo
import cv2
import os
import numpy as np
from tkVideoPlayer import TkinterVideo
from tkinter import *
from mtkinter import *
from typing import List


class VideoMaker:
    def getVideoFromEnvAndModel(
        env, model, fps: int, frameCount: int, width: int, height: int, outputFile: str
    ):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        pathExists = os.path.exists("videos")
        if not pathExists:
            os.makedirs("videos")
        video = cv2.VideoWriter(
            "videos/" + outputFile, fourcc, float(fps), (width, height)
        )

        observation, info = env.reset()

        # step_starting_index = 0
        # episode_index = 0

        # for step_index in range(frameCount):
        # action, _ = model.predict(observation)
        # observation, reward, terminated, truncated, info = env.step(action)

        # action, _ = model.predict(observation)
        # observation, reward, terminated, truncated, info = env.step(action)
        # if truncated or terminated:
        #     save_video(
        #         env.render(),
        #         "videos",
        #         fps=env.metadata["render_fps"],
        #         step_starting_index=step_starting_index,
        #         episode_index=episode_index
        #     )
        #     step_starting_index = step_index + 1
        #     episode_index += 1
        #     env.reset()

        for _ in range(frameCount):
            action, _ = model.predict(observation)
            observation, reward, terminated, truncated, info = env.step(action)

            video.write(env.render())

            if terminated or truncated:
                observation, info = env.reset()
        video.release()


class HumanFeedback:
    def requestHumanFeedbackFromVideos(videos, winWidth: int, winHeight: int):
        video_width = 600
        video_height = 400
        win = Tk()

        canvas = Canvas(win, width=winWidth, height=winHeight)
        canvas.pack()

        # prefer_1_button = Button(text="Prefer 1", command=lambda: set_preference(0))
        # prefer_2_button = Button(text="Prefer 2", command=lambda: set_preference(1))
        # no_preference_button = Button(
        #     text="No preference", command=lambda: set_preference(2)
        # )
        # replay_button = Button(text="Replay", command=lambda: playVideos())

        # Add buttons to canvas
        # prefer_1_button_window = canvas.create_window(
        #     30, winHeight - 30, anchor="n", window=prefer_1_button
        # )
        # prefer_2_button_window = canvas.create_window(
        #     80, winHeight - 30, anchor="n", window=prefer_2_button
        # )
        # no_preference_button_window = canvas.create_window(
        #     150, winHeight - 30, anchor="n", window=no_preference_button
        # )
        # replay_button_window = canvas.create_window(
        #     300, winHeight - 30, anchor="nw", window=replay_button
        # )

        video_players: List[TkinterVideo] = []
        for idx, video in enumerate(videos):
            temp_player = TkinterVideo(master=win, scaled=True)
            temp_player.load(video)
            temp_player.pack(expand=True, fill="both")
            canvas.create_window(
                0, 10, anchor="nw", window=temp_player, height=400, width=600
            )
            video_players.append(temp_player)

        def playVideos():
            for video_player in video_players:
                video_player.play()

        playVideos()

        preference = -1

        # Callback function to update preference variable
        def set_preference(value):
            nonlocal preference
            preference = value
            win.destroy()

        win.mainloop()

        # Return user's preference
        return preference

    def playVideo(video1, winWidth: int, winHeight: int):
        win = Tk()

        canvas = Canvas(win, width=winWidth, height=winHeight)
        canvas.pack()

        replay_button = Button(text="Replay", command=lambda: playVideo())

        # Add buttons to canvas
        replay_button_window = canvas.create_window(
            300, winHeight - 30, anchor="nw", window=replay_button
        )

        # Player 1
        videoplayer = TkinterVideo(master=win, scaled=True)
        videoplayer.load(video1)
        videoplayer.pack(expand=True, fill="both")

        videoCanvas = canvas.create_window(
            0,
            10,
            anchor="nw",
            window=videoplayer,
            height=winHeight - 100,
            width=winWidth,
        )

        def playVideo():
            videoplayer.play()

        playVideo()

        win.mainloop()
