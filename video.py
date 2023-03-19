import sys
import gymnasium as gym
sys.modules["gym"] = gym
from gymnasium.utils.save_video import save_video
from tkVideoPlayer import TkinterVideo
import cv2
import os
import numpy as np
from tkVideoPlayer import TkinterVideo
from tkinter import *

class VideoMaker:

    def getVideoFromEnvAndModel(env, model, fps: int, frameCount: int, width: int, height: int, outputFile: str):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        pathExists = os.path.exists('videos')
        if not pathExists:
            os.makedirs('videos')
        video = cv2.VideoWriter('videos/' + outputFile, fourcc, float(fps), (width, height))

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
    def requestHumanFeedbackFromVideos(video1, video2, winWidth: int, winHeight: int):

        def pref1Callback():
            win.destroy()
            return 1
        def pref2Callback():
            win.destroy()
            return 2
        def noPrefCallback():
            win.destroy()
            return 0
        def resultsIncomparableCallback():
            win.destroy()
            return -1

        
        win = Tk()
        canvas = Canvas(win, width=winWidth, height=winHeight)
        canvas.pack()

        prefer_1_button = Button(text="Prefer 1", command=pref1Callback)
        prefer_2_button = Button(text="Prefer 2", command=pref2Callback)
        no_preference_button = Button(text="No preference", command=noPrefCallback)
        results_incomparable_button = Button(text="Incomparable", command=resultsIncomparableCallback)

        # Add buttons to canvas
        prefer_1_button_window = canvas.create_window(30, winHeight - 30, anchor='n', window=prefer_1_button)
        prefer_2_button_window = canvas.create_window(80, winHeight - 30, anchor='n', window=prefer_2_button)    
        no_preference_button_window = canvas.create_window(150, winHeight - 30, anchor='n', window=no_preference_button)
        results_incomparable_button_window = canvas.create_window(240, winHeight - 30, anchor='n', window=results_incomparable_button)  

        # Player 1
        videoplayer = TkinterVideo(master=win, scaled=True)
        videoplayer.load(video1)
        videoplayer.pack(expand=True, fill="both")

        # Player 2
        videoplayer2 = TkinterVideo(master=win, scaled=True)
        videoplayer2.load(video2)
        videoplayer2.pack(expand=True, fill="both")

        video1Canvas = canvas.create_window( 0, 10, anchor='nw', window=videoplayer, height=winHeight - 50, width=winWidth/2 - 5)  
        video2Canvas = canvas.create_window(605, 10, anchor='nw', window=videoplayer2, height=winHeight - 50, width=winWidth/2 - 5)

        videoplayer.play()
        videoplayer2.play()

        win.mainloop()
