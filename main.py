import sys
import gymnasium as gym
sys.modules["gym"] = gym
from stable_baselines3 import A2C
from video import *



def main():

   # result = HumanFeedback.requestHumanFeedbackFromVideos("./videos/test.mp4", "./videos/test.mp4", 1200, 500)
   # print(result)
    # do something if the user finds the videos incomparable
   env = gym.make("CartPole-v1")
   model = A2C("MlpPolicy", env)
   model.learn(total_timesteps=int(1000))

   model.save("a2c_lunar")

   del model
   model = A2C.load("a2c_lunar")

   render_env = gym.make("CartPole-v1", render_mode="rgb_array")
   VideoMaker.getVideoFromEnvAndModel(render_env, model, 30, 1000, 600, 400, "test.mp4")

   env.close()
   render_env.close()

if __name__ == "__main__":
    main()