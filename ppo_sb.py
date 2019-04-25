import gym
import imageio
import numpy as np

from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecVideoRecorder, VecFrameStack, DummyVecEnv
from stable_baselines import PPO2

ENV_NAME = 'MsPacmanNoFrameskip-v4'
SAVE_NETWORK_PATH = 'saved_networks/PPO_stable_baselines/PPO2_' + ENV_NAME
SAVE_SUMMARY_PATH = 'summary/PPO_stable_baselines/PPO2_' + ENV_NAME
VIDEO_FOLDER = 'video/stable_baselines/'
VIDEO_LENGTH = 2000
TRAIN = False

num_env = 1
if TRAIN:
    num_env = 4
env = make_atari_env(ENV_NAME, num_env=num_env, seed=0)
env = VecFrameStack(env, n_stack=4)

if TRAIN:
    model = PPO2(CnnPolicy, env, verbose=1, tensorboard_log=SAVE_SUMMARY_PATH)
    model.learn(total_timesteps=20000000)
    model.save(SAVE_NETWORK_PATH)

    del model
else:
    obs = env.reset()
    env = VecVideoRecorder(env, VIDEO_FOLDER,
                           record_video_trigger=lambda x: x == 0, video_length=VIDEO_LENGTH,
                           name_prefix="PPO_"+ENV_NAME)
    env.reset()
    model = PPO2.load(SAVE_NETWORK_PATH)
    cnt = 0
    for i in range(VIDEO_LENGTH):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        cnt += int(done)
        if cnt==5:
            break
env.close()
