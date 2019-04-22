import gym

from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO1

ENV_NAME = 'Breakout-v0'

env = gym.make(ENV_NAME)
env = DummyVecEnv([lambda: env])

model = PPO1(CnnPolicy, env, verbose=1)
model.learn(total_timesteps=1000000)
model.save("PPO1_"+ENV_NAME)

del model