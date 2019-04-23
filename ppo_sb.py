import gym

from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO1

ENV_NAME = 'Breakout-v0'
TRAIN = True

env = gym.make(ENV_NAME)
env = DummyVecEnv([lambda: env])

if TRAIN:
    model = PPO1(CnnPolicy, env, verbose=1)
    model.learn(total_timesteps=1000000)
    model.save("PPO1_"+ENV_NAME)

    del model
else:
    model = PPO1.load("PPO1_"+ENV_NAME)
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        if done:
            break