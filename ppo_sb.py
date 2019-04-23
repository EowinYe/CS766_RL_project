import gym

from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines import PPO1

ENV_NAME = 'Breakout-v0'
SAVE_NETWORK_PATH = 'saved_networks/PPO_stable_baselines/PPO1_' + ENV_NAME
TRAIN = True

env = make_atari_env(ENV_NAME, num_env=4, seed=0)
env = VecFrameStack(env, n_stack=4)

if TRAIN:
    model = PPO1(CnnPolicy, env, verbose=1)
    model.learn(total_timesteps=1000000)
    model.save(SAVE_NETWORK_PATH)

    del model
else:
    model = PPO1.load(SAVE_NETWORK_PATH)
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        if done:
            break