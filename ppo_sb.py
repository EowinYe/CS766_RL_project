import gym

from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines import PPO2

ENV_NAME = 'BreakoutNoFrameskip-v4'
SAVE_NETWORK_PATH = 'saved_networks/PPO_stable_baselines/PPO2_' + ENV_NAME
SAVE_SUMMARY_PATH = 'summary/PPO_stable_baselines/PPO2_' + ENV_NAME
TRAIN = True

env = make_atari_env(ENV_NAME, num_env=4, seed=0)
env = VecFrameStack(env, n_stack=4)

if TRAIN:
    model = PPO2(CnnPolicy, env, verbose=1)
    model.learn(total_timesteps=120000000, tensorboard_log=SAVE_SUMMARY_PATH)
    model.save(SAVE_NETWORK_PATH)

    del model
else:
    model = PPO2.load(SAVE_NETWORK_PATH)
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        if done:
            break