import os
import gym
import random
import numpy as np
import tensorflow as tf
from skimage.color import rgb2gray
from skimage.transform import resize
from collections import deque

ENV_NAME = 'Breakout-v0'  # Environment name
BATCH_SIZE = 32
WIDTH = 84  # Resized frame width
HEIGHT = 84  # Resized frame height
NUM_EPISODES = 12000  # Number of episodes the agent plays
STATE_LENGTH = 4  # Number of most recent frames to produce the input to the network
LR = 0.00025                   # learning rate
MOMENTUM = 0.95  # Momentum used by RMSProp
MIN_GRAD = 0.01  # Constant added to the squared gradient in the denominator of the RMSProp update
EXPLORATION_STEPS = 1000000  # Number of steps over which the initial value of epsilon is linearly annealed to its final value
INITIAL_EPSILON = 1.0  # Initial value of epsilon in epsilon-greedy
FINAL_EPSILON = 0.1  # Final value of epsilon in epsilon-greedy
GAMMA = 0.99                 # reward discount
TARGET_REPLACE_ITER = 10000   # target update frequency
TRAIN_INTERVAL = 4  # The agent selects 4 actions between successive updates
START_LEARNING = 20000
MEMORY_CAPACITY = 400000
NO_OP_STEPS = 30  # Maximum number of "do nothing" actions to be performed by the agent at the start of an episode
TRAIN = False
LOAD_NETWORK = True
SAVE_INTERVAL = 300000  # The frequency with which the network is saved
DDQN = True
MODEL_NAME = 'DDQN-' if DDQN else 'DQN-'
SAVE_NETWORK_PATH = 'saved_networks/' + MODEL_NAME + ENV_NAME
SAVE_SUMMARY_PATH = 'summary/' + MODEL_NAME + ENV_NAME
NUM_EPISODES_AT_TEST = 1  # Number of episodes the agent plays at test time
DATAFORMAT = 'channels_first'


class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            learning_rate=LR,
            reward_decay=GAMMA,
            replace_target_iter=TARGET_REPLACE_ITER,
            memory_size=MEMORY_CAPACITY,
            batch_size=BATCH_SIZE,
    ):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon = INITIAL_EPSILON
        self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS

        # total learning step
        self.learn_step_counter = 0

        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.time = 0
        self.episode = 0

        self.memory = deque()

        self._build_net()
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_net")
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="eval_net")
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver(e_params)
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        if TRAIN:
            self.summary_writer = tf.summary.FileWriter(SAVE_SUMMARY_PATH, self.sess.graph)

        if not os.path.exists(SAVE_NETWORK_PATH):
            os.makedirs(SAVE_NETWORK_PATH)

        self.sess.run(tf.global_variables_initializer())

        # Load network
        if LOAD_NETWORK:
            self.load_network()

        self.sess.run(self.replace_target_op)

    def _build_net(self):
        # build evaluate_net
        self.s = tf.placeholder(tf.float32, [None, STATE_LENGTH, WIDTH, HEIGHT], name='s')
        x = self.s
        if DATAFORMAT == "channels_last":
            x = tf.transpose(x, [0, 2, 3, 1])
        self.a = tf.placeholder(tf.int64, [None])
        self.y = tf.placeholder(tf.float32, [None])

        with tf.variable_scope('eval_net'):
            x = tf.layers.conv2d(x, 32, 8, (4, 4), activation=tf.nn.relu, data_format=DATAFORMAT)
            x = tf.layers.conv2d(x, 64, 4, (2, 2), activation=tf.nn.relu, data_format=DATAFORMAT)
            x = tf.layers.conv2d(x, 64, 3, (1, 1), activation=tf.nn.relu, data_format=DATAFORMAT)
            if DATAFORMAT == "channels_last":
                x = tf.transpose(x, [0, 3, 1, 2])
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, 512, activation=tf.nn.relu)
            self.q_eval = tf.layers.dense(x, self.n_actions)

        a_one_hot = tf.one_hot(self.a, self.n_actions, 1.0, 0.0)
        q_value = tf.reduce_sum(tf.multiply(self.q_eval, a_one_hot), reduction_indices=1)
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.y, q_value))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr, momentum=MOMENTUM, epsilon=MIN_GRAD).minimize(self.loss)

        # build target_net
        self.st = tf.placeholder(tf.float32, [None, STATE_LENGTH, WIDTH, HEIGHT], name='st')    # input
        x = self.st
        if DATAFORMAT == "channels_last":
            x = tf.transpose(x, [0, 2, 3, 1])
        with tf.variable_scope('target_net'):
            x = tf.layers.conv2d(x, 32, 8, (4, 4), activation=tf.nn.relu, data_format=DATAFORMAT)
            x = tf.layers.conv2d(x, 64, 4, (2, 2), activation=tf.nn.relu, data_format=DATAFORMAT)
            x = tf.layers.conv2d(x, 64, 3, (1, 1), activation=tf.nn.relu, data_format=DATAFORMAT)
            if DATAFORMAT == "channels_last":
                x = tf.transpose(x, [0, 3, 1, 2])
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, 512, activation=tf.nn.relu)
            self.target_q_eval = tf.layers.dense(x, self.n_actions)

    def choose_action(self, state):
        if not TRAIN:
            actions_value = self.q_eval.eval(feed_dict={self.s: [np.float32(state / 255.0)]})
            action = np.argmax(actions_value)
            return action

        if np.random.uniform() > self.epsilon and self.learn_step_counter >= START_LEARNING:
            actions_value = self.q_eval.eval(feed_dict={self.s: [np.float32(state / 255.0)]})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)

        if self.epsilon > FINAL_EPSILON and self.learn_step_counter >= START_LEARNING:
            self.epsilon -= self.epsilon_step
        return action

    def get_initial_state(self, observation, last_observation):
        processed_observation = np.maximum(observation, last_observation)
        processed_observation = np.uint8(resize(rgb2gray(processed_observation), (WIDTH, HEIGHT)) * 255)
        state = [processed_observation for _ in range(STATE_LENGTH)]
        return np.stack(state, axis=0)

    def train_network(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []
        y_batch = []

        minibatch = random.sample(self.memory, BATCH_SIZE)
        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            terminal_batch.append(data[4])

        terminal_batch = np.array(terminal_batch) + 0

        target_q_values_batch = self.target_q_eval.eval(feed_dict={self.st: np.float32(np.array(next_state_batch) / 255.0)})
        if DDQN:
            next_action_batch = np.argmax(self.q_eval.eval(feed_dict={self.s: np.float32(np.array(next_state_batch) / 255.0)}), axis=1)
            for i in range(BATCH_SIZE):
                y_batch.append(reward_batch[i] + (1 - terminal_batch[i]) * GAMMA * target_q_values_batch[i][next_action_batch[i]])
        else:
            y_batch = reward_batch + (1 - terminal_batch) * GAMMA * np.max(target_q_values_batch, axis=1)

        loss, _ = self.sess.run([self.loss, self._train_op], feed_dict={
            self.s: np.float32(np.array(state_batch) / 255.0),
            self.a: action_batch,
            self.y: y_batch
        })

        self.total_loss += loss

    def learn(self, state, action, reward, done, observation):
        next_state = np.append(state[1:, :, :], observation, axis=0)
        reward = np.clip(reward, -1, 1)

        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > MEMORY_CAPACITY:
            self.memory.popleft()

        if self.learn_step_counter >= START_LEARNING:
            if self.learn_step_counter % TRAIN_INTERVAL == 0:
                self.train_network()

            if self.learn_step_counter % self.replace_target_iter == 0:
                self.sess.run(self.replace_target_op)
                print('\ntarget_params_replaced\n')

            if self.learn_step_counter % SAVE_INTERVAL == 0:
                save_path = self.saver.save(self.sess, SAVE_NETWORK_PATH + '/' + ENV_NAME, global_step=self.learn_step_counter)
                print('Successfully saved: ' + save_path)

        self.total_reward += reward
        self.total_q_max += np.max(self.q_eval.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))
        self.time += 1

        if done:
            if self.learn_step_counter >= START_LEARNING:
                stats = [self.total_reward, self.total_q_max / float(self.time),
                        self.time, self.total_loss / (float(self.time) / float(TRAIN_INTERVAL))]
                for i in range(len(stats)):
                    self.sess.run(self.update_ops[i], feed_dict={
                        self.summary_placeholders[i]: float(stats[i])
                    })
                summary_str = self.sess.run(self.summary_op)
                self.summary_writer.add_summary(summary_str, self.episode + 1)

            print('EPISODE: {0:6d} \t STEP: {1:8d} \t TIME: {2:5d} \t EPSILON: {3:.4f}\nTOTAL_REWARD: {4:3.0f} \t AVG_MAX_Q: {5:2.4f} \t AVG_LOSS: {6:.5f}'.format(
                self.episode + 1, self.learn_step_counter, self.time, self.epsilon, self.total_reward,
                self.total_q_max / float(self.time), self.total_loss / (float(self.time) / float(TRAIN_INTERVAL))))

            self.total_reward = 0
            self.total_q_max = 0
            self.total_loss = 0
            self.time = 0
            self.episode += 1

        self.learn_step_counter += 1

        return next_state

    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Total Reward/Episode', episode_total_reward)
        episode_avg_max_q = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Average Max Q/Episode', episode_avg_max_q)
        episode_duration = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Duration/Episode', episode_duration)
        episode_avg_loss = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Average Loss/Episode', episode_avg_loss)
        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

    def load_network(self):
        checkpoint = tf.train.get_checkpoint_state(SAVE_NETWORK_PATH)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
        else:
            print('Training new network...')


def preprocess(observation, last_observation):
    processed_observation = np.maximum(observation, last_observation)
    processed_observation = np.uint8(resize(rgb2gray(processed_observation), (WIDTH, HEIGHT)) * 255)
    return np.reshape(processed_observation, (1, WIDTH, HEIGHT))


def main():
    env = gym.make(ENV_NAME)
    # env = gym.wrappers.Monitor(env, 'vedio/'+MODEL_NAME+ENV_NAME, force = True)
    agent = DeepQNetwork(n_actions=env.action_space.n)

    if TRAIN:
        for _ in range(NUM_EPISODES):
            done = False
            observation = env.reset()
            for _ in range(random.randint(1, NO_OP_STEPS)):
                last_observation = observation
                observation, _, _, _ = env.step(0)
            state = agent.get_initial_state(observation, last_observation)
            while not done:
                last_observation = observation
                action = agent.choose_action(state)
                observation, reward, done, _ = env.step(action)
                # env.render()
                processed_observation = preprocess(observation, last_observation)
                state = agent.learn(state, action, reward, done, processed_observation)
    else:
        total_reward = 0
        for _ in range(NUM_EPISODES_AT_TEST):
            done = False
            observation = env.reset()
            for _ in range(random.randint(1, NO_OP_STEPS)):
                last_observation = observation
                observation, _, _, _ = env.step(0)  # Do nothing
            state = agent.get_initial_state(observation, last_observation)
            while not done:
                last_observation = observation
                action = agent.choose_action(state)
                observation, reward, done, _ = env.step(action)
                total_reward += reward
                env.render()
                processed_observation = preprocess(observation, last_observation)
                state = np.append(state[1:, :, :], processed_observation, axis=0)
        total_reward = total_reward / 10
        print("Mean total reward: " + str(total_reward))
    env.close()

if __name__ == '__main__':
    main()