import os
import gym
import random
import numpy as np
import tensorflow as tf
from skimage.color import rgb2gray
from skimage.transform import resize

ENV_NAME = 'Breakout-v0'  # Environment name
BATCHSIZE = 1024
WIDTH = 84  # Resized frame width
HEIGHT = 84  # Resized frame height
NUM_EPISODES = 12000  # Number of episodes the agent plays
STATE_LENGTH = 4  # Number of most recent frames to produce the input to the network
LR = 1e-3                   # learning rate
# MOMENTUM = 0.95  # Momentum used by RMSProp
# MIN_GRAD = 0.01  # Constant added to the squared gradient in the denominator of the RMSProp update
GAMMA = 0.99                 # reward discount
NO_OP_STEPS = 30  # Maximum number of "do nothing" actions to be performed by the agent at the start of an episode
TRAIN = True
LOAD_NETWORK = False
SAVE_INTERVAL = 300000  # The frequency with which the network is saved
SAVE_NETWORK_PATH = 'saved_networks/PG_' + ENV_NAME
SAVE_SUMMARY_PATH = 'summary/PG_' + ENV_NAME
NUM_EPISODES_AT_TEST = 10  # Number of episodes the agent plays at test time


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            learning_rate=LR,
            reward_decay=GAMMA,
    ):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = reward_decay

        self.total_reward = 0
        self.total_loss = 0
        self.time = 0
        self.episode = 0

        self.memory = [[], [], []]

        self._build_net()

        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver()
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(SAVE_SUMMARY_PATH, self.sess.graph)

        if not os.path.exists(SAVE_NETWORK_PATH):
            os.makedirs(SAVE_NETWORK_PATH)

        self.sess.run(tf.global_variables_initializer())

        # Load network
        if LOAD_NETWORK:
            self.load_network()

    def _build_net(self):
        # build evaluate_net
        self.s = tf.placeholder(tf.float32, [None, STATE_LENGTH, WIDTH, HEIGHT], name='s') 
        self.a = tf.placeholder(tf.int64, [None])
        self.r = tf.placeholder(tf.float32, [None])

        x = None
        with tf.variable_scope('policy_gradient'):
            x = tf.layers.conv2d(self.s, 32, 8, (4, 4), activation=tf.nn.relu, data_format='channels_first')
            x = tf.layers.conv2d(x, 64, 4, (2, 2), activation=tf.nn.relu, data_format='channels_first')
            x = tf.layers.conv2d(x, 64, 3, (1, 1), activation=tf.nn.relu, data_format='channels_first')
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, 512, activation=tf.nn.relu)
            x = tf.layers.dense(x, self.n_actions)
        
        self.act_prob = tf.nn.softmax(x)

        with tf.variable_scope('loss'):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x, labels=self.a)
            self.loss = tf.reduce_mean(neg_log_prob * self.r)
        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def choose_action(self, state):
        act_prob = self.act_prob.eval(feed_dict={self.s: [np.float32(state / 255.0)]})
        action = np.random.choice(range(act_prob.shape[1]), p=act_prob.ravel())
        return action

    def get_initial_state(self, observation, last_observation):
        processed_observation = np.maximum(observation, last_observation)
        processed_observation = np.uint8(resize(rgb2gray(processed_observation), (WIDTH, HEIGHT)) * 255)
        state = [processed_observation for _ in range(STATE_LENGTH)]
        return np.stack(state, axis=0)

    def learn(self, state, action, reward, done, observation):
        next_state = np.append(state[1:, :, :], observation, axis=0)
        reward = np.clip(reward, -1, 1)

        self.memory[0].append(state)
        self.memory[1].append(action)
        self.memory[2].append(reward)

        self.total_reward += reward
        self.time += 1

        if done:
            if self.episode % SAVE_INTERVAL == 0:
                save_path = self.saver.save(self.sess, SAVE_NETWORK_PATH + '/' + ENV_NAME, global_step=self.episode)
                print('Successfully saved: ' + save_path)

            discount_norm_r = self._discount_and_norm_rewards()

            if len(self.memory[2]) < BATCHSIZE:
                loss, _ = self.sess.run([self.loss, self._train_op], feed_dict={
                    self.s: np.float32(np.array(self.memory[0]) / 255.0),
                    self.a: self.memory[1],
                    self.r: self.memory[2]
                })
                self.total_loss = loss
            else:
                index = np.random.choice(len(self.memory[2]), size=BATCHSIZE)
                s = np.array(self.memory[0])[index]
                a = np.array(self.memory[1])[index]
                r = np.array(self.memory[2])[index]

                loss, _ = self.sess.run([self.loss, self._train_op], feed_dict={
                    self.s: np.float32(s / 255.0),
                    self.a: a,
                    self.r: r
                })
                self.total_loss = loss

            stats = [self.total_reward, self.time, self.total_loss]
            for i in range(len(stats)):
                self.sess.run(self.update_ops[i], feed_dict={
                    self.summary_placeholders[i]: float(stats[i])})
            summary_str = self.sess.run(self.summary_op)
            self.summary_writer.add_summary(summary_str, self.episode + 1)

            print('EPISODE: {:d} \t TIME: {:d} \t TOTAL_REWARD: {:.2f} \t AVG_LOSS: {:.5f}'.format(
                self.episode + 1, self.time, float(self.total_reward), self.total_loss))

            self.total_reward = 0
            self.total_loss = 0
            self.time = 0
            self.episode += 1

            self.memory = [[], [], []]

        return next_state

    def _discount_and_norm_rewards(self):
        discount_norm_r = np.zeros_like(self.memory[2])
        discount_r = 0
        for i in range(len(self.memory[2])-1, -1, -1):
            discount_r = self.memory[2][i] + discount_r*self.gamma
            discount_norm_r[i] = discount_r

        discount_norm_r -= np.mean(discount_norm_r)
        if np.std(discount_norm_r) != 0:
            discount_norm_r /= np.std(discount_norm_r)
        return discount_norm_r

    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Total Reward/Episode', episode_total_reward)
        episode_duration = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Duration/Episode', episode_duration)
        episode_avg_loss = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Average Loss/Episode', episode_avg_loss)
        summary_vars = [episode_total_reward, episode_duration, episode_avg_loss]
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
    agent = PolicyGradient(n_actions=env.action_space.n)

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
                observation, _, done, _ = env.step(action)
                env.render()
                processed_observation = preprocess(observation, last_observation)
                state = np.append(state[1:, :, :], processed_observation, axis=0)

if __name__ == '__main__':
    main()