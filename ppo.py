import os
import gym
import random
import numpy as np
import tensorflow as tf
from skimage.color import rgb2gray
from skimage.transform import resize
from collections import deque

ENV_NAME = 'Breakout-v0'  # Environment name
BATCHSIZE = 32
WIDTH = 84  # Resized frame width
HEIGHT = 84  # Resized frame height
NUM_EPISODES = 12000  # Number of episodes the agent plays
STATE_LENGTH = 4  # Number of most recent frames to produce the input to the network
A_LR = 2.5e-4
C_LR = 5e-4
EPSILON = 0.2
UPDATE_STEPS = 10
TIME_HORIZON = 128
GAMMA = 0.99                 # reward discount
NO_OP_STEPS = 30  # Maximum number of "do nothing" actions to be performed by the agent at the start of an episode
START_LEARNING = 20000
MEMORY_CAPACITY = 400000
TRAIN_INTERVAL = 4  # The agent selects 4 actions between successive updates
TRAIN = True
LOAD_NETWORK = False
SAVE_INTERVAL = 1000  # The frequency with which the network is saved
SAVE_NETWORK_PATH = 'saved_networks/PPO_' + ENV_NAME
SAVE_SUMMARY_PATH = 'summary/PPO_' + ENV_NAME
NUM_EPISODES_AT_TEST = 10  # Number of episodes the agent plays at test time
DATAFORMAT = 'channels_first'


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            reward_decay=GAMMA,
    ):
        self.n_actions = n_actions
        self.gamma = reward_decay

        self.memory = deque()

        self.learn_step_counter = 0

        self.total_reward = 0
        self.total_loss = 0
        self.time = 0
        self.episode = 0

        self.s = tf.placeholder(tf.float32, [None, STATE_LENGTH, WIDTH, HEIGHT], name='s')

        self.v = self._build_cnet("critic")
        self.r = tf.placeholder(tf.float32, [None, 1])
        with tf.variable_scope('closs'):
            self.advantage = self.r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
        with tf.variable_scope('ctrain'):
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        self.pi = self._build_anet("pi")
        self.pi_dist = tf.distributions.Categorical(probs=self.pi)
        self.action = self.pi_dist.sample(1)[0]
        self.oldpi = self._build_anet("oldpi", trainable=False)
        self.oldpi_dist = tf.distributions.Categorical(probs=self.oldpi)

        pi_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="pi")
        oldpi_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="oldpi")
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [tf.assign(oldp, p) for oldp, p in zip(oldpi_params, pi_params)]

        self.a = tf.placeholder(tf.int64, [None])
        self.adv = tf.placeholder(tf.float32, [None, 1])
        with tf.variable_scope('aloss'):
            # a_one_hot = tf.one_hot(self.a, self.n_actions, 1.0, 0.0)
            # pi_prob = tf.reduce_sum(tf.multiply(self.pi, a_one_hot), axis=1)
            # oldpi_prob = tf.stop_gradient(tf.reduce_sum(tf.multiply(self.oldpi, a_one_hot), axis=1))
            # ratio = tf.div(pi_prob, oldpi_prob)
            log_pi_prob = self.pi_dist.log_prob(self.a)
            log_oldpi_prob = tf.stop_gradient(self.oldpi_dist.log_prob(self.a))
            ratio = tf.exp(log_pi_prob - log_oldpi_prob)
            ratio = tf.reshape(ratio, [-1, 1])
            surr = ratio * self.adv
            self.aloss = -tf.reduce_mean(tf.minimum(surr,
                tf.clip_by_value(ratio, 1.-EPSILON, 1.+EPSILON)*self.adv))

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

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

    def _build_anet(self, name, trainable=True):
        x = self.s
        if DATAFORMAT == "channels_last":
            x = tf.transpose(x, [0, 2, 3, 1])
        with tf.variable_scope(name):
            x = tf.layers.conv2d(x, 32, 8, (4, 4), activation=tf.nn.relu, data_format=DATAFORMAT, trainable=trainable)
            x = tf.layers.conv2d(x, 64, 4, (2, 2), activation=tf.nn.relu, data_format=DATAFORMAT, trainable=trainable)
            x = tf.layers.conv2d(x, 64, 3, (1, 1), activation=tf.nn.relu, data_format=DATAFORMAT, trainable=trainable)
            if DATAFORMAT == "channels_last":
                x = tf.transpose(x, [0, 3, 1, 2])
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, 512, activation=tf.nn.relu, trainable=trainable)
            x = tf.layers.dense(x, self.n_actions, trainable=trainable)

        act_prob = tf.nn.softmax(x)
        return act_prob

    def _build_cnet(self, name):
        x = self.s
        if DATAFORMAT == "channels_last":
            x = tf.transpose(x, [0, 2, 3, 1])
        with tf.variable_scope(name):
            x = tf.layers.conv2d(x, 32, 8, (4, 4), activation=tf.nn.relu, data_format=DATAFORMAT)
            x = tf.layers.conv2d(x, 64, 4, (2, 2), activation=tf.nn.relu, data_format=DATAFORMAT)
            x = tf.layers.conv2d(x, 64, 3, (1, 1), activation=tf.nn.relu, data_format=DATAFORMAT)
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, 512, activation=tf.nn.relu)
            x = tf.layers.dense(x, 1)

        return x

    def choose_action(self, state):
        # act_prob = self.pi.eval(feed_dict={self.s: [np.float32(state / 255.0)]})
        # action = np.random.choice(range(act_prob.shape[1]), p=act_prob.ravel())
        action = self.action.eval(feed_dict={self.s: [np.float32(state)]})[0]
        return action

    def get_initial_state(self, observation, last_observation):
        processed_observation = np.maximum(observation, last_observation)
        processed_observation = np.array((resize(rgb2gray(processed_observation), (WIDTH, HEIGHT)) * 255)/255, dtype=np.float32)
        state = [processed_observation for _ in range(STATE_LENGTH)]
        return np.stack(state, axis=0)

    def train_network(self):
        self.sess.run(self.update_oldpi_op)

        for i in range(UPDATE_STEPS):
            state_batch = []
            action_batch = []
            reward_batch = []
            next_state_batch = []
            terminal_batch = []
            y_batch = []

            minibatch = random.sample(self.memory, BATCHSIZE)
            for data in minibatch:
                state_batch.append(data[0])
                action_batch.append(data[1])
                reward_batch.append(data[2])
                next_state_batch.append(data[3])
                terminal_batch.append(data[4])

            terminal_batch = np.array(terminal_batch) + 0

            target_values_batch = self.v.eval(feed_dict={self.s: np.float32(next_state_batch)})
            y_batch = reward_batch + (1 - terminal_batch) * GAMMA * np.reshape(target_values_batch, [-1])

            state_batch = np.float32(state_batch)
            y_batch = np.array(y_batch)[:, np.newaxis]

            adv = self.advantage.eval(feed_dict={
                self.s: state_batch,
                self.r: y_batch})

            aloss, _ =  self.sess.run([self.aloss, self.atrain_op], feed_dict={
                self.s: state_batch,
                self.a: action_batch,
                self.adv: adv})
            self.total_loss += aloss

            closs, _ =  self.sess.run([self.closs, self.ctrain_op], feed_dict={
                self.s: state_batch,
                self.r: y_batch})
            self.total_loss += closs

    def learn(self, state, action, reward, done, observation):
        next_state = np.append(state[1:, :, :], observation, axis=0)
        reward = np.clip(reward, -1, 1)

        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > MEMORY_CAPACITY:
            self.memory.popleft()

        self.total_reward += reward
        self.time += 1

        self.learn_step_counter += 1

        if self.learn_step_counter >= START_LEARNING:
            if self.learn_step_counter % TRAIN_INTERVAL == 0:
                self.train_network()

        if done:
            if self.episode % SAVE_INTERVAL == 0:
                save_path = self.saver.save(self.sess, SAVE_NETWORK_PATH + '/' + ENV_NAME, global_step=self.episode)
                print('Successfully saved: ' + save_path)

            self.total_loss /= self.time
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

        return next_state

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
    processed_observation = np.array((resize(rgb2gray(processed_observation), (WIDTH, HEIGHT)) * 255)/255, dtype=np.float32)
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