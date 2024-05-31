#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, sys, copy, argparse, random
import matplotlib.pyplot as plt
import math
import os

np.random.seed(10701)
tf.set_random_seed(10701)
random.seed(10701)

class QNetwork():

    def __init__(self, environment_name, networkname, trianable):
        if environment_name == 'grid':
            self.nObservation = 12
            self.nAction = 6
            self.learning_rate = 0.0001
            self.architecture = [32, 64, 32]

        kernel_init = tf.random_uniform_initializer(-0.5, 0.5)
        bias_init = tf.constant_initializer(0)
        self.input = tf.placeholder(tf.float32, shape=[None, self.nObservation], name='input')
        with tf.variable_scope(networkname):
            layer1 = tf.layers.dense(self.input, self.architecture[0], tf.nn.relu, kernel_initializer=kernel_init, bias_initializer=bias_init, name='layer1', trainable=trianable)
            layer2 = tf.layers.dense(layer1, self.architecture[1], tf.nn.relu, kernel_initializer=kernel_init, bias_initializer=bias_init, name='layer2', trainable=trianable)
            layer3 = tf.layers.dense(layer2, self.architecture[2], tf.nn.relu, kernel_initializer=kernel_init, bias_initializer=bias_init, name='layer3', trainable=trianable)
            self.output = tf.layers.dense(layer3, self.nAction, kernel_initializer=kernel_init, bias_initializer=bias_init, name='output', trainable=trianable)

        self.targetQ = tf.placeholder(tf.float32, shape=[None, self.nAction], name='target')
        if trianable == True:
            self.loss = tf.losses.mean_squared_error(self.targetQ, self.output)
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        with tf.variable_scope(networkname, reuse=True):
            self.w1 = tf.get_variable('layer1/kernel')
            self.b1 = tf.get_variable('layer1/bias')
            self.w2 = tf.get_variable('layer2/kernel')
            self.b2 = tf.get_variable('layer2/bias')
            self.w3 = tf.get_variable('layer3/kernel')
            self.b3 = tf.get_variable('layer3/bias')
            self.w4 = tf.get_variable('output/kernel')
            self.b4 = tf.get_variable('output/bias')


class Replay_Memory():

    def __init__(self, memory_size=50000, burn_in=10000):
        self.memory = []
        self.is_burn_in = False
        self.memory_max = memory_size
        self.burn_in = burn_in

    def sample_batch(self, batch_size=32):
        index = np.random.randint(len(self.memory), size=batch_size)
        batch = [self.memory[i] for i in index]
        return batch

    def append(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.memory_max:
            self.memory.pop(0)


class DoubleDQN_Agent():

    def __init__(self, environment_name, sess, gridgraph, render=False):
        self.epsilon = 0.05

        if environment_name == 'grid':
            self.gamma = 0.95
        self.max_episodes = 10000
        self.batch_size = 32
        self.render = render

        self.qNetwork = QNetwork(environment_name, 'q', trianable=True)
        self.tNetwork = QNetwork(environment_name, 't', trianable=False)
        self.replay = Replay_Memory()

        self.gridgraph = gridgraph

        self.as_w1 = tf.assign(self.tNetwork.w1, self.qNetwork.w1)
        self.as_b1 = tf.assign(self.tNetwork.b1, self.qNetwork.b1)
        self.as_w2 = tf.assign(self.tNetwork.w2, self.qNetwork.w2)
        self.as_b2 = tf.assign(self.tNetwork.b2, self.qNetwork.b2)
        self.as_w3 = tf.assign(self.tNetwork.w3, self.qNetwork.w3)
        self.as_b3 = tf.assign(self.tNetwork.b3, self.qNetwork.b3)
        self.as_w4 = tf.assign(self.tNetwork.w4, self.qNetwork.w4)
        self.as_b4 = tf.assign(self.tNetwork.b4, self.qNetwork.b4)

        self.init = tf.global_variables_initializer()

        self.sess = sess
        self.sess.run(self.init)
        self.saver = tf.train.Saver(max_to_keep=20)

    def epsilon_greedy_policy(self, q_values):
        rnd = np.random.rand()
        if rnd <= self.epsilon:
            return np.random.randint(len(q_values))
        else:
            return np.argmax(q_values)

    def greedy_policy(self, q_values):
        return np.argmax(q_values)

    def network_assign(self):
        self.sess.run([self.as_w1, self.as_b1, self.as_w2, self.as_b2, self.as_w3, self.as_b3, self.as_w4, self.as_b4])

    def train(self, twoPinNum, twoPinNumEachNet, netSort, savepath, model_file=None):
        if model_file is not None:
            self.saver.restore(self.sess, model_file)

        reward_log = []
        test_reward_log = []
        test_episode = []
        solution_combo = []
        reward_plot_combo = []
        reward_plot_combo_pure = []

        for episode in np.arange(self.max_episodes * len(self.gridgraph.twopin_combo)):
            solution_combo.append(self.gridgraph.route)

            state, reward_plot, is_best = self.gridgraph.reset()
            reward_plot_pure = reward_plot - self.gridgraph.posTwoPinNum * 100

            if (episode) % twoPinNum == 0:
                reward_plot_combo.append(reward_plot)
                reward_plot_combo_pure.append(reward_plot_pure)
            is_terminal = False
            rewardi = 0.0
            if episode % 100 == 0:
                self.network_assign()

            rewardfortwopin = 0
            while not is_terminal:
                observation = self.gridgraph.state2obsv()
                q_values = self.sess.run(self.qNetwork.output, feed_dict={self.qNetwork.input: observation})
                action = self.epsilon_greedy_policy(q_values)
                nextstate, reward, is_terminal, debug = self.gridgraph.step(action)
                observation_next = self.gridgraph.state2obsv()
                self.replay.append([observation, action, reward, observation_next, is_terminal])
                state = nextstate
                rewardi = rewardi + reward
                rewardfortwopin = rewardfortwopin + reward

                batch = self.replay.sample_batch(self.batch_size)
                batch_observation = np.squeeze(np.array([trans[0] for trans in batch]))
                batch_action = np.array([trans[1] for trans in batch])
                batch_reward = np.array([trans[2] for trans in batch])
                batch_observation_next = np.squeeze(np.array([trans[3] for trans in batch]))
                batch_is_terminal = np.array([trans[4] for trans in batch])

                # Double DQN update
                q_batch_next = self.sess.run(self.qNetwork.output, feed_dict={self.qNetwork.input: batch_observation_next})
                q_batch_next_target = self.sess.run(self.tNetwork.output, feed_dict={self.tNetwork.input: batch_observation_next})
                best_actions = np.argmax(q_batch_next, axis=1)
                y_batch = batch_reward + self.gamma * (1 - batch_is_terminal) * q_batch_next_target[np.arange(self.batch_size), best_actions]

                q_batch = self.sess.run(self.qNetwork.output, feed_dict={self.qNetwork.input: batch_observation})
                targetQ = q_batch.copy()
                targetQ[np.arange(self.batch_size), batch_action] = y_batch

                _, train_error = self.sess.run([self.qNetwork.opt, self.qNetwork.loss], feed_dict={self.qNetwork.input: batch_observation, self.qNetwork.targetQ: targetQ})

            reward_log.append(rewardi)

            self.gridgraph.instantrewardcombo.append(rewardfortwopin)

        score = self.gridgraph.best_reward
        solution = self.gridgraph.best_route[-twoPinNum:]

        solutionDRL = []

        for i in range(len(netSort)):
            solutionDRL.append([])

        if self.gridgraph.posTwoPinNum == twoPinNum:
            dumpPointer = 0
            for i in range(len(netSort)):
                netToDump = netSort[i]
                for j in range(twoPinNumEachNet[netToDump]):
                    solutionDRL[netToDump].append(solution[dumpPointer])
                    dumpPointer = dumpPointer + 1
        else:
            solutionDRL = solution

        self.sess.close()
        tf.reset_default_graph()

        return solutionDRL, reward_plot_combo, reward_plot_combo_pure, solution, self.gridgraph.posTwoPinNum

    def test(self, model_file=None, no=20, stat=False):
        if model_file is not None:
            self.saver.restore(self.sess, model_file)
        reward_list = []
        cum_reward = 0.0
        for episode in np.arange(no):
            episode_reward = 0.0
            state = self.gridgraph.reset()
            is_terminal = False
            while not is_terminal:
                observation = self.gridgraph.state2obsv()
                q_values = self.sess.run(self.qNetwork.output, feed_dict={self.qNetwork.input: observation})
                action = self.greedy_policy(q_values)
                nextstate, reward, is_terminal, debug = self.gridgraph.step(action)
                state = nextstate
                episode_reward = episode_reward + reward
                cum_reward = cum_reward + reward
            reward_list.append(episode_reward)
        if stat:
            return cum_reward, reward_list
        else:
            return cum_reward

    def burn_in_memory(self):
        print('Start burn in...')
        state = self.gridgraph.reset()
        for i in np.arange(self.replay.burn_in):
            if i % 2000 == 0:
                print('burn in {} samples'.format(i))
            observation = self.gridgraph.state2obsv()
            action = self.gridgraph.sample()
            nextstate, reward, is_terminal, debug = self.gridgraph.step(action)
            observation_next = self.gridgraph.state2obsv()
            self.replay.append([observation, action, reward, observation_next, is_terminal])
            if is_terminal:
                state = self.gridgraph.reset()
            else:
                state = nextstate
        self.replay.is_burn_in = True
        print('Burn in finished.')

    def burn_in_memory_search(self, observationCombo, actionCombo, rewardCombo, observation_nextCombo, is_terminalCombo):
        print('Start burn in with search algorithm...')
        for i in range(len(observationCombo)):
            observation = observationCombo[i]
            action = actionCombo[i]
            reward = rewardCombo[i]
            observation_next = observation_nextCombo[i]
            is_terminal = is_terminalCombo[i]

            self.replay.append([observation, action, reward, observation_next, is_terminal])

        self.replay.is_burn_in = True
        print('Burn in with search algorithm finished.')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env', dest='env', type=str)
    parser.add_argument('--render', dest='render', type=int, default=0)
    parser.add_argument('--train', dest='train', type=int, default=1)
    parser.add_argument('--test', dest='test', type=int, default=0)
    parser.add_argument('--lookahead', dest='lookahead', type=int, default=0)
    parser.add_argument('--test_final', dest='test_final', type=int, default=0)
    parser.add_argument('--model_no', dest='model_file_no', type=str)
    return parser.parse_args()

def main(args):

    args = parse_arguments()
    environment_name = args.env

    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    keras.backend.tensorflow_backend.set_session(sess)

    model_path = '../model/'
    data_path = '../data/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    agent = DoubleDQN_Agent(environment_name, sess, render=args.render)
    if args.train == 1:
        agent.train()
    if args.test == 1:
        print(agent.test(model_file="../model/model_{}.ckpt".format(args.model_file_no)) / 20.0)
    sess.close()

if __name__ == '__main__':
    main(sys.argv)
