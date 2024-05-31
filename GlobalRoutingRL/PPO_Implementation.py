#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, sys, copy, argparse, random
import matplotlib.pyplot as plt
import math
import os

np.random.seed(10701)
tf.compat.v1.set_random_seed(10701)
random.seed(10701)

class ActorCriticNetwork():
    def __init__(self, environment_name, networkname, trainable):
        if environment_name == 'grid':
            self.nObservation = 12
            self.nAction = 6
            self.learning_rate = 0.0001
            self.architecture = [32, 64, 32]

        kernel_init = tf.random_uniform_initializer(-0.5, 0.5)
        bias_init = tf.constant_initializer(0)
        self.input = tf.placeholder(tf.float32, shape=[None, self.nObservation], name='input')

        with tf.variable_scope(networkname):
            layer1 = tf.layers.dense(self.input, self.architecture[0], tf.nn.relu, kernel_initializer=kernel_init, bias_initializer=bias_init, name='layer1', trainable=trainable)
            layer2 = tf.layers.dense(layer1, self.architecture[1], tf.nn.relu, kernel_initializer=kernel_init, bias_initializer=bias_init, name='layer2', trainable=trainable)
            layer3 = tf.layers.dense(layer2, self.architecture[2], tf.nn.relu, kernel_initializer=kernel_init, bias_initializer=bias_init, name='layer3', trainable=trainable)
            self.policy_output = tf.layers.dense(layer3, self.nAction, kernel_initializer=kernel_init, bias_initializer=bias_init, name='policy_output', trainable=trainable)
            self.value_output = tf.layers.dense(layer3, 1, kernel_initializer=kernel_init, bias_initializer=bias_init, name='value_output', trainable=trainable)

        self.action = tf.placeholder(tf.int32, shape=[None], name='action')
        self.advantage = tf.placeholder(tf.float32, shape=[None], name='advantage')
        self.target_value = tf.placeholder(tf.float32, shape=[None], name='target_value')
        self.old_policy = tf.placeholder(tf.float32, shape=[None, self.nAction], name='old_policy')

        if trainable:
            self.action_onehot = tf.one_hot(self.action, self.nAction)
            self.policy = tf.nn.softmax(self.policy_output)
            self.old_policy_prob = tf.reduce_sum(self.old_policy * self.action_onehot, axis=1)
            self.policy_prob = tf.reduce_sum(self.policy * self.action_onehot, axis=1)

            self.ratio = self.policy_prob / (self.old_policy_prob + 1e-10)
            self.clip_ratio = tf.clip_by_value(self.ratio, 1.0 - 0.2, 1.0 + 0.2)

            self.policy_loss = -tf.reduce_mean(tf.minimum(self.ratio * self.advantage, self.clip_ratio * self.advantage))
            self.value_loss = tf.reduce_mean(tf.square(self.target_value - self.value_output))
            self.loss = self.policy_loss + 0.5 * self.value_loss - 0.01 * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.policy_output, labels=self.policy))

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

        self.init = tf.global_variables_initializer()

class PPOAgent():
    def __init__(self, environment_name, sess, gridgraph, render=False):
        self.gamma = 0.95
        self.lamda = 0.95
        self.max_episodes = 10000
        self.batch_size = 32
        self.render = render

        self.actor_critic = ActorCriticNetwork(environment_name, 'ppo', trainable=True)
        self.sess = sess
        self.gridgraph = gridgraph

        self.sess.run(self.actor_critic.init)
        self.saver = tf.train.Saver(max_to_keep=20)

    def policy(self, observation):
        policy = self.sess.run(self.actor_critic.policy, feed_dict={self.actor_critic.input: observation})
        return policy

    def value(self, observation):
        value = self.sess.run(self.actor_critic.value_output, feed_dict={self.actor_critic.input: observation})
        return value

    def sample_action(self, policy):
        return np.random.choice(len(policy[0]), p=policy[0])

    def train(self, twoPinNum, twoPinNumEachNet, netSort, savepath, model_file=None):
        if model_file is not None:
            self.saver.restore(self.sess, model_file)

        reward_log = []
        for episode in np.arange(self.max_episodes * len(self.gridgraph.twopin_combo)):
            state, reward_plot, is_best = self.gridgraph.reset()
            states, actions, rewards, values, old_policies, advantages = [], [], [], [], [], []
            rewards_pure = reward_plot - self.gridgraph.posTwoPinNum * 100

            is_terminal = False
            while not is_terminal:
                observation = self.gridgraph.state2obsv()
                policy = self.policy(observation)
                value = self.value(observation)
                action = self.sample_action(policy)

                nextstate, reward, is_terminal, debug = self.gridgraph.step(action)
                observation_next = self.gridgraph.state2obsv()

                states.append(observation)
                actions.append(action)
                rewards.append(reward)
                values.append(value)
                old_policies.append(policy)

                state = nextstate

            next_value = 0 if is_terminal else self.value(observation_next)
            returns, advantages = self.compute_gae(rewards, values, next_value)

            states = np.vstack(states)
            actions = np.array(actions)
            returns = np.array(returns)
            advantages = np.array(advantages)
            old_policies = np.vstack(old_policies)

            for _ in range(10):  # PPO epochs
                for start in range(0, len(states), self.batch_size):
                    end = start + self.batch_size
                    batch_states = states[start:end]
                    batch_actions = actions[start:end]
                    batch_returns = returns[start:end]
                    batch_advantages = advantages[start:end]
                    batch_old_policies = old_policies[start:end]

                    feed_dict = {
                        self.actor_critic.input: batch_states,
                        self.actor_critic.action: batch_actions,
                        self.actor_critic.advantage: batch_advantages,
                        self.actor_critic.target_value: batch_returns,
                        self.actor_critic.old_policy: batch_old_policies
                    }

                    self.sess.run(self.actor_critic.train_op, feed_dict=feed_dict)

            reward_log.append(sum(rewards))
            if episode % 100 == 0:
                save_path = self.saver.save(self.sess, "{}/model_{}.ckpt".format(savepath, episode))
                print("Model saved in path: %s" % save_path)

        return reward_log

    def compute_gae(self, rewards, values, next_value):
        values = np.append(values, next_value)
        advantages = np.zeros(len(rewards))
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] - values[i]
            gae = delta + self.gamma * self.lamda * gae
            advantages[i] = gae
        returns = advantages + values[:-1]
        return returns, advantages

    def test(self, model_file=None, no=20):
        if model_file is not None:
            self.saver.restore(self.sess, model_file)
        reward_list = []
        for episode in np.arange(no):
            state = self.gridgraph.reset()
            is_terminal = False
            episode_reward = 0
            while not is_terminal:
                observation = self.gridgraph.state2obsv()
                policy = self.policy(observation)
                action = self.sample_action(policy)
                nextstate, reward, is_terminal, debug = self.gridgraph.step(action)
                state = nextstate
                episode_reward += reward
            reward_list.append(episode_reward)
        return np.mean(reward_list)

def parse_arguments():
    parser = argparse.ArgumentParser(description='PPO Argument Parser')
    parser.add_argument('--env', dest='env', type=str)
    parser.add_argument('--render', dest='render', type=int, default=0)
    parser.add_argument('--train', dest='train', type=int, default=1)
    parser.add_argument('--test', dest='test', type=int, default=0)
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

    agent = PPOAgent(environment_name, sess, render=args.render)
    if args.train == 1:
        reward_log = agent.train()
        np.save(os.path.join(data_path, 'reward_log.npy'), reward_log)
    if args.test == 1:
        reward = agent.test(model_file="../model/model_{}.ckpt".format(args.model_file_no))
        print(f"Average reward over {args.test} episodes: {reward}")
    sess.close()

if __name__ == '__main__':
    main(sys.argv)
