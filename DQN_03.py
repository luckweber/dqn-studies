import numpy as np
import gym
import random

from collections import deque
from keras.layers import Input, Activation, Dense, Flatten, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D
from keras.models import Model
import highway_env


class Agent:

    def __init__(self, env):
        self.env = env
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.n
        self.create_model()

    def create_model(self, hidden_dims=[64, 64]):
        X = Input(shape=(self.input_dim, ))

        net = RepeatVector(self.input_dim)(X)
        net = Reshape([self.input_dim, self.input_dim, 1])(net)

        for h_dim in hidden_dims:
            net = Conv2D(h_dim, [3, 3], padding='SAME')(net)
            net = Activation('relu')(net)

        net = Flatten()(net)
        net = Dense(self.output_dim)(net)

        self.model = Model(inputs=X, outputs=net)
        self.model.compile('rmsprop', 'mse')

    def act(self, X, eps=1.0):
        if np.random.rand() < eps:
            return self.env.action_space.sample()

        X = X.reshape(-1, self.input_dim)
        Q = self.model.predict_on_batch(X)
        return np.argmax(Q, 1)[0]

    def train(self, X_batch, y_batch):
        return self.model.train_on_batch(X_batch, y_batch)

    def predict(self, X_batch):
        return self.model.predict_on_batch(X_batch)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        return self.model.set_weights(weights)


def create_batch(agent, agent_T, memory, batch_size, discount_rate):
    sample = random.sample(memory, batch_size)
    sample = np.asarray(sample)

    s = sample[:, 0]
    a = sample[:, 1].astype(np.int32)
    r = sample[:, 2]
    s2 = sample[:, 3]
    d = sample[:, 4].astype(np.int32)

    X_batch = np.vstack(s)
    y_batch = agent.predict(X_batch)

    y_batch[np.arange(batch_size), a] = r + discount_rate * np.max(agent_T.predict(np.vstack(s2)), 1) * (1 - d)

    return X_batch, y_batch


def print_info(episode, reward, eps):
    msg = f"[Episode {episode:>5}] Reward: {reward:>5} EPS: {eps:>3.2f}"
    print(msg)


def test_play(agent, env):
    s = env.reset()
    done = False
    episode_reward = 0
    while not done:
        env.render()
        a = agent.act(s, 0)
        s2, r, done, info = env.step(a)
        episode_reward += r

    print(f"Episode Reward: {episode_reward}")
    return episode_reward


def main():
    n_episode = 1000
    discount_rate = 0.95
    n_memory = 50000
    batch_size = 32
    eps = 1.0
    min_eps = 0.01

    env_name = 'highway-v0'
    env = gym.make(env_name)
    # env = gym.wrappers.Monitor(env, './records/dqn.v2')
    agent = Agent(env)
    agent_T = Agent(env)

    memory = deque()
    last_200_rewards = deque()

    for episode in range(n_episode):
        done = False
        s = env.reset()
        eps = max(min_eps, eps - 3 / n_episode)
        episode_reward = 0
        while not done:
            a = agent.act(s, eps)
            s2, r, done, info = env.step(a)
            episode_reward += r

            if done and episode_reward < 200:
                r = -100

            memory.append([s, a, r, s2, done])

            if len(memory) > n_memory:
                memory.popleft()

            if len(memory) > batch_size and last_200_rewards[-1] < 200:
                X_batch, y_batch = create_batch(agent, agent_T, memory, batch_size, discount_rate)
                agent.train(X_batch, y_batch)

            if episode % 5 == 0:
                weights = agent.get_weights()
                agent_T.set_weights(weights)

            s = s2

        print_info(episode, episode_reward, eps)

        last_200_rewards.append(episode_reward)

        if len(last_200_rewards) > 200:
            last_200_rewards.popleft()

            avg_score = np.mean(last_200_rewards)

            if avg_score > 195:
                print(f"Game Cleared in {episode}: Avg Rewards: {avg_score}")
                break

    test_play(agent, env)
    env.close()


if __name__ == '__main__':
    main()