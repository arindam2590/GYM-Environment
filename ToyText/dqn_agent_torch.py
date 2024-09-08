import os
import torch
import numpy as np
import gymnasium as gym
from dqn_torch import DQNModel


class DQNAgent:
    def __init__(self, env_name, is_training, render=False, batch_size=32, model_save_path='dqn_model'):
        self.env_name = env_name
        self.is_training = is_training

        self.env = gym.make(env_name, map_name="4x4", is_slippery=True, render_mode='human' if render else None)
        self.state_size = self.env.observation_space.n
        self.action_size = self.env.action_space.n

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f'Info: GPU is available...')
        else:
            self.device = torch.device("cpu")
            print(f'Info: CPU is available...')

        self.batch_size = batch_size
        self.dqn_model = DQNModel(self.env, self.device, self.is_training)
        self.model_save_path = model_save_path
        self.model_filename = None

    def play_game(self, episodes):
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        self.model_filename = '/frozen_lake_dql_torch.pt'

        print(f'Size of the State Space: {self.state_size}')
        print(f'Size of the Action Space: {self.action_size}')

        self.train_agent(episodes) if self.is_training else self.test_agent(50)

        self.env.close()

    def train_agent(self, episodes):
        Total_rewards_per_episode = np.zeros(episodes)

        for episode in range(episodes):
            state, info = self.env.reset()
            done, reward, steps = False, 0, 0

            while not done:
                action = self.dqn_model.act(state)
                new_state, reward, terminated, truncated, _ = self.env.step(action.item())

                self.dqn_model.remember(state, action, reward, new_state, done)
                state = new_state
                steps += 1
                done = terminated or truncated

            if reward == 1:
                Total_rewards_per_episode[episode] = reward

            if len(self.dqn_model.replay_buffer.buffer) > self.batch_size:
                self.dqn_model.train(self.batch_size)

            print(f"Episode {episode + 1}/{episodes} - Steps: {steps}, Reward: {Total_rewards_per_episode[episode]}, "
                  f"Epsilon: {self.dqn_model.epsilon:.3f}")

            self.dqn_model.epsilon = max(self.dqn_model.epsilon - self.dqn_model.epsilon_decay,
                                         self.dqn_model.epsilon_min)

            if episode % self.dqn_model.update_rate == 0:
                self.dqn_model.update_target_network()

        torch.save(self.dqn_model.main_network.state_dict(), self.model_save_path + self.model_filename)

    def test_agent(self, episodes):
        self.dqn_model = DQNModel(self.env, self.device, self.is_training)
        self.dqn_model.main_network.load_state_dict(torch.load(self.model_save_path + self.model_filename,
                                                               weights_only=True))
        self.dqn_model.main_network.eval()

        for episode in range(episodes):
            state, info = self.env.reset()
            done, reward = False, 0

            while not done:
                with torch.no_grad():
                    action = self.dqn_model.main_network(
                                                            self.dqn_model.encode_state(state).to(self.device)
                                                        ).argmax().view(1, 1)

                new_state, reward, terminated, truncated, _ = self.env.step(action.item())
                done = terminated or truncated
            print(f"Episode {episode + 1}/{episodes}, Reward: {reward}")

