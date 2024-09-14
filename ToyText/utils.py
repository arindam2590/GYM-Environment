import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DataVisualization:
    def __init__(self, episodes, returns, steps, training_error=0.0, epsilon_decay=0.0):
        with open('config.json', 'r') as file:
            params = json.load(file)

        self.fig_dir = params['fig_dir']
        if not os.path.exists(self.fig_dir):
            os.makedirs(self.fig_dir)
        self.data_filename = params['excel_filename']
        self.n_episodes = episodes
        self.returns = returns
        self.steps = steps
        self.training_error = training_error
        self.epsilon_decay_history = epsilon_decay

    def save_data(self):
        filepath = self.fig_dir + self.data_filename
        df = pd.DataFrame({'Rewards': self.returns,
                           'Steps': self.steps,
                           'Epsilon Decay': self.epsilon_decay_history,
                           'Training Error': self.training_error})

        if not os.path.isfile(filepath):
            with pd.ExcelWriter(filepath, mode='w') as writer:
                df.to_excel(writer)

        else:
            with pd.ExcelWriter(filepath, mode='a') as writer:
                df.to_excel(writer)

    def plot_rewards(self, filename):
        plot_filename = self.fig_dir + '/FrozenLake-v1_' + '_torch' + filename

        sum_rewards = np.zeros(self.n_episodes)
        #print(type(self.returns))
        for episode in range(self.n_episodes):
            sum_rewards[episode] = np.sum(self.returns[0:(episode + 1)])

        # Plot and save the cumulative reward graph
        plt.plot(sum_rewards)
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.title('Cumulative Reward per Episodes')
        plt.savefig(plot_filename)
        plt.clf()

    def plot_episode_length(self, filename):
        plot_filename = self.fig_dir + '/FrozenLake-v1_' + '_torch' + filename
        plt.plot(range(self.n_episodes), self.steps)
        plt.xlabel('Episodes')
        plt.ylabel('Episode Length')
        plt.title('Episode Length per Episode')
        plt.savefig(plot_filename)
        plt.clf()

    def plot_epsilon_decay(self, filename):
        plot_filename = self.fig_dir + '/FrozenLake-v1_' + '_torch' + filename
        plt.plot(range(self.n_episodes), self.epsilon_decay_history)
        plt.xlabel('Episodes')
        plt.ylabel('Epsilon Decay')
        plt.title('Epsilon Decay per Episode')
        plt.savefig(plot_filename)
        plt.clf()

    def plot_training_error(self, filename):
        plot_filename = self.fig_dir + '/FrozenLake-v1_' + '_torch' + filename
        plt.plot(range(self.n_episodes), self.training_error)
        plt.xlabel('Episodes')
        plt.ylabel('Temporal Difference')
        plt.title('Temporal Difference per Episode')
        plt.savefig(plot_filename)
        plt.clf()
