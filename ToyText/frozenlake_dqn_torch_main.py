from dqn_agent_torch import DQNAgent
from utils import DataVisualization


def main():
    env_name = "FrozenLake-v1"
    is_slippery = False
    episodes = 25000
    render = False
    is_training = False

    agent = DQNAgent(env_name, is_training, is_slippery, render)
    returns, epsilon_decay, training_error, steps = agent.play_game(episodes)
    if is_training:
        plot_filename = 'training_reward.png'
        plot_episode_len_filename = 'training_episode_len.png'
        decay_plot_filename = 'epsilon_decay.png'
        error_plot_filename = 'training_error.png'
        
        train_data_visual = DataVisualization(episodes, returns, steps, training_error, epsilon_decay)
        train_data_visual.save_data()
        train_data_visual.plot_rewards(plot_filename)
        train_data_visual.plot_episode_length(plot_episode_len_filename)
        train_data_visual.plot_epsilon_decay(decay_plot_filename)
        train_data_visual.plot_training_error(error_plot_filename)


if __name__ == '__main__':
    main()
