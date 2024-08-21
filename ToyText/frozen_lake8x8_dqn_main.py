import numpy as np
import matplotlib.pyplot as plt
from rl_agent import RLAgent  # Import the RLAgent class from the rl_agent module


def plot_rewards(rewards, filename, episodes):
    """
    Plots the sum of rewards over the episodes and saves the plot as an image file.

    Args:
        rewards (list or np.array): A list or array containing rewards obtained per episode.
        filename (str): The name of the file where the plot will be saved.
        episodes (int): The number of episodes to consider for plotting.
    """
    sum_rewards = np.zeros(episodes)  # Initialize an array to hold the sum of rewards for each episode
    for t in range(episodes):
        # Sum the rewards of the last 100 episodes or up to the current episode
        sum_rewards[t] = np.sum(rewards[max(0, t - 100):(t - 1)])

    plt.plot(sum_rewards)  # Plot the summed rewards
    plt.savefig(filename)  # Save the plot to the specified file


def main():
    """
    Main function to set up the environment, agent, and handle the training or testing phase.
    """
    env_name = "FrozenLake-v1"  # Define the environment name
    episodes = 150  # Set the number of episodes to run
    render = True  # Boolean to decide whether to render the environment visually
    is_training = True  # Boolean to determine if the agent is in training mode or testing mode

    if is_training:
        plot_filename = 'frozen_lake8x8_training.png'  # Filename for saving the training plot
        print('***** Training Phase *****')  # Inform the user that the training phase is starting
    else:
        plot_filename = 'frozen_lake8x8_testing.png'  # Filename for saving the testing plot
        print('***** Testing Phase *****')  # Inform the user that the testing phase is starting

    agent = RLAgent(env_name, is_training, render)  # Initialize the RLAgent with the specified parameters
    rewards = agent.play_game(episodes)  # Run the agent through the episodes and collect rewards
    plot_rewards(rewards, plot_filename, episodes)  # Plot and save the rewards


if __name__ == '__main__':
    main()  # Execute the main function if the script is run directly
