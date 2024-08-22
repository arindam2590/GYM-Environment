import gymnasium as gym  # Import the Gymnasium library for creating and managing environments
import numpy as np  # Import NumPy for numerical computations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting graphs
import pickle  # Import Pickle for saving and loading objects


# Define the reinforcement learning agent class
class RLAgent:
    def __init__(self, env_name, filename, is_training=True, render=True):
        """
        Initialize the RLAgent class.

        Parameters:
        - env_name: The name of the environment to create.
        - filename: The file to save/load the Q-table.
        - is_training: Flag indicating if the agent is in training mode.
        - render: Flag indicating if the environment should be rendered.
        """
        self.env_name = env_name  # Name of the environment
        self.is_training = is_training  # Boolean flag for training mode
        self.env = gym.make(env_name, render_mode='human' if render else None)  # Create the environment

        if is_training:
            print('*****Training Phase*****')
            # Initialize the Q-table with zeros during training
            self.q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        else:
            print('*****Testing Phase*****')
            # Load the Q-table from a file during testing
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)

        # Set hyperparameters
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_decay_rate = 0.001  # Rate at which exploration rate decays
        self.rng = np.random.default_rng()  # Random number generator

    # Method to save the Q-table to a file
    def save_q_table(self, filename):
        """
        Save the Q-table to a file.

        Parameters:
        - filename: The name of the file to save the Q-table.
        """
        if self.is_training:
            with open(filename, 'wb') as f:
                pickle.dump(self.q_table, f)

    # Method to choose an action based on the current state
    def choose_action(self, state):
        """
        Choose an action based on the current policy.

        Parameters:
        - state: The current state of the environment.

        Returns:
        - The chosen action.
        """
        if self.is_training and self.rng.random() < self.epsilon:
            # Exploration: choose a random action
            return self.env.action_space.sample()
        else:
            # Exploitation: choose the best action from the Q-table
            return np.argmax(self.q_table[state, :])

    # Method to update the Q-table based on the agent's experience
    def train(self, state, new_state, action, reward):
        """
        Update the Q-table based on the agent's experience.

        Parameters:
        - state: The previous state of the environment.
        - new_state: The current state of the environment.
        - action: The action taken.
        - reward: The reward received.

        Returns:
        - The temporal difference error.
        """
        # Calculate the temporal difference error
        temporal_difference = reward + self.gamma * np.max(self.q_table[new_state, :]) - self.q_table[state, action]

        # Update the Q-value for the state-action pair
        self.q_table[state, action] += self.alpha * temporal_difference

        return temporal_difference

    # Method to play the game for a specified number of episodes
    def play_game(self, episodes):
        """
        Play multiple episodes of the game, training the agent if in training mode.

        Parameters:
        - episodes: The number of episodes to play.

        Returns:
        - rewards_per_episode: Array of rewards per episode.
        - epsilon_decay_per_episode: Array of epsilon values per episode.
        - training_error: Array of temporal difference errors per episode.
        - steps_per_episode: Array of steps taken per episode.
        """
        # Initialize arrays to store performance metrics
        rewards_per_episode = np.zeros(episodes)
        epsilon_decay_per_episode = np.zeros(episodes)
        steps_per_episode = np.zeros(episodes)
        training_error = np.zeros(episodes)

        for episode in range(episodes):
            state = self.env.reset()[0]  # Reset the environment to the initial state
            done = False
            steps, total_reward = 0, 0

            while not done:
                # Choose an action based on the current state
                action = self.choose_action(state)
                # Execute the action and observe the result
                new_state, reward, terminated, truncated, _ = self.env.step(action)

                if self.is_training:
                    # Update the Q-table based on the experience
                    error = self.train(state, new_state, action, reward)

                # Move to the new state
                state = new_state
                total_reward += reward
                steps += 1
                done = terminated or truncated  # Check if the episode has ended

            # Print episode results
            print(f"Episode: {episode + 1}, Steps: {steps}, Reward: {reward}")
            print("=======================================")

            # Record performance metrics for the episode
            rewards_per_episode[episode] = total_reward
            epsilon_decay_per_episode[episode] = self.epsilon
            steps_per_episode[episode] = steps
            if self.is_training:
                training_error[episode] = error
            self.epsilon = max(self.epsilon - self.epsilon_decay_rate, 0)  # Decay the exploration rate

            if self.epsilon == 0.0:
                self.alpha = 0.0001  # Reduce the learning rate when exploration is minimal

        # Return the recorded performance metrics
        return rewards_per_episode, epsilon_decay_per_episode, training_error, steps_per_episode


# Function to plot cumulative rewards per episode
def plot_rewards(rewards, filename, episodes):
    """
    Plot the cumulative rewards over episodes.

    Parameters:
    - rewards: Array of rewards per episode.
    - filename: The name of the file to save the plot.
    - episodes: The number of episodes.
    """
    sum_rewards = np.zeros(episodes)
    window_size = 100  # Define the window size for smoothing
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards[max(0, t - window_size + 1):(t + 1)])

    # Plot and save the cumulative reward graph
    plt.plot(sum_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Cumulative Reward per 100 Episodes')
    plt.savefig(filename)
    plt.clf()


# Function to plot epsilon decay over episodes
def plot_epsilon_decay(epsilon_decay_per_episode, filename, episodes):
    """
    Plot the decay of epsilon over episodes.

    Parameters:
    - epsilon_decay_per_episode: Array of epsilon values per episode.
    - filename: The name of the file to save the plot.
    - episodes: The number of episodes.
    """
    plt.plot(range(episodes), epsilon_decay_per_episode)
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon Decay')
    plt.title('Epsilon Decay per Episode')
    plt.savefig(filename)
    plt.clf()


# Function to plot training error (temporal difference) over episodes
def plot_training_error(training_error, filename, episodes):
    """
    Plot the temporal difference error over episodes.

    Parameters:
    - training_error: Array of temporal difference errors per episode.
    - filename: The name of the file to save the plot.
    - episodes: The number of episodes.
    """
    plt.plot(range(episodes), training_error)
    plt.xlabel('Episodes')
    plt.ylabel('Temporal Difference')
    plt.title('Temporal Difference per Episode')
    plt.savefig(filename)
    plt.clf()


# Function to plot episode length over episodes
def plot_episode_length(steps, filename, episodes):
    """
    Plot the length of episodes over time.

    Parameters:
    - steps: Array of steps taken per episode.
    - filename: The name of the file to save the plot.
    - episodes: The number of episodes.
    """
    plt.plot(range(episodes), steps)
    plt.xlabel('Episodes')
    plt.ylabel('Episode Length')
    plt.title('Episode Length per Episode')
    plt.savefig(filename)


# Main function to run the RL agent
def main():
    """
        Main function to train or test the RL agent on the Taxi-v3 environment.
        """
    env_name = 'Taxi-v3'  # Define the environment name
    model_filename = 'taxi_q_table.pkl'  # Filename to save/load the Q-table
    render = False  # Boolean flag to render the environment
    is_training = True  # Boolean flag for training mode
    episodes = 10000  # Number of episodes to run

    # Initialize the RL agent
    agent = RLAgent(env_name, model_filename, is_training, render)
    # Play the game and collect performance metrics
    rewards, epsilon_decay, training_error, steps = agent.play_game(episodes)

    if is_training:
        # Plot results and save Q-table during training
        plot_filename = env_name + '_training_reward.png'
        plot_episode_len_filename = env_name + '_training_episode_len.png'
        decay_plot_filename = env_name + '_epsilon_decay.png'
        error_plot_filename = env_name + '_training_error.png'
        plot_epsilon_decay(epsilon_decay, decay_plot_filename, episodes)
        plot_training_error(training_error, error_plot_filename, episodes)
        agent.save_q_table(model_filename)
    else:
        # Plot results and calculate success rate during testing
        plot_filename = env_name + '_testing_reward.png'
        plot_episode_len_filename = env_name + '_testing_episode_len.png'
        success = 100 * np.sum(rewards) / episodes
        print('Test Success Rate:', success)

    # Plot and save reward and episode length graphs
    plot_rewards(rewards, plot_filename, episodes)
    plot_episode_length(steps, plot_episode_len_filename, episodes)


# Driver code
if __name__ == "__main__":
    main()
