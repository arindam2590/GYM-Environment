import gymnasium as gym  # Import the gymnasium module for environment management
import numpy as np  # Import numpy for numerical operations
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import pickle  # Import pickle for saving and loading objects


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
        self.env_name = env_name
        self.is_training = is_training
        self.env = gym.make(env_name, map_name="4x4", is_slippery=True, render_mode='human' if render else None)

        if is_training:
            print('***** Training Phase *****')
            # Initialize Q-table with zeros
            self.q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        else:
            print('***** Testing Phase *****')
            # Load pre-trained Q-table from file
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)

        # Set hyperparameters
        self.alpha = 0.01  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay_rate = 0.001  # Rate of epsilon decay
        self.rng = np.random.default_rng()  # Random number generator

    def save_q_table(self, filename):
        """
        Save the Q-table to a file.

        Parameters:
        - filename: The name of the file to save the Q-table.
        """
        if self.is_training:
            with open(filename, 'wb') as f:
                pickle.dump(self.q_table, f)

    def choose_action(self, state):
        """
        Choose an action based on the current policy.

        Parameters:
        - state: The current state of the environment.

        Returns:
        - The chosen action.
        """
        if self.is_training and self.rng.random() < self.epsilon:
            # Explore: choose a random action
            return self.env.action_space.sample()
        else:
            # Exploit: choose the action with the highest Q-value
            return np.argmax(self.q_table[state, :])

    def train(self, state, action, reward, new_state, new_action):
        """
        Update the Q-table based on the agent's experience using the SARSA algorithm.

        Parameters:
        - state: The previous state of the environment.
        - action: The action taken in the previous state.
        - reward: The reward received after taking the action.
        - new_state: The current state of the environment.
        - new_action: The action taken in the new state.

        Returns:
        - The temporal difference error.
        """
        # Calculate the temporal difference error using the SARSA update rule
        temporal_difference = reward + self.gamma * self.q_table[new_state, new_action] - self.q_table[state, action]

        # Update the Q-value for the state-action pair
        self.q_table[state, action] += self.alpha * temporal_difference

        return temporal_difference

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
        # Initialize arrays to track performance metrics
        rewards_per_episode = np.zeros(episodes)
        epsilon_decay_per_episode = np.zeros(episodes)
        steps_per_episode = np.zeros(episodes)
        training_error = np.zeros(episodes)

        for episode in range(episodes):
            state = self.env.reset()[0]  # Reset environment and get the initial state
            action = self.choose_action(state)  # Choose an initial action
            done = False
            steps, total_reward = 0, 0

            while not done:
                new_state, reward, terminated, truncated, _ = self.env.step(action)  # Take the action and observe results
                new_action = self.choose_action(new_state)  # Choose the next action

                if self.is_training:
                    error = self.train(state, action, reward, new_state, new_action)  # Train the agent using SARSA

                state, action = new_state, new_action  # Update the state and action for the next step
                total_reward += reward
                steps += 1
                done = terminated or truncated  # Check if episode is done

            # Print episode summary
            print(f"Episode: {episode + 1}, Steps: {steps}, Reward: {reward}")
            print("=======================================")

            # Record metrics
            rewards_per_episode[episode] = total_reward
            epsilon_decay_per_episode[episode] = self.epsilon
            steps_per_episode[episode] = steps
            if self.is_training:
                training_error[episode] = error
            # Decay epsilon
            self.epsilon = max(self.epsilon - self.epsilon_decay_rate, 0)

            if self.epsilon == 0.0:
                self.alpha = 0.0001  # Reduce learning rate when exploration stops

        return rewards_per_episode, epsilon_decay_per_episode, training_error, steps_per_episode


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


def main():
    """
    Main function to train or test the RL agent on the FrozenLake-v1 environment.
    """
    env_name = "FrozenLake-v1"
    model_filename = 'frozenlake_q_table.pkl'
    render = False  # Set to True to render the environment
    is_training = False  # Set to False for testing
    episodes = 20  # Number of episodes to run

    # Initialize the agent
    agent = RLAgent(env_name, model_filename, is_training, render)
    # Play the game and collect performance metrics
    rewards, epsilon_decay, training_error, steps = agent.play_game(episodes)

    if is_training:
        # Plot results and save the Q-table
        plot_filename = env_name + '_training_reward_sarsa.png'
        plot_episode_len_filename = env_name + '_training_episode_len_sarsa.png'
        decay_plot_filename = env_name + '_epsilon_decay_sarsa.png'
        error_plot_filename = env_name + '_training_error_sarsa.png'
        plot_epsilon_decay(epsilon_decay, decay_plot_filename, episodes)
        plot_training_error(training_error, error_plot_filename, episodes)
        agent.save_q_table(model_filename)
    else:
        # Plot testing results and calculate success rate
        plot_filename = env_name + '_testing_reward_sarsa.png'
        plot_episode_len_filename = env_name + '_testing_episode_len_sarsa.png'
        success = 100 * np.sum(rewards) / episodes
        print('Test Success Rate:', success)

    # Plot and save graphs for rewards and episode length
    plot_rewards(rewards, plot_filename, episodes)
    plot_episode_length(steps, plot_episode_len_filename, episodes)


# Driver code
if __name__ == "__main__":
    main()
