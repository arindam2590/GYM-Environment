import os
import numpy as np
import gymnasium as gym
from dqn import DQNModel


class RLAgent:
    def __init__(self, env_name, is_training=True, render=True, batch_size=32, model_save_path='dqn_model'):
        # Initialize the reinforcement learning agent with environment name, training mode,
        # rendering option, batch size for training, and model save path.
        self.env_name = env_name
        self.is_training = is_training

        # Create the environment using the specified environment name.
        # The environment is configured with a 4x4 grid map and slippery surface.
        # The rendering mode is set to 'human' if rendering is enabled.
        self.env = gym.make(env_name, map_name="4x4", is_slippery=True, render_mode='human' if render else None)

        # Get the size of the state space (number of possible states) and action space (number of possible actions).
        self.state_size = self.env.observation_space.n
        self.action_size = self.env.action_space.n

        # Set the batch size for training and initialize the DQN model.
        self.batch_size = batch_size
        self.dqn_model = DQNModel(is_training, self.state_size, self.action_size)

        # Set the path where the trained model will be saved.
        self.model_save_path = model_save_path

    def encode_state(self, state):
        # Encode the state as a one-hot vector.
        # This converts the state (an integer) into a binary vector of length state_size.
        one_hot_state = np.zeros(self.state_size)
        one_hot_state[state] = 1
        return one_hot_state

    def reset(self):
        # Reset the environment to the initial state and encode it as a one-hot vector.
        state = self.env.reset()[0]
        return self.encode_state(state)

    def play_game(self, episodes):
        # Create the directory for saving the model if it doesn't exist.
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        # Print the size of the state and action spaces for reference.
        print(f'Size of the State Space: {self.state_size}')
        print(f'Size of the Action Space: {self.action_size}')

        # Initialize an array to store total rewards for each episode.
        Total_rewards_per_episode = np.zeros(episodes)

        # Loop over each episode.
        for episode in range(episodes):
            # Reset the environment and get the initial state.
            state = self.reset()
            done, rewards_per_episode, step = False, 0, 0

            # Continue taking actions until the episode is done.
            while not done:
                # Update the target network periodically during training.
                if step % self.dqn_model.update_rate == 0 and self.is_training:
                    self.dqn_model.update_target_network()

                # Select an action using the DQN model.
                action = self.dqn_model.act(state)

                # Take the selected action in the environment and observe the new state and reward.
                new_state, reward, terminated, truncated, _ = self.env.step(action)

                # Encode the new state as a one-hot vector.
                new_state = self.encode_state(new_state)

                # Store the experience (state, action, reward, new state, done) in the replay buffer.
                self.dqn_model.remember(state, action, reward, new_state, done)

                # Update the state to the new state.
                state = new_state
                rewards_per_episode += reward
                step += 1

                # Check if the episode has ended (either terminated or truncated).
                done = terminated or truncated

                # Train the DQN model if enough experiences are stored in the replay buffer and in training mode.
                if len(self.dqn_model.replay_buffer) > self.batch_size and self.is_training:
                    self.dqn_model.train(self.batch_size)

            # Print the episode summary including steps, total reward, and epsilon value.
            print(
                f"Episode {episode + 1}/{episodes} - Steps: {step}, Total Reward: {rewards_per_episode}, Epsilon: {self.dqn_model.epsilon:.2f}")

            # Store the total rewards for the episode.
            Total_rewards_per_episode[episode] = rewards_per_episode

        # Save the trained model after all episodes are completed.
        self.dqn_model.main_network.save(self.model_save_path + '/dqn_frozenlake_model.keras')

        return Total_rewards_per_episode
