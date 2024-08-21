import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque


class DQNModel:
    def __init__(self, is_training, state_size, action_size, gamma=0.9, alpha=0.001, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995, model_save_path='dqn_model'):
        # Initialize the DQNModel with the given parameters
        self.state_size = state_size  # Size of the state space
        self.action_size = action_size  # Size of the action space
        self.replay_buffer = deque(maxlen=5000)  # Experience replay buffer
        self.gamma = gamma  # Discount factor for future rewards
        self.alpha = alpha  # Learning rate for the optimizer
        self.epsilon = epsilon  # Exploration rate for the epsilon-greedy policy
        self.epsilon_min = epsilon_min  # Minimum exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for epsilon
        self.update_rate = 2  # Frequency of updating the target network

        # Build the main network if in training mode, otherwise load a pre-trained model
        self.main_network = self._build_network() if is_training else tf.keras.models.load_model(
            model_save_path + '/dqn_frozenlake_model.keras')
        # Build the target network and initialize it with the same weights as the main network
        self.target_network = self._build_network()
        self.target_network.set_weights(self.main_network.get_weights())

    def _build_network(self):
        # Build a simple neural network with two hidden layers of 24 neurons each
        model = tf.keras.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))  # First hidden layer
        model.add(layers.Dense(24, activation='relu'))  # Second hidden layer
        model.add(layers.Dense(self.action_size, activation='linear'))  # Output layer with one output per action
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha), loss='mse')  # Compile the model
        return model

    def remember(self, state, action, reward, next_state, done):
        # Store an experience in the replay buffer
        self.replay_buffer.append((state, action, reward, next_state, done))

    def update_target_network(self):
        # Update the target network to have the same weights as the main network
        self.target_network.set_weights(self.main_network.get_weights())

    def train(self, batch_size):
        # Train the main network using a batch of experiences from the replay buffer
        minibatch_indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        for idx in minibatch_indices:
            state, action, reward, next_state, done = self.replay_buffer[idx]

            # Calculate the target Q-value
            if not done:
                target_Q = reward + self.gamma * np.amax(
                    self.target_network.predict(np.reshape(next_state, [1, self.state_size]), verbose=0)[0])
            else:
                target_Q = reward

            # Predict Q-values for the current state and update the Q-value for the selected action
            Q_values = self.main_network.predict(np.reshape(state, [1, self.state_size]), verbose=0)
            Q_values[0][action] = target_Q

            # Train the main network on the updated Q-values
            self.main_network.fit(np.reshape(state, [1, self.state_size]), Q_values, epochs=1, verbose=0)

        # Decay the exploration rate after each training session
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state):
        # Choose an action based on the epsilon-greedy policy
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)  # Explore: choose a random action
        state = np.reshape(state, [1, self.state_size])
        Q_values = self.main_network.predict(state, verbose=0)  # Exploit: choose the best action
        return np.argmax(Q_values[0])  # Return the action with the highest Q-value
