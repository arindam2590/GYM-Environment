import numpy as np
import json
import torch
from torch import nn
import torch.nn.functional as F
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque([], maxlen=capacity)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        minibatch_indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        minibatch = []
        for idx in minibatch_indices:
            state, action, reward, next_state, done = self.buffer[idx]
            minibatch.append((state, action, reward, next_state, done))
        return minibatch


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.layer1 = nn.Linear(state_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class DQNModel:
    def __init__(self, env, device, is_training):
        with open('config.json', 'r') as file:
            params = json.load(file)

        self.env = env
        self.state_size = self.env.observation_space.n
        self.action_size = self.env.action_space.n
        self.device = device
        self.gamma = params['gamma']
        self.alpha = params['alpha']
        self.epsilon = params['epsilon']
        self.epsilon_min = params['epsilon_min']
        self.epsilon_decay = params['epsilon_decay']
        self.update_rate = params['update_rate']
        self.buffer_size = params['capacity']
        self.replay_buffer = ReplayBuffer(self.buffer_size)

        self.main_network = DQN(self.state_size, self.action_size).to(device=device)
        self.target_network = DQN(self.state_size, self.action_size).to(device=device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr=self.alpha)

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            action = torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                action = self.main_network(self.encode_state(state).to(self.device)).argmax().view(1, 1)
        return action

    def train(self, batch_size):
        minibatch = self.replay_buffer.sample(batch_size)
        current_q_values, target_q_values = [], []
        for state, action, reward, new_state, terminated in minibatch:
            state = self.encode_state(state).unsqueeze(0).to(self.device)
            new_state = self.encode_state(new_state).unsqueeze(0).to(self.device)
            reward = torch.tensor([reward], device=self.device)
            action = torch.tensor([action], device=self.device, dtype=torch.long)

            if terminated:
                target = reward
            else:
                with torch.no_grad():
                    target = reward + self.gamma * self.target_network(new_state).max(dim=1)[0]

            current_q = self.main_network(state).gather(1, action.view(1, 1))
            current_q_values.append(current_q)
            target_q_values.append(target)

        current_q_tensor = torch.cat(current_q_values).squeeze()
        target_q_tensor = torch.cat(target_q_values).squeeze()
        loss = self.loss_fn(current_q_tensor, target_q_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def encode_state(self, state):
        one_hot_tensor = torch.zeros(self.state_size, device=self.device)
        one_hot_tensor[state] = 1
        return one_hot_tensor

    def update_target_network(self):
        self.target_network.load_state_dict(self.main_network.state_dict())
