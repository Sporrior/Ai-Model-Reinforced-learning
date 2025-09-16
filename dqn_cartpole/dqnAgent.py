import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from collections import deque
from model import QNetwork

class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, epsilon=1.0, epsilon_min=0.1,
                 epsilon_decay=0.995, lr=0.001, batch_size=64, memory_size=10000, save_path="dqn_checkpoint.pth"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.model = QNetwork(state_dim, action_dim)
        self.target_model = QNetwork(state_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.save_path = save_path

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = self.model(states)
        next_q_values = self.target_model(next_states)
        target_q_values = rewards + self.gamma * torch.max(next_q_values, dim=1)[0] * (1 - dones)

        loss = nn.MSELoss()(q_values.gather(1, actions.unsqueeze(1)), target_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # --- Save / Load functions (PyTorch 2.6+ safe) ---
    def save(self):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "target_model_state_dict": self.target_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon
        }
        torch.save(checkpoint, self.save_path)
        print(f"Checkpoint saved to {self.save_path}")

    def load(self):
        if os.path.exists(self.save_path):
            checkpoint = torch.load(self.save_path, weights_only=True)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epsilon = checkpoint["epsilon"]
            print(f"Checkpoint loaded from {self.save_path}")
        else:
            print("No checkpoint found, starting from scratch.")
