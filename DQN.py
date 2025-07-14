import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class DQNNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super(DQNNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(
        self,
        obs_dim,
        action_dim,
        device="cpu",
        hidden_dim=128,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=500
    ):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        self.policy_net = DQNNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.target_net = DQNNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def choose_action(self, obs, eval_mode=False):
        """
        obs: torch.Tensor or np.ndarray, shape [batch_dim, obs_dim]
        Returns: torch.LongTensor, shape [batch_dim, 1]
        """
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        batch_size = obs.shape[0]
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon - self.epsilon_end) * \
            np.exp(-1. * self.steps_done / self.epsilon_decay)
        if not eval_mode:
            self.steps_done += 1
            self.epsilon = eps_threshold

        if sample > eps_threshold or eval_mode:
            with torch.no_grad():
                q_values = self.policy_net(obs)
                actions = q_values.argmax(dim=1, keepdim=True)
        else:
            actions = torch.randint(0, self.action_dim, (batch_size, 1), device=self.device)
        return actions

    def update(self, batch):
        """
        batch: dict with keys 'obs', 'action', 'reward', 'next_obs', 'done'
        All values are torch tensors.
        """
        obs = batch['obs'].to(self.device)
        action = batch['action'].to(self.device)
        reward = batch['reward'].to(self.device)
        next_obs = batch['next_obs'].to(self.device)
        done = batch['done'].to(self.device)

        q_values = self.policy_net(obs).gather(1, action)
        with torch.no_grad():
            next_q_values = self.target_net(next_obs).max(1, keepdim=True)[0]
            target = reward + (1 - done) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())